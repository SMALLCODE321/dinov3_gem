#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np
import faiss
from tqdm import tqdm
import pandas as pd

def parse_args():
    """
    解析命令行参数，同时接收用于构建 FAISS 索引和生成图像裁剪 CSV 索引的参数
    """
    parser = argparse.ArgumentParser(
        description="同时构建 FAISS 索引和生成图像裁剪CSV索引的工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # FAISS 索引相关参数
    faiss_group = parser.add_argument_group("FAISS Index Options", "用于构建 FAISS 索引的参数")
    faiss_group.add_argument("--npy_directory", type=str, required=True,
                             help="存放描述子 .npy 文件的目录路径")
    faiss_group.add_argument("--index_file", type=str, required=True,
                             help="保存生成的 FAISS 索引的文件路径")
    faiss_group.add_argument("--index_type", type=str, default='flat', choices=['flat', 'ivf'],
                             help="索引创建方式：'flat' 使用 IndexFlatL2，'ivf' 使用 IndexIVFFlat")
    faiss_group.add_argument("--group_num", type=int, default=1,
                             help="分组数，用于分批加载 .npy 文件（group_num=1 表示一次加载所有文件）")
    
    # 图像裁剪 CSV 索引相关参数
    patches_group = parser.add_argument_group("Image Patch Options", "用于生成图像裁剪CSV索引的参数")
    patches_group.add_argument("--base_info_file", type=str, required=True,
                             help="图像基础信息文件路径，每行包含图像名称, 宽度, 高度（以逗号分隔）")
    patches_group.add_argument("--crop_size", nargs='+', type=lambda s: tuple(map(int, s.split(','))), required=True,
                             help="裁剪尺寸，格式为 高度,宽度，例如: --crop_size 1080,2048")
    patches_group.add_argument("--step_size", nargs='+', type=lambda s: tuple(map(int, s.split(','))), required=True,
                             help="裁剪步长，格式为 纵向步长,横向步长，例如: --step_size 864,1638")
    patches_group.add_argument("--index_info_file", type=str, required=True,
                             help="保存裁剪块索引信息的 CSV 文件路径")
    
    return parser.parse_args()

def create_faiss_index_for_group(file_group, group_name, index=None, index_type='flat', nlist=100):
    """
    为一个文件组创建或更新 FAISS 索引

    参数：
    - file_group: 包含 .npy 文件路径的列表，内部保存部分描述子数据。
    - group_name: 当前文件组名称，用于日志输出。
    - index: 已存在的 FAISS 索引对象；若为 None，则新建索引。
    - index_type: 索引类型，'flat' 或 'ivf'。
    - nlist: 当 index_type 为 'ivf' 时使用的聚类中心数。

    返回更新后的索引对象。
    """
    arrays = []
    for file_path in file_group:
        descriptors = np.load(file_path)
        arrays.append(descriptors)
        # 假设所有文件描述子的维度均一致
        dimension = descriptors.shape[-1]
    descriptors = np.concatenate(arrays, axis=0)
    
    if index is None:
        if index_type == 'flat':
            index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(descriptors)
        else:
            raise ValueError("Unsupported index type specified.")
    
    index.add(descriptors)
    print(f"Group {group_name}: Number of vectors in the index: {index.ntotal}")
    return index

def generate_groups(npy_directory, num_groups):
    """
    将指定目录下的所有 .npy 文件均匀分组

    参数：
    - npy_directory: .npy 文件所在目录。
    - num_groups: 分组数量。

    返回：
    - 分组后的文件列表，每个子列表包含一部分文件的路径。
    """
    all_files = sorted(glob.glob(os.path.join(npy_directory, "*.npy")))
    
    groups = []
    for i in range(num_groups):
        group = all_files[i::num_groups]
        groups.append(group)
    
    return groups

def process_faiss_index(args):
    """
    根据参数构造 FAISS 索引，并保存至指定文件
    """
    file_groups = generate_groups(args.npy_directory, args.group_num)
    index = None
    for i, group in enumerate(file_groups):
        group_name = f"group_{i+1}"
        index = create_faiss_index_for_group(group, group_name, index=index, index_type=args.index_type)
    faiss.write_index(index, args.index_file)
    print(f"Saved FAISS index to {args.index_file}")

def process_patches(args):
    """
    根据图像基础信息生成图像裁剪块 CSV 索引

    修改说明：
    - base_info_file 中保存的地理信息为左上和右下的经纬度坐标，
      分别为：左上纬度、左上经度、右下纬度、右下经度。
    - 根据图像尺寸计算像素的纬度及经度分辨率，并据此计算每个 patch 中心的经纬度。
    """
    base_info_file = args.base_info_file
    # 这里 args.crop_size 与 args.step_size 为列表，每个元素为一个二元组，取第一个传入的元组
    crop_h, crop_w = args.crop_size[0]
    step_h, step_w = args.step_size[0]
    
    patches = []
    with open(base_info_file, "r") as file:
        for line in tqdm(file, desc="处理图像信息"):
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue

            image_name = parts[0]
            W, H = int(parts[1]), int(parts[2])
            
            is_tif = image_name.lower().endswith('.tif')
            if is_tif and len(parts) >= 7:
                # 读取左上角和右下角的地理坐标信息
                lat_top = float(parts[3])
                lon_left = float(parts[4])
                lat_bottom = float(parts[5])
                lon_right = float(parts[6])
                # 计算每个像素对应的经纬度变化量
                lat_pixel_size = (lat_top - lat_bottom) / H
                lon_pixel_size = (lon_right - lon_left) / W
            else:
                lat_top = None
                lon_left = None
                lat_pixel_size = None
                lon_pixel_size = None

            # 按规定步长遍历图像区域，计算裁剪块的坐标信息
            for top in range(0, H, step_h):
                for left in range(0, W, step_w):
                    bottom = min(top + crop_h, H)
                    right = min(left + crop_w, W)
                    
                    # 若裁剪块不足指定尺寸，调整起始位置
                    top_adj = top if (bottom - top) >= crop_h else max(bottom - crop_h, 0)
                    left_adj = left if (right - left) >= crop_w else max(right - crop_w, 0)
                    
                    patch = {
                        "image_name": image_name,
                        "x1": left_adj,
                        "y1": top_adj,
                        "x2": right,
                        "y2": bottom
                    }
                    
                    # 如果图像为 tif 且提供地理信息，则计算 patch 中心点的经纬度
                    if is_tif and lat_top is not None:
                        # 计算 patch 中心在像素中的位置
                        patch_center_x = (left_adj + right) / 2.0
                        patch_center_y = (top_adj + bottom) / 2.0
                        # 根据图像坐标与地理坐标的线性关系计算中心经纬度
                        center_lat = lat_top - patch_center_y * lat_pixel_size
                        center_lon = lon_left + patch_center_x * lon_pixel_size
                        patch["center_lat"] = center_lat
                        patch["center_lon"] = center_lon

                    patches.append(patch)
    
    df = pd.DataFrame(patches)
    df.to_csv(args.index_info_file, index=False)
    print(f"Saved image patch CSV index to {args.index_info_file}")

def main():
    args = parse_args()
    # 一次性执行两部分功能：
    process_faiss_index(args)
    process_patches(args)

if __name__ == '__main__':
    main()