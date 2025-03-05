# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.utils as vutils
import glob
import os
import random
import numpy as np

"""
该类继承自 torch.utils.data.Dataset，用于处理图像数据集
"""
class ImageFolderDataset(Dataset):
    def __init__(self,
                 data_path=None, #数据集存放路径
                 img_per_place=8,
                 sat_aug_per_place=5,
                 transform_sat=None,
                 transform_drone=None     #图像预处理变换
                 ):
        super(ImageFolderDataset, self).__init__()
        self.data_path = data_path 
        self.img_per_place = img_per_place
        self.sat_aug_per_place = sat_aug_per_place
        self.transform_sat = transform_sat
        self.transform_drone = transform_drone        
        # 构造每个place的文件信息字典
        self.places_ids, self.total_images = self.__getdataframes()
    
    def __getdataframes(self):
        place_ids = {}
        total_images = 0
        valid_place_idx = 0
        if os.path.isdir(self.data_path):
            # 遍历 data_path 下的所有子文件夹（例如 query 文件夹下的 "1", "2", "3", ...）
            subfolders = sorted(os.listdir(self.data_path))
            for subfolder in subfolders:
                subfolder_path = os.path.join(self.data_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                # 获取当前子文件夹中所有图片文件
                all_images = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    all_images.extend(glob.glob(os.path.join(subfolder_path, ext)))
                if not all_images:
                    continue

                # 根据文件名前缀（例如 "w1", "w2", ...）分组，每组即一个 place
                groups = {}
                for img in all_images:
                    base = os.path.basename(img)
                    parts = base.split("-")
                    if len(parts) < 2:
                        continue  # 文件名不符合预期格式则跳过
                    place_key = parts[0]  # 例如 "w1"
                    groups.setdefault(place_key, []).append(img)

                # 对每个分组生成一个地点条目
                for key, group_images in groups.items():
                    # 筛选卫星图：文件名以“-0.jpg”结尾
                    satellite_imgs = [img for img in group_images if img.lower().endswith("-0.jpg")]
                    if not satellite_imgs:
                        continue  # 如果该组没有卫星图则跳过
                    satellite_img = satellite_imgs[0]
                    # 筛选无人机视图：文件名包含“-70.jpg”,“-75.jpg”,“-80.jpg”,“-82.jpg”,“-85.jpg”
                    uav_imgs = [img for img in group_images if any(sub in img.lower() for sub in ["-70.jpg", "-75.jpg", "-80.jpg", "-82.jpg", "-85.jpg"])]
                    place_ids[valid_place_idx] = {
                        "satellite": satellite_img, 
                        "uav": uav_imgs,
                        "label": valid_place_idx
                        }
                    total_images += 1 + len(uav_imgs)
                    valid_place_idx += 1
        else:
            raise NotImplementedError("data_path 应为包含各个子文件夹的目录。")
        return place_ids, total_images
    
    def __getitem__(self, index):

        entry = self.places_ids[index]
        sat_img_path = entry["satellite"]
        uav_img_paths = entry["uav"]
        
        # 加载卫星底图并应用简单仿射变换进行数据增强
        sat_img = self.image_loader(sat_img_path)
        sat_base = self.transform_drone(sat_img)

        imgs = []
        imgs.append(sat_base)
        for i in range(self.sat_aug_per_place):
        # 对原图复制后分别进行随机仿射变换，产生不同的增强版本
            img_copy = sat_img.copy()
            aug_img = self.transform_sat(img_copy)
            imgs.append(aug_img)

        # 加载无人机视图，并应用预处理
        for path in uav_img_paths:
            img = self.image_loader(path)
            if self.transform_drone is not None:
                img = self.transform_drone(img)
            imgs.append(img)

        if self.img_per_place > len(imgs):
            raise ValueError("所请求的图片数量大于可用图片数量")
        
        chosen_imgs = random.sample(imgs[1:], self.img_per_place - 1)
        final_imgs = [sat_base] + chosen_imgs
        stacked_imgs = torch.stack(final_imgs)
        label = torch.tensor(entry["label"]).repeat(len(final_imgs))
        # 返回：(卫星图像, 堆叠后的无人机图像, 当前place的标签)
        return stacked_imgs, label
    
    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    """
    加载为RGB模式
    若加载失败，将返回一个新的空白图像
    """
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))


class LTAValDataset(Dataset):
    """
    LTA 验证数据集。
      - 一张卫星基础图（文件名以“-0.jpg”结尾）
      - 多张无人机视图（文件名中包含“-70.jpg”, “-75.jpg”, “-80.jpg”, “-82.jpg”, “-85.jpg”）
    
    验证时仅对每张图像进行单次预处理（例如 Resize、Normalize 等），
    由参数 input_transform 传入，通常不包含随机性的数据增强操作。
    """
    def __init__(self, data_root, input_transform=None):
        """
        Args:
            data_root (str): 存储各个地点子文件夹的根目录
            input_transform (callable, optional): 对图像进行预处理的函数/变换
        """
        super(LTAValDataset, self).__init__()
        self.data_root = data_root
        self.input_transform = input_transform
        self.places_ids, self.total_images = self._get_places_data()

    def _get_places_data(self):
        places = {}
        total_images = 0
        valid_place_idx = 0

        if os.path.isdir(self.data_root):
            # 遍历 data_root 下的所有子文件夹（例如 query 或 train 文件夹下的 "1", "2", "3", ...）
            subfolders = sorted(os.listdir(self.data_root))
            for subfolder in subfolders:
                subfolder_path = os.path.join(self.data_root, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                # 获取当前子文件夹中所有图片文件
                all_images = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    all_images.extend(glob.glob(os.path.join(subfolder_path, ext)))
                if not all_images:
                    continue

                # 根据文件名前缀（例如 "w1", "w2", ...）分组
                groups = {}
                for img in all_images:
                    base = os.path.basename(img)
                    parts = base.split("-")
                    if len(parts) < 2:
                        continue
                    place_key = parts[0]
                    groups.setdefault(place_key, []).append(img)

                # 为每个分组构造地点条目
                for key, group_images in groups.items():
                    # 选取卫星底图：文件名以“-0.jpg”结尾
                    satellite_imgs = [img for img in group_images if img.lower().endswith("-0.jpg")]
                    if not satellite_imgs:
                        continue
                    satellite_img = satellite_imgs[0]

                    # 选取 UAV 视图，必须包含以下五张图："-70.jpg", "-75.jpg", "-80.jpg", "-82.jpg", "-85.jpg"
                    required_suffixes = ["-70.jpg", "-75.jpg", "-80.jpg", "-82.jpg", "-85.jpg"]
                    uav_imgs = []
                    for suffix in required_suffixes:
                        found_img = None
                        for img in group_images:
                            if img.lower().endswith(suffix):
                                found_img = img
                                break  # 只取第一个找到的
                        if found_img is None:
                            # 如果缺少某个必需的 UAV 图像，则跳过该地点
                            uav_imgs = []
                            break
                        uav_imgs.append(found_img)

                    # 如果 UAV 图像不足 5 张，则跳过该地点
                    if len(uav_imgs) != 5:
                        continue

                    places[valid_place_idx] = {
                        "satellite": satellite_img,
                        "uav": uav_imgs,
                        "label": valid_place_idx
                    }
                    total_images += 1 + len(uav_imgs)
                    valid_place_idx += 1
        else:
            raise NotImplementedError("data_root 应为包含各个子文件夹的目录。")
        return places, total_images

    def __getitem__(self, index):
        """
        对应某个地点，加载卫星图和所有无人机图像，并应用预处理变换。
        
        Returns:
            (sat_img, uav_imgs, label)
              sat_img: 预处理后的卫星图像
              uav_imgs: 经过 input_transform 后堆叠的无人机图像张量 (shape: [n, C, H, W])
              label: 当前地点的编号（torch.tensor 对象）
        """
        entry = self.places_ids[index]
        sat_img_path = entry["satellite"]
        uav_img_paths = entry["uav"]

        # 加载卫星图像
        sat_img = self.image_loader(sat_img_path)
        if self.input_transform is not None:
            sat_img = self.input_transform(sat_img)

        # 加载所有无人机视图并应用预处理
        imgs = []
        imgs.append(sat_img)
        for path in uav_img_paths:
            img = self.image_loader(path)
            if self.input_transform is not None:
                img = self.input_transform(img)
            imgs.append(img)
        label = torch.tensor(entry["label"]).repeat(len(imgs))

        return torch.stack(imgs), label

    def __len__(self):
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        """
        加载图像为 RGB 模式。如果加载失败则返回一个空白图像。
        """
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert("RGB")
        except Exception as e:
            print(f"加载图像 {path} 失败，异常：{e}")
            # 返回一个默认的空白图片（尺寸可根据需要调整）
            return Image.new("RGB", (224, 224))