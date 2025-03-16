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
import re

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

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
                all_images = sorted(all_images)
                
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
        
class ValDataset(Dataset):
    def __init__(self, im_path='', image_size=None, mean_std=IMAGENET_MEAN_STD):
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.input_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        self.image_size = image_size
        # 调用 __getdata 方法获取所有图片路径
        self.data = self.__getdata__(im_path)
    
    def __getdata__(self, root_dir):
        """
        遍历 root_dir 下的所有子文件夹，每个子文件夹代表一个底图，
        对每个子文件夹：
          - 查找其中的 xlsx 文件（包含标注框坐标）
          - 查找以 "w" 开头且是 jpg/jpeg/png 的图片（对应 UAV 图片）
          - 按照文件名中 "w数字" 分组
          - 列表中每个样本的数据结构为：
                {
                  "uav": [图片路径列表],
                  "label": {
                      "coords": (d1, d2),
                      "base_img": 子文件夹名称（代表底图）
                  }
                }
        """
        data = {}
        index = 0
        # 遍历所有子文件夹
        for sub_folder in os.listdir(root_dir):
            sub_folder_path = os.path.join(root_dir, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue
            
            image_paths = []
            xlsx_file = None  # 每个子文件夹中 xlsx 文件的路径
            # 遍历子文件夹中的所有文件
            for file_name in os.listdir(sub_folder_path):
                if file_name.endswith('.xlsx'):
                    xlsx_file = os.path.join(sub_folder_path, file_name)
                elif file_name.startswith("w") and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(sub_folder_path, file_name))
            
            if xlsx_file is None:
                # 如果没有 xlsx 文件，则跳过该文件夹
                continue
            
            # 按照文件名中第一个分割部分 (例如 "w1") 分组
            groups = {}
            for img_path in image_paths:
                base = os.path.basename(img_path)
                parts = base.split("-")
                if len(parts) < 2:
                    continue  # 文件名格式不符合预期时跳过
                place_key = parts[0]  # 例如 "w1"
                groups.setdefault(place_key, []).append(img_path)
            
            # 读取 xlsx 文件，获取每个组的坐标信息
            coords_map = self.__read_coords__(xlsx_file)
            for key, group_images in groups.items():
                # 获取该组的 d1 和 d2 坐标信息
                coord = coords_map.get(key, ((None, None), (None, None)))
                # 将底图信息（子文件夹名称）也写入 label 中
                data[index] = {
                    "uav": group_images,
                    "label": {
                        "coords": coord,
                        "base_img": sub_folder  # 该子文件夹即代表底图
                    }
                }
                index += 1
        return data
    
    def __read_coords__(self, xlsx_file):
        """
        读取 xlsx 文件，解析每行记录。
        假设 xlsx 文件中有列 '组标名称'、'x' 和 'y'：
            '组标名称' 格式为 "w1-d1" 或 "w1-d2"
            对应 'x' 和 'y' 列分别表示该点的横纵坐标
        根据相同的 w 标识，分别保存 d1（左上坐标）和 d2（右下坐标）
        """
        df = pd.read_excel(xlsx_file)
        # 临时字典，用于按基础标识整理 d1 和 d2
        coords_temp = {}
        pattern = re.compile(r'^(w\d+)-(d[12])$')
        for _, row in df.iterrows():
            group_name = str(row['坐标名称'])
            m = pattern.match(group_name)
            if not m:
                continue

            parts = group_name.split('-')
            if len(parts) == 2:
                base_id, pos = parts  # pos 可能为 "d1" 或 "d2"
                coordinate = (row['x'], row['y'])
                if base_id not in coords_temp:
                    coords_temp[base_id] = {}
                coords_temp[base_id][pos] = coordinate
        # 生成最终的映射字典：base_id -> (d1, d2)
        coords_map = {}
        for base_id, pos_dict in coords_temp.items():
            d1 = pos_dict.get('d1', None)
            d2 = pos_dict.get('d2', None)
            coords_map[base_id] = (d1, d2)
        return coords_map

    def __getitem__(self, index):
        """
        返回的样本现在包含：
          - 多张 UAV 图片（经过数据变换）
          - label 信息，其中包含了标注框坐标和该样本所属的底图标识
        """
        sample = self.data[index]
        imgs = []
        for img_path in sample["uav"]:
            img = Image.open(img_path).convert('RGB')
            if self.input_transform:
                img = self.input_transform(img)
            imgs.append(img)
        label = sample["label"]
        return imgs, label

    def __len__(self):
        return len(self.data)