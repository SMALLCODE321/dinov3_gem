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
import xml.etree.ElementTree as ET


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
            aug_img = self.transform_sat(img_copy)  #消融实验
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
        
class ImageFolderDataset_Ablation(Dataset):
    def __init__(self,
                 data_path=None, # 数据集存放路径
                 img_per_place=8,
                 sat_aug_per_place=5,
                 transform_sat=None,
                 transform_drone=None     # 图像预处理变换
                 ):
        super(ImageFolderDataset_Ablation, self).__init__()
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
            subfolders = sorted(os.listdir(self.data_path))
            for subfolder in subfolders:
                subfolder_path = os.path.join(self.data_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                # 收集所有图片
                all_images = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    all_images.extend(glob.glob(os.path.join(subfolder_path, ext)))
                if not all_images:
                    continue
                all_images = sorted(all_images)
                
                # 按前缀分组
                groups = {}
                for img in all_images:
                    base = os.path.basename(img)
                    parts = base.split("-")
                    if len(parts) < 2:
                        continue
                    place_key = parts[0]
                    groups.setdefault(place_key, []).append(img)

                for key, group_images in groups.items():
                    satellite_imgs = [img for img in group_images if img.lower().endswith("-0.jpg")]
                    if not satellite_imgs:
                        continue
                    satellite_img = satellite_imgs[0]
                    uav_imgs = [img for img in group_images 
                                if any(sub in img.lower() for sub in ["-70.jpg","-75.jpg","-80.jpg","-82.jpg","-85.jpg"])]
                    if not uav_imgs:
                        continue
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
        
        sat_img = self.image_loader(sat_img_path)
        sat_base = self.transform_drone(sat_img) if self.transform_drone else sat_img
        
        aug_sats = []
        for i in range(self.sat_aug_per_place):
            img_copy = sat_img.copy()
            aug_img = self.transform_sat(img_copy) if self.transform_sat else img_copy
            aug_sats.append(aug_img)
        
        selected_uav_path = random.choice(uav_img_paths)
        uav_img = self.image_loader(selected_uav_path)
        uav_img = self.transform_drone(uav_img) if self.transform_drone else uav_img
        
        final_imgs = [sat_base] + aug_sats + [uav_img]
    
        
        stacked = torch.stack(final_imgs, dim=0)
        label = torch.tensor(entry["label"]).repeat(len(final_imgs))
        return stacked, label
    
    def __len__(self):
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded, return blank.')
            return Image.new('RGB', (224, 224))


class ValDataset(Dataset):
    def __init__(self, im_path='', image_size=None, mean_std=IMAGENET_MEAN_STD):
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.image_size = image_size
        self.input_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        # 调用 __getdata__ 方法获取所有样本信息
        self.data = self.__getdata__(im_path)
    
    def __getdata__(self, root_dir):
        """
        遍历 root_dir 下的所有子文件夹，每个子文件夹内包含一个 XML 文件和 UAV 图片，
        对每个子文件夹进行如下处理：
          - 根据 XML 文件获取底图路径及所有 object 的标注，
          - 针对 XML 中的每个 object，根据其名称（例如 "w1"）在当前子文件夹内查找对应 UAV 图片，
          - 将 UAV 图片列表、当前 object 对应的 bndbox 坐标、底图路径以及 object 名称保存到数据字典中。
        """
        data = {}
        index = 0
        for sub_folder in os.listdir(root_dir):
            sub_folder_path = os.path.join(root_dir, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue

            # 查找 XML 文件（假设每个子文件夹只有一个 XML 文件）
            xml_files = glob.glob(os.path.join(sub_folder_path, "*.xml"))
            if not xml_files:
                continue
            xml_file = xml_files[0]
            
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except Exception as e:
                print(f"XML文件 {xml_file} 解析失败: {e}")
                continue
            
            # 从 XML 中获取底图路径（<path> 标签内容）
            base_img_path = root.findtext("path")
            if base_img_path is None:
                print(f"XML文件 {xml_file} 缺少 <path> 标签信息")
                continue

            # 获取当前子文件夹中所有图片（假定 UAV 图片存放在此处）
            image_paths = []
            for file_name in os.listdir(sub_folder_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(sub_folder_path, file_name))
            
            # 遍历 XML 中的每个 object 节点，每个 object 对应一个样本
            for obj in root.iter("object"):
                obj_name = obj.findtext("name")
                if not obj_name:
                    continue

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                try:
                    xmin = int(bndbox.findtext("xmin"))
                    ymin = int(bndbox.findtext("ymin"))
                    xmax = int(bndbox.findtext("xmax"))
                    ymax = int(bndbox.findtext("ymax"))
                except Exception as e:
                    print(f"解析 {xml_file} 中 object {obj_name} 的 bndbox 错误: {e}")
                    continue

                # 根据 object 名称（例如 "w1"）查找对应的 UAV 图片，要求文件名以 "w1-" 开头，如 "w1-70.jpg"
                group_images = []
                for img_path in image_paths:
                    base_file = os.path.basename(img_path)
                    if base_file.startswith(obj_name + "-"):
                        suffix = base_file[len(obj_name) + 1:]  # 获取 "-" 后面的部分
                        if not suffix.startswith("0"):
                            group_images.append(img_path)
                        group_images.append(img_path)
                
                if not group_images:
                    # 如果未找到对应的 UAV 图片，则跳过该 object
                    continue

                # 保存当前 object 对应的样本信息
                data[index] = {
                    "uav": group_images,
                    "label": {
                        "bndbox": (xmin, ymin, xmax, ymax),
                        "base_img": base_img_path,
                        "obj_name": obj_name
                    }
                }
                index += 1
        
        return data

    def __getitem__(self, index):
        """
        对于每个样本：
          - 加载所有 UAV 图片，并对其应用统一的数据预处理变换；
          - 返回预处理后的 UAV 图片列表以及标签字典，
            其中标签字典包含 bndbox 坐标、底图路径和对象名称（obj_name）。
        """
        sample = self.data[index]
        imgs = []
        for img_path in sample["uav"]:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"加载图像 {img_path} 失败: {e}")
                img = Image.new('RGB', (self.image_size, self.image_size))
            if self.input_transform:
                img = self.input_transform(img)
            imgs.append(img)
        label = sample["label"]
        return imgs, label

    def __len__(self):
        return len(self.data)
        return len(self.data)