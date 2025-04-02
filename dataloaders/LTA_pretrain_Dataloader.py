import os
import glob
import random
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms as T
from prettytable import PrettyTable

# 标准化参数
IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}

class SatellitePatchDataset(Dataset):
    """
    基于高分辨率卫星图采样 place：
      - 每个样本从指定的卫星底图中随机裁剪一个大小为 patch_size 的补丁
      - 针对该 patch 进行一次预处理（基础版）和多次随机数据增强（sat_aug_per_place 次），
        返回堆叠后的张量及当前 place 的 label
    """
    def __init__(self,
                 data_path,           # 卫星图像路径，可以是单个文件或包含多张图的目录
                 patch_size=(322, 322),
                 num_places=1000,     # 总共采样的 place 数量
                 sat_aug_per_place=5,
                 base_transform=None,  # 基础预处理：例如 ToTensor + Normalize
                 aug_transform=None    # 数据增强 / 仿射变换
                 ):
        super(SatellitePatchDataset, self).__init__()
        self.data_path = data_path
        self.patch_size = patch_size
        self.num_places = num_places
        self.sat_aug_per_place = sat_aug_per_place
        self.base_transform = base_transform
        self.aug_transform = aug_transform

        # 处理 data_path：如果是目录则收集所有图片，否则如果是文件则直接使用
        if os.path.isdir(self.data_path):
            self.image_paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.image_paths.extend(glob.glob(os.path.join(self.data_path, ext)))
            if len(self.image_paths) == 0:
                raise ValueError("指定目录中未找到任何图片文件。")
            # 排序确保顺序固定，便于后续 label 分配和映射构造
            self.image_paths = sorted(self.image_paths)
        elif os.path.isfile(self.data_path):
            self.image_paths = [self.data_path]
        else:
            raise ValueError("data_path 指定的路径不存在。")
        
        self.num_images = len(self.image_paths)

        # 计算基础分配数量和余数
        base_count = self.num_places // self.num_images
        remainder = self.num_places % self.num_images

        # 构造索引映射：每个全局采样 index 对应一个 (image_index, local_sample_index)
        self.index_mapping = []  # [(image_index, local_sample_index), ...]
        for i in range(self.num_images):
            # 对于前 remainder 张图片，多分配一个采样
            assigned = base_count + (1 if i < remainder else 0)
            for j in range(assigned):
                self.index_mapping.append((i, j))
        # 经过此步骤，self.adjusted_num_places 应等于 num_places
        self.adjusted_num_places = len(self.index_mapping)

    def __len__(self):
        return self.adjusted_num_places

    def __getitem__(self, index):
        # 根据索引映射确定当前 sample 对应的底图和该图内的采样编号
        image_index, local_sample_index = self.index_mapping[index]
        img_path = self.image_paths[image_index]
        try:
            img = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Image {img_path} 无法加载，将替换为一张空白图像。")
            img = Image.new('RGB', (self.patch_size[0] * 2, self.patch_size[1] * 2))

        W, H = img.size
        crop_w, crop_h = self.patch_size
        if W < crop_w or H < crop_h:
            raise ValueError(f"图片 {img_path} 的尺寸 {img.size} 小于组成 patch 的尺寸 {self.patch_size}")

        # 可选：通过固定 seed 保证同一 place 每次采样出相同区域
        # random.seed(hash((image_index, local_sample_index)))
        left = random.randint(0, W - crop_w) 
        top = random.randint(0, H - crop_h)
        patch = img.crop((left, top, left + crop_w, top + crop_h)) 

        # 基础预处理
        if self.base_transform is not None:
            base_patch = self.base_transform(patch)
        else:
            base_patch = T.ToTensor()(patch)
        
        # 使用 torchvision.transforms.RandomResizedCrop 定义随机尺度裁剪
        # random_resized_crop = T.RandomResizedCrop(size=self.patch_size, scale=(0.25, 1.0))
        # 多次随机数据增强
        aug_patches = []
        for _ in range(self.sat_aug_per_place):
            patch_copy = patch.copy()  # 防止 in-place 修改
            # resized_patch = random_resized_crop(patch_copy)
            if self.aug_transform is not None:
                aug_patch = self.aug_transform(patch_copy)
            else:
                aug_patch = T.ToTensor()(patch_copy)
            aug_patches.append(aug_patch)
        
        # 合并所有版本：第一版为基础版，其余为增强版
        final_patches = [base_patch] + aug_patches  # 总共 (1 + sat_aug_per_place) 个版本
        stacked_patches = torch.stack(final_patches)  # shape: [1+sat_aug_per_place, C, H, W]

        label = torch.tensor(index).repeat(len(final_patches))
        return stacked_patches, label

    
class SatelliteImageDataModule(pl.LightningDataModule):
    """
    Lightning DataModule 示例，只需传入高分辨率卫星图的路径（可以是文件或目录）。
    每个样本会随机从卫星图中裁剪出一个 patch，并生成多组仿射变换版本。
    """
    def __init__(self,
                 batch_size=32,
                 image_size=(322, 322),
                 patch_size=None,
                 num_places=1000,
                 sat_aug_per_place=5,
                 num_workers=4,
                 data_path=None,
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_places = num_places
        self.sat_aug_per_place = sat_aug_per_place
        self.num_workers = num_workers
        self.data_path = data_path
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']

        self.save_hyperparameters()
        
        # 定义基础预处理（只做 ToTensor 和标准化，可根据需要添加 Resize 等操作）
        self.base_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        
        # 定义仿射增强预处理：
        self.aug_transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.7, 1.0)),   #放在数据增广之前，750×750再进行crop，多尺度patch
            T.RandomRotation(degrees=360),
            T.RandomPerspective(distortion_scale=0.7, p=0.7, interpolation=3),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])
        
        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': True
        }
        
    def setup(self, stage=None):
        # 初始化训练数据集
        self.train_dataset = SatellitePatchDataset(
            data_path=self.data_path,
            patch_size=self.patch_size,
            num_places=self.num_places,
            sat_aug_per_place=self.sat_aug_per_place,
            base_transform=self.base_transform,
            aug_transform=self.aug_transform
        )
    
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)
    
    def print_stats(self):
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False 
        table.add_row(["# of places", f'{len(self.train_dataset)}'])
        print(table.get_string(title="Training Dataset"))
        print()
