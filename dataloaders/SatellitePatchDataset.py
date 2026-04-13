import os
import glob
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import pytorch_lightning as pl
from prettytable import PrettyTable
IMAGENET_MEAN_STD = {
    'mean': [0.430, 0.411, 0.296],
    'std': [0.213, 0.156, 0.143]
}


class SatelliteSmallDataset(Dataset):
    """
    用多张 322x322 小图直接训练，无需裁剪大图。
    每张图作为一个 place，并可进行多次数据增强。
    """
    def __init__(self,
                 data_path,           # 多张小图的路径或单个文件
                 sat_aug_per_place=5,
                 base_transform=None,
                 aug_transform=None):
        super().__init__()
        self.data_path = data_path
        self.sat_aug_per_place = sat_aug_per_place
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        
        # 收集图片路径
        if os.path.isdir(self.data_path):
            self.image_paths = []
            # 使用 os.walk 递归地获取所有子目录下的图片路径
            for root, dirs, files in os.walk(self.data_path):
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    for img_path in glob.glob(os.path.join(root, ext)):
                        self.image_paths.append(img_path)

            if len(self.image_paths) == 0:
                raise ValueError("指定目录中未找到任何图片文件。")
            self.image_paths = sorted(self.image_paths)
        elif os.path.isfile(self.data_path):
            self.image_paths = [self.data_path]
        else:
            raise ValueError("data_path 指定的路径不存在。")

        self.num_images = len(self.image_paths)
        self.angles = np.linspace(0, 180, self.sat_aug_per_place, endpoint=False)
    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Image {img_path} 无法加载，将替换为一张空白图像。")
            img = Image.new('RGB', (322, 322))

        # 基础预处理
        if self.base_transform is not None:
            base_patch = self.base_transform(img)
        else:
            base_patch = T.ToTensor()(img)

        # 数据增强
        aug_patches = []
        for angle in self.angles:
            rotated_img = TF.rotate(
                img,
                angle=angle,
                interpolation=T.InterpolationMode.BILINEAR,
                expand=False
            )
            # 2. 进入增强（里面不再包含 RandomRotation！）
            aug_patch = self.aug_transform(rotated_img)
            aug_patches.append(aug_patch)

        # 合并基础版和增强版
        final_patches = [base_patch] + aug_patches
        stacked_patches = torch.stack(final_patches)  # shape: [1+sat_aug_per_place, C, H, W]

        # 每张小图作为一个 place，label 就是 index
        label = torch.tensor(index).repeat(len(final_patches))
        return stacked_patches, label


class SatelliteSmallDataModule(pl.LightningDataModule):
    """
    Lightning DataModule，用于训练多张小图。
    """
    def __init__(self,
                 data_path,
                 batch_size=32,
                 image_size=(224, 224),
                 sat_aug_per_place=5,
                 num_workers=4,
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.sat_aug_per_place = sat_aug_per_place
        self.num_workers = num_workers
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.image_size = image_size

        # 基础预处理
        self.base_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        '''
        self.aug_transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.7, 1.0)),
            T.RandomRotation(degrees=(90, 360)),
            T.RandomPerspective(distortion_scale=0.7, p=0.7, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])
        '''
        
        self.aug_transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.5, 0.9)), # 扩大缩放范围，模拟高度差异
            T.RandomHorizontalFlip(p=0.5), # 无人机视角下左右翻转语义一致
            
            T.RandomPerspective(distortion_scale=0.5, p=0.7),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1), # 模拟光照和传感器差异
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2)), # 模拟建筑物遮挡
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': True
        }

    def setup(self, stage=None):
        self.train_dataset = SatelliteSmallDataset(
            data_path=self.data_path,
            sat_aug_per_place=self.sat_aug_per_place,
            base_transform=self.base_transform,
            aug_transform=self.aug_transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader_config)

    def print_stats(self):
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of images", f'{len(self.train_dataset)}'])
        print(table.get_string(title="Training Dataset"))
        return table
