import os
import glob
import random
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl
from prettytable import PrettyTable

IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class UniversityPairDataset(Dataset):
    """
    University-1652 成对数据集：
    每个 Place 返回 1 张卫星图和多张随机采样的无人机图。
    """
    def __init__(self,
                 data_path,
                 drone_per_place=4,
                 base_transform=None,
                 aug_transform=None):
        super().__init__()
        self.data_path = data_path
        self.drone_per_place = drone_per_place
        self.base_transform = base_transform
        self.aug_transform = aug_transform

        self.sat_dir = os.path.join(data_path, 'satellite')
        self.drone_dir = os.path.join(data_path, 'drone')

        # 1. 建立索引：只选取同时拥有卫星和无人机图的 ID
        all_pids = sorted(os.listdir(self.sat_dir)) if os.path.exists(self.sat_dir) else []
        self.place_ids = []
        self.data_info = {}

        for pid in all_pids:
            # 兼容 .jpeg, .jpg 等后缀
            sat_paths = glob.glob(os.path.join(self.sat_dir, pid, "*.[jJ][pP][eE][gG]")) + \
                        glob.glob(os.path.join(self.sat_dir, pid, "*.[jJ][pP][gG]"))
            drone_paths = glob.glob(os.path.join(self.drone_dir, pid, "*.[jJ][pP][eE][gG]")) + \
                          glob.glob(os.path.join(self.drone_dir, pid, "*.[jJ][pP][gG]"))

            if sat_paths and drone_paths:
                self.data_info[pid] = {
                    'sat': sat_paths[0],
                    'drones': drone_paths
                }
                self.place_ids.append(pid)

        if len(self.place_ids) == 0:
            raise ValueError(f"在 {data_path} 下未找到有效的 Satellite-Drone 对，请检查后缀名是否为 .jpeg 或 .jpg")

    def __len__(self):
        return len(self.place_ids)

    def __getitem__(self, index):
        pid = self.place_ids[index]
        info = self.data_info[pid]

        # 1. 读取卫星图 (Anchor)
        try:
            sat_img = Image.open(info['sat']).convert('RGB')
        except UnidentifiedImageError:
            sat_img = Image.new('RGB', (336, 336))

        sat_patch = self.base_transform(sat_img) if self.base_transform else T.ToTensor()(sat_img)

        # 2. 采样并读取无人机图 (Positives)
        # 如果图片不够，则使用 random.choices 进行带放回采样
        if len(info['drones']) >= self.drone_per_place:
            selected_drones = random.sample(info['drones'], self.drone_per_place)
        else:
            selected_drones = random.choices(info['drones'], k=self.drone_per_place)

        drone_patches = []
        for d_path in selected_drones:
            try:
                d_img = Image.open(d_path).convert('RGB')
                # 无人机图建议使用 aug_transform 
                d_patch = self.aug_transform(d_img) if self.aug_transform else T.ToTensor()(d_img)
            except UnidentifiedImageError:
                d_patch = torch.zeros_like(sat_patch)
            drone_patches.append(d_patch)

        # 3. 合并：[1 + drone_per_place, C, H, W]
        # 顺序：[卫星图, 无人机1, 无人机2, ...]
        stacked_patches = torch.stack([sat_patch] + drone_patches)

        # 4. 标签：[1 + drone_per_place]
        label = torch.tensor(index).repeat(1 + self.drone_per_place)
        
        return stacked_patches, label


class UniversityPairDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path,
                 batch_size=8,
                 image_size=(336, 336),
                 drone_per_place=4,
                 num_workers=4,
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.drone_per_place = drone_per_place
        self.num_workers = num_workers
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.image_size = image_size

        # 卫星图预处理：保持尺寸和标准化
        self.base_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        
        # 无人机图预处理：增加视角变换相关的鲁棒性
        self.aug_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

    def setup(self, stage=None):
        self.train_dataset = UniversityPairDataset(
            data_path=self.data_path,
            drone_per_place=self.drone_per_place,
            base_transform=self.base_transform,
            aug_transform=self.aug_transform
        )

    def train_dataloader(self):
        # 注意：这里不需要在 DataModule 里写特殊的 collate_fn，
        # 因为我们希望 DataLoader 返回 [BS, N, C, H, W] 这种 5 维结构
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True 
        )

    def print_stats(self):
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of Places", f'{len(self.train_dataset)}'])
        table.add_row(["Drones per Place", f'{self.drone_per_place}'])
        table.add_row(["Batch Size", f'{self.batch_size}'])
        print(table.get_string(title="University-1652 Training Set"))
        return table
