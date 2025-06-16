import os
import glob
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms as T
from prettytable import PrettyTable

# 默认 ImageNet 的标准化参数
IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std':  [0.229, 0.224, 0.225]
}


class PatchDataset(Dataset):
    """
    从本地预生成的 patch + query 目录读取数据。
    root_dir/
      ├── satellite/
      │     ├── 1.jpg
      │     ├── 2.jpg
      │     └── ...
      └── query/
            ├── 1/
            │   ├── 1_a.jpg
            │   ├── 1_b.jpg
            │   └── ...
            ├── 2/
            │   └── ...
            └── ...
    每个 sample 返回一个 tensor 堆叠：[原图, query1, query2, ...]，以及对应的 label (index).
    """
    def __init__(self,
                 root_dir,
                 image_size=(322, 322),
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.root_dir = root_dir
        self.sat_dir = os.path.join(root_dir, "satellite")
        self.query_dir = os.path.join(root_dir, "drone")

        # 收集所有 satellite images
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.sat_paths = []
        for sub in os.listdir(self.sat_dir):
            subdir = os.path.join(self.sat_dir, sub)
            if not os.path.isdir(subdir):
                continue
            for e in exts:
                self.sat_paths += glob.glob(os.path.join(subdir, e))
        if not self.sat_paths:
            raise ValueError(f"在 {self.sat_dir} 下没有找到任何图片文件。")
        # 保证排序一致
        self.sat_paths = sorted(self.sat_paths)
        # 提取 id（不带后缀），用于匹配 query 子目录
        self.ids = [os.path.basename(os.path.dirname(p)) for p in self.sat_paths]

        # 通用的加载 transform：Resize + ToTensor + Normalize
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

    def __len__(self):
        return len(self.sat_paths)

    def __getitem__(self, idx):
        patch_id = self.ids[idx]
        sat_path = self.sat_paths[idx]

        # 加载原始 patch
        try:
            img = Image.open(sat_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Image {sat_path} 无法加载，将用全黑图替代。")
            img = Image.new('RGB', (self.transform.transforms[0].size))

        base = self.transform(img)

        # 加载对应的 query 目录下所有图
        q_dir = os.path.join(self.query_dir, patch_id)
        if not os.path.isdir(q_dir):
            raise ValueError(f"找不到 {patch_id} 对应的 query 目录：{q_dir}")
        q_paths = []
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for e in exts:
            q_paths += glob.glob(os.path.join(q_dir, e))
        q_paths = sorted(q_paths)
        if not q_paths:
            raise ValueError(f"{q_dir} 下没有找到任何 query 图像。")

        queries = []
        for qp in q_paths:
            try:
                qimg = Image.open(qp).convert('RGB')
            except UnidentifiedImageError:
                print(f"Image {qp} 无法加载，跳过。")
                continue
            queries.append(self.transform(qimg))

        # 将原始 patch 放在第一位，其它 query 紧随其后
        all_patches = [base] + queries
        stacked = torch.stack(all_patches, dim=0)  # [1 + N_query, C, H, W]

        # label 可用 idx，也可用 patch_id 转 int
        label = torch.tensor(idx, dtype=torch.long).repeat(len(all_patches))
        return stacked, label


class PatchImageDataModule(pl.LightningDataModule):
    """
    LightningDataModule，直接读取预生成好的 patch/query 目录，不做动态增强。
    """
    def __init__(self,
                 root_dir,
                 val_path,
                 batch_size=32,
                 image_size=(322, 322),
                 num_workers=4,
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_std = mean_std

        self.train_loader_cfg = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': False
        }

    def setup(self, stage=None):
        # 只做 train
        self.train_dataset = PatchDataset(
            root_dir=self.root_dir,
            image_size=self.image_size,
            mean_std=self.mean_std
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.train_loader_cfg)

    def print_stats(self):
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of places", f"{len(self.train_dataset)}"])
        print(table.get_string(title="Training Dataset"))
        print()