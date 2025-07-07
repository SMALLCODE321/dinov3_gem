import os
import glob
from PIL import Image, UnidentifiedImageError
import random

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
                 satellite_path,
                 drone_path,
                 num_queries=5,
                 num_augs=3,
                 image_size=(322, 322),
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.sat_dir = satellite_path
        self.query_dir = drone_path
        self.num_queries = num_queries
        self.num_augs = num_augs

        self.sat_paths, self.ids = self.get_all_satellite()

        # 通用的加载 transform：Resize + ToTensor + Normalize
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])
        self.aug_transform = T.Compose([
            T.RandomRotation(degrees=360),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

    def __len__(self):
        return len(self.sat_paths)

    def get_all_satellite(self):
        """
        返回所有卫星图像的路径列表以及对应的 id（子目录名）。
        支持的后缀：jpg, jpeg, png, bmp, tif, tiff，
        不再区分先采非 TIFF 再采 TIFF，所有格式全部保留。
        """
        sat_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        sat_paths = []
        ids       = []

        for sub in os.listdir(self.sat_dir):
            subdir = os.path.join(self.sat_dir, sub)
            if not os.path.isdir(subdir):
                continue

            # 收集所有支持的后缀
            imgs = []
            for ext in sat_exts:
                imgs.extend(glob.glob(os.path.join(subdir, ext)))

            if not imgs:
                continue

            sat_paths.append(sorted(imgs))
            ids.append(sub)

        if not sat_paths:
            raise ValueError(f"在 {self.sat_dir} 下没有找到任何支持的卫星图像。")
        return sat_paths, ids
    
    def __getitem__(self, idx):
        pid = self.ids[idx]
        sat_path = self.sat_paths[idx]
        galleries = []
        if sat_path is list:
            for path in sat_path:
                img = Image.open(path).convert('RGB')
                galleries.append(self.transform(img))
        else:
            img = Image.open(sat_path).convert('RGB')
            galleries.append(self.transform(img))

        # load all query images
        qd = os.path.join(self.query_dir, pid)
        exts = ("*.jpg","*.JPG", "*.jpeg","*.png","*.bmp")
        qps = []
        for e in exts:
            qps += sorted(glob.glob(os.path.join(qd, e)))
        if not qps:
            raise ValueError(f"No query images in {qd}")

        # sample or take all
        if self.num_queries is not None and len(qps) > self.num_queries:
            qps = random.sample(qps, self.num_queries)

        queries = []
        for qp in qps:
            try:
                qimg = Image.open(qp).convert('RGB')
            except UnidentifiedImageError:
                continue
            queries.append(self.transform(qimg))
        
        for _ in range(self.num_augs):
            queries.append(self.aug_transform(img))

        patches = galleries + queries
        stacked = torch.stack(patches, dim=0)  # (1+Nq, C, H, W)

        # use pid as integer label
        try:
            label_id = int(pid)
        except:
            # fallback to idx if pid not numeric
            label_id = idx
        label = torch.full((len(patches),), label_id, dtype=torch.long)
        return stacked, label


class PatchImageDataModule(pl.LightningDataModule):
    """
    LightningDataModule，直接读取预生成好的 patch/query 目录，不做动态增强。
    """
    def __init__(self,
                 train_satellite_path,
                 train_drone_path,
                 val_set_names,
                 num_queries=5,
                 num_augs=3,
                 batch_size=32,
                 image_size=(322, 322),
                 num_workers=4,
                 mean_std=IMAGENET_MEAN_STD):
        super().__init__()
        self.train_satellite_path = train_satellite_path
        self.train_drone_path = train_drone_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_std = mean_std
        self.val_set_names = val_set_names

        self.num_queries = num_queries
        self.num_augs = num_augs

        self.train_loader_cfg = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': False
        }

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage):
        if stage == 'fit':
            self.reload()
        self.val_datasets = []
        for valid_set_name in self.val_set_names:
            if 'university-1652' in valid_set_name.lower():
                self.val_datasets.append(PatchDataset(
                    satellite_path=os.path.join('./datasets/'+valid_set_name,'test/gallery_satellite'),
                    drone_path=os.path.join('./datasets/'+valid_set_name,'test/query_satellite'),
                    num_queries=None,
                    image_size=self.image_size,
                    mean_std=self.mean_std
                ))
            else:
                print(
                    f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                raise NotImplementedError
        self.print_stats()

    def reload(self):
        self.train_dataset = PatchDataset(
            satellite_path=self.train_satellite_path,
            drone_path=self.train_drone_path,
            num_queries=self.num_queries,
            num_augs=self.num_augs,
            image_size=self.image_size,
            mean_std=self.mean_std
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(self.train_dataset, **self.train_loader_cfg)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False 
        table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        # table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        print(table.get_string(title="Validation Datasets"))
        print()
        

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(
            ["Batch size (PxK)", f"{self.batch_size}x{8}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))