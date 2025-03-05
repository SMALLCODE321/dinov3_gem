import random
import pytorch_lightning as pl
import torch
import os
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.ImageFolderDataset_patch import ImageFolderDataset  
from prettytable import PrettyTable 
from . import TZBDataset 

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}
 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
BASE_PATH = os.path.join(data_dir, 'base_map')
BASE_INFO_PATH = os.path.join(data_dir, 'base_map_info.txt')
        
class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_place=4,
                 min_img_per_place=4,
                 shuffle_all=False, #是否打乱顺序
                 image_size=(480, 640),
                 crop_size=(1000, 2000),
                 step_size=(500, 1000),
                 min_scale=100,
                 max_scale=1000,
                 num_workers=4,
                 show_data_stats=True, #在控制台打印统计信息 
                 mean_std=IMAGENET_MEAN_STD, #标准化的均值和标准差
                 batch_sampler=None,
                 random_sample_from_each_place=True, #是否随机抽取图像
                 base_path=BASE_PATH, # can be an image path
                 base_info_path=BASE_INFO_PATH, # unuseful if base_path is an image path
                 val_set_names=['tzb_val2']
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.crop_size = crop_size
        self.step_size = step_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        
        self.base_path = base_path
        self.base_info_path = base_info_path
        
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        """
        数据增强与预处理
        """
        """
        train_transform：为训练集定义了一系列的数据增强操作：

        随机裁剪：RandomResizedCrop，设置图像大小并随机缩放。
        随机旋转：RandomRotation，允许图像在 360 度内旋转。
        数据增强：RandAugment，进行随机的图像增强（例如色彩、对比度等）。
        """
        self.train_transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(self.min_scale, self.max_scale), ratio=(0.5, 1.75)),#ratio=(0.75, 1.33)),
            T.RandomRotation(degrees=360),
            # T.RandomPerspective(distortion_scale=1.0, p=1.0, interpolation=3),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            # T.ToTensor(),
            # T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        
        """
        train_transform2：额外的训练预处理步骤，首先调整图像大小，接着将图像转为张量，并进行标准化
        """
        self.train_transform2 = T.Compose([
            T.Resize(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        """
        验证集的预处理，首先调整图像大小，然后转为张量并进行标准化。
        """
        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}
        
        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}
  
    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader with reload routine
            self.reload()

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if 'tzb_val' in valid_set_name.lower():
                    self.val_datasets.append(TZBDataset.TZB(data_root='./datasets/'+valid_set_name,
                        input_transform=self.valid_transform)) 
                elif 'tz_val' in valid_set_name.lower():
                    self.val_datasets.append(TZBDataset.TZB(data_root='./datasets/'+valid_set_name,
                        input_transform=self.valid_transform))
                elif 'taizhou_val1' in valid_set_name.lower():
                    self.val_datasets.append(TZBDataset.TZB(data_root='./datasets/'+valid_set_name,
                        input_transform=self.valid_transform)) 
                else:
                    print(
                        f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                    raise NotImplementedError
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = ImageFolderDataset(
            base_path=self.base_path,
            base_info_path=self.base_info_path,
            crop_size=self.crop_size,
            step_size=self.step_size,
            img_per_place=self.img_per_place, 
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            transform2=self.train_transform2)

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

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
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
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
            ["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
