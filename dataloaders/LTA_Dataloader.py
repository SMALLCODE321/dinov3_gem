import pytorch_lightning as pl
import torch
import os
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from dataloaders.LTA_Dataset import ImageFolderDataset, LTAValDataset
from prettytable import PrettyTable 
from . import TZBDataset 

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}
    
class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 shuffle_all=False,
                 img_per_place=8, # 每个地点取的图像
                 sat_aug_per_place=5, #卫星放射变换图像数
                 image_size=(322, 322),
                 num_workers=4,
                 show_data_stats=True,
                 mean_std=IMAGENET_MEAN_STD,
                 data_path=None,
                 val_set_names=['val']
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.img_per_place = img_per_place
        self.sat_aug_per_place = sat_aug_per_place
        self.num_workers = num_workers
        self.show_data_stats = show_data_stats
        self.data_path = data_path
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.val_set_names = val_set_names

        self.save_hyperparameters() # save hyperparameter with Pytorch Lightening

        self.transform_sat = T.Compose([
            T.RandomRotation(degrees=360),
            T.RandomPerspective(distortion_scale=1.0, p=1.0, interpolation=3),
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])
        
        self.transform_drone = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}
        
        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': max(self.num_workers // 2, 1),
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}
  
    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader with reload routine
            self.reload()

            # # load validation sets (pitts_val, msls_val, ...etc)
            # self.val_datasets = []
            # for valid_set_name in self.val_set_names:
            #     if 'UAV_Large_Tilt_Angle/val/query'  in valid_set_name:
            #         self.val_datasets.append(LTAValDataset(data_root='./datasets/'+valid_set_name, input_transform=self.transform_drone))
            #     else:
            #         print(
            #             f'Validation set {self.val_set_name} does not exist or has not been implemented yet')
            #         raise NotImplementedError
            # if self.show_data_stats:
            #     self.print_stats()

    def reload(self):
        # 创建无人机数据数据集，对应每个地点直接加载对应的图像
        self.train_dataset = ImageFolderDataset(
            data_path=self.data_path,
            img_per_place = self.img_per_place,
            sat_aug_per_place = self.sat_aug_per_place,
            transform_sat=self.transform_sat,
            transform_drone=self.transform_drone
            )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloader=DataLoader(dataset=self.val_datasets, **self.valid_loader_config)
        return val_dataloader

    def print_stats(self):
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False 
        table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_images}'])
        print(table.get_string(title="Training Dataset"))
        print()
