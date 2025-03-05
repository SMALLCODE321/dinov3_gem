# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import glob
import os
import random

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = './data/base_map/'
BASE_INFO_PATH = './data/base_map_info.txt'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

class ImageFolderDataset(Dataset):
    def __init__(self,
                 base_path=BASE_PATH,
                 base_info_path=BASE_INFO_PATH,
                 crop_size=500,
                 img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 ):
        super(ImageFolderDataset, self).__init__()
        self.base_path = base_path 
        self.base_info_path = base_info_path
        self.crop_size = crop_size
 
        self.img_per_place = img_per_place 
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.patches_all_places, self.patches_coor, self.total_nb_images, self.places_ids = self.__getdataframes()
         
    def generate_random_rectangle(self):
        """
        在一个指定尺寸的矩形框内随机生成一个矩形框，要求矩形框的长和宽都在500到800像素之间。
        
        参数:
        - outer_width (int): 外部矩形框的宽度。
        - outer_height (int): 外部矩形框的高度。
        - min_size (int): 矩形框的最小尺寸（宽度和高度）。
        - max_size (int): 矩形框的最大尺寸（宽度和高度）。
        
        返回:
        - tuple: 生成的矩形框的坐标和尺寸 (left, top, width, height)。
        """
        # 随机选择矩形框的宽度和高度
        outer_width  = self.crop_size
        outer_height = self.crop_size
        min_size = self.crop_size - self.crop_size//4 
        max_size = self.crop_size
        rect_width = random.randint(min_size, max_size)
        rect_height = random.randint(min_size, max_size)
          
        # 随机选择矩形框的左上角坐标
        left = random.randint(0, outer_width - rect_width)
        top = random.randint(0, outer_height - rect_height)
        
        return [left, top, rect_width, rect_height]  
    
    def __generate_crop_corrs(self, image_size): 
        """  
        将图像裁剪成固定大小的patch。  
    
        :param img: 输入的图像张量，形状为(C, H, W)  
        :param patch_size: 每个patch的大小，形状为(h, w)  
        :param step_size: 裁剪时的步长  
        :return: 裁剪后的patch列表  
        """  
        H, W = image_size 
        patches = []  
        patch_size = self.crop_size
        step = patch_size-patch_size//2
        
        for top in range(0, H, step):  
            for left in range(0, W, step):  
                bottom = min(top + patch_size, H)
                right = min(left + patch_size, W) 
                
                # if bottom - top < patch_size:
                #     top = max(bottom - patch_size, 0)
                # if right - left < patch_size:
                #     left = max(right - patch_size, 0) 
                    
                patch = (left, top, right, bottom) 
                patches.append(patch)    
                    
        return patches
    
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe  
        im_infos = []
        patches_all = {}
        patches_coor = []
        place_ids = []
        total_nb_images = 0
        with open(self.base_info_path, 'r') as file: 
            for line in file:
                # Strip any leading/trailing whitespace and split the line by spaces
                parts = line.strip().split() 
                # Unpack the line into variables: image_name, width, height
                image_name, height, width  = parts[0], int(parts[1]), int(parts[2])
                patches_list = self.__generate_crop_corrs((height, width)) 
                 
                for i, patch in enumerate(patches_list):
                    patches_all[f'{i}_{image_name}'] = [self.generate_random_rectangle() for i in range(self.img_per_place*4)]  
                    place_ids.append(f'{i}_{image_name}')
                    total_nb_images += self.img_per_place*4
                    patches_coor.append(patch)
                    
        return patches_all, patches_coor, total_nb_images, place_ids
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        place_coor = self.patches_coor[index]
        # get the place in form of a dataframe (each row corresponds to one image)
        patches_corrs = self.patches_all_places[place_id]
        
        # sample K images (rows) from this place
        # we can either sort and take the most recent k images
        # or randomly sample them
        if self.random_sample_from_each_place:
            patches_corrs = random.sample(patches_corrs, self.img_per_place) 
        else:  # always get the same most recent images 
            patches_corrs = patches_corrs[: self.img_per_place]
            
        imgs = [] 
        img_path = os.path.join(self.base_path, place_id.split('_')[1])
        base_im =  self.image_loader(img_path)
        base_im = base_im.crop(place_coor)
        for i in range(self.img_per_place): 
            corrs = patches_corrs[i]
            img = base_im.crop(corrs) 
            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
 
        return torch.stack(imgs), torch.tensor(int(place_id.split('.')[0])).repeat(self.img_per_place)

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))
 
