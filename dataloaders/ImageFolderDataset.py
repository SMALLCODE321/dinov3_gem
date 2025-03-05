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
import numpy as np
import cv2

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = '/home/lyf/data/xiangshan/image'
BASE_INFO_PATH = '/home/lyf/data/xiangshan/base_map_info.txt'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')


class ImageFolderDataset(Dataset):
    def __init__(self,
                 base_path=BASE_PATH,
                 base_info_path=BASE_INFO_PATH, 
                 crop_size=(1000, 2000),
                 step_size=(500, 1000),
                 img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 ):
        super(ImageFolderDataset, self).__init__()
        self.base_path = base_path 
        self.base_info_path = base_info_path
        self.crop_size = crop_size
        self.step_size = step_size
 
        self.img_per_place = img_per_place 
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # generate the dataframe contraining images metadata
        self.patches_all_places, self.total_nb_images, self.places_ids = self.__getdataframes()
         
        
    def __generate_crop_corrs(self, image_size): 
        """  
        将图像裁剪成固定大小的patch。  
    
        :param img: 输入的图像张量，形状为(C, H, W)  
        :param patch_size: 每个patch的大小,形状为(h, w)  
        :param step_size: 裁剪时的步长  
        :return: 裁剪后的patch列表  
        """  
        H, W = image_size 
        patches = []  
        for patch_size, step in zip(self.crop_size, self.step_size):
            # step = patch_size-patch_size//2 
            for top in range(0, H, step):  
                for left in range(0, W, step):  
                    bottom = min(top + patch_size, H)
                    right = min(left + patch_size, W) 
                    
                    if bottom - top < patch_size:
                        top = max(bottom - patch_size, 0)
                    if right - left < patch_size:
                        left = max(right - patch_size, 0) 
                        
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
        place_ids = []
        total_nb_images = 0
        with open(self.base_info_path, 'r') as file: 
            for line in file:
                # Strip any leading/trailing whitespace and split the line by spaces
                parts = line.strip().split() 
                # Unpack the line into variables: image_name, width, height
                image_name, height, width  = parts[0], int(parts[1]), int(parts[2])
                patches_list = self.__generate_crop_corrs((height, width)) 
                patches_all[image_name] = patches_list 
                total_nb_images += len(patches_list)
                place_ids.append(image_name)
                
        return patches_all, total_nb_images, place_ids
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        imgs = []
        img_path = os.path.join(self.base_path, place_id)
        base_im =  self.image_loader(img_path)
        for i in range(self.img_per_place):  
            img = self.random_crop(base_im)
            # img = self.rotate_image_with_probability(img)
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
        
    @staticmethod
    def rotate_image_with_probability(image, probability=0):
        # Decide whether to rotate the image based on the probability
        if random.random() < probability:
            # Generate a random angle for rotation
            angle = random.uniform(0, 360)
            
            # Rotate the image with expand=True to ensure the entire image is visible
            rotated_image = image.rotate(angle, expand=True)
            
            return rotated_image
        else:
            # Return the image unchanged
            return image
    # @staticmethod
    # def perspect_transform(image):
    #         img = np.array(image)
    #         height, width = img.shape[:2]
    #         # 定义原始图像中的四点坐标（假设是正方形或者矩形）
    #         src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    #         # 计算新的高度和宽度（这里简化处理，根据角度计算）
    #         # 假设图像底部不变，顶部变窄，宽度变化比例根据tan(30°) = 1/√3 ≈ 0.577
    #         angle = random.randint(0, 40)
    #         new_width = int(width * np.tan(np.deg2rad(angle)))
    #         new_height = height
    #         # 根据变换后的图像尺寸定义目标点坐标
    #         dst_points = np.float32([
    #             [new_width / 2, 0],
    #             [width - new_width / 2, 0],
    #             [width, new_height],
    #             [0, new_height]
    #         ])
    #         # 获取透视变换矩阵
    #         M = cv2.getPerspectiveTransform(src_points, dst_points)
    #         # 应用透视变换
    #         transformed_img = cv2.warpPerspective(img, M, (width, new_height))
    #         transformed_img =Image.fromarray(transformed_img)
    #         image = transformed_img
    #         return image
    

    @staticmethod    
    def random_crop(image, min_size=200, max_size=800):
        """
        从给定图像中随机采样一个切片，切片的尺寸在 min_size 到 max_size 之间。

        :param image: PIL.Image 对象
        :param min_size: 切片的最小尺寸
        :param max_size: 切片的最大尺寸
        :return: PIL.Image 对象，切片图像
        """
        width, height = image.size

        # 随机选择切片的尺寸
        crop_size = random.randint(min_size, max_size)
        
        # 确保切片尺寸不会超出原图像的尺寸
        crop_width = min(crop_size, width)
        crop_height = min(crop_size, height)

        # 随机选择切片的位置
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        # 裁剪图像
        crop_image = image.crop((left, top, right, bottom))
        return crop_image