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
import numpy as np

"""
default_transform 是一个默认的图像预处理变换，包括：
ToTensor()：将图片转换为张量格式。
Normalize()：标准化图片数据，采用 ImageNet 数据集的均值和标准差。
"""
default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

"""
BASE_PATH 和 BASE_INFO_PATH 是数据集路径和图片信息文件路径
默认指向 ./data/base_map/ 和 ./data/base_map_info.txt
"""
# NOTE: Hard coded path to dataset folder 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
BASE_PATH = os.path.join(data_dir, 'base_map')
BASE_INFO_PATH = os.path.join(data_dir, 'base_map_info.txt')

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

"""
该类继承自 torch.utils.data.Dataset，用于处理图像数据集
"""
class ImageFolderDataset(Dataset):
    def __init__(self,
                 base_path=BASE_PATH, #数据集存放路径
                 base_info_path=BASE_INFO_PATH, 
                 img_per_place=5, #每个地点的图像数量
                 random_sample_from_each_place=True, #是否从每个地点随机采样图像
                 transform=default_transform,   #图像预处理变换
                 ):
        super(ImageFolderDataset, self).__init__()
        self.base_path = base_path 
        self.base_info_path = base_info_path
        self.img_per_place = img_per_place 
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform

        """
        perspective_transform 是一个随机透视变换，用于增强图像数据
        透视变换会改变图像的几何形状，类似于拍摄者在不同位置或角度拍摄相同的场景。
        这个操作用于数据增强（data augmentation），通过模拟不同视角来增加训练数据的多样性，有助于提高模型的鲁棒性。
        distortion_scale=1.0：这个参数控制透视变换的强度。较高的值会导致更强烈的变形。1.0 表示最大变形
        p=1.0：这个参数控制该变换应用到图像的概率。1.0 表示每次都会应用透视变换
        interpolation=3：这个参数指定了在进行变换时使用的插值方法,3对应的是 Bicubic interpolation（双三次插值）
        """  
        self.perspective_transform = T.RandomPerspective(distortion_scale=1.0, p=1.0, interpolation=3)  
        
        # generate the dataframe contraining images metadata
        self.places_ids, self.total_nb_images = self.__getdataframes()
        
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
        for patch_size, step in zip(self.crop_size, self.step_size):
            #从图像的左上角开始，按照步长和patch大小裁剪图像，直到图像的右下角
            for top in range(0, H, step):  
                for left in range(0, W, step):  
                    bottom = min(top + patch_size, H)
                    right = min(left + patch_size, W) 
                    
                    # if bottom - top < patch_size:
                    #     top = max(bottom - patch_size, 0)
                    # if right - left < patch_size:
                    #     left = max(right - patch_size, 0) 
                        
                    patch = (left, top, right, bottom) 
                    #将裁剪的patch存储起来
                    patches.append(patch)    
                    
        return patches
    
    """
    这个方法的目的是从指定的图像文件路径或描述图像的文本文件中读取图像数据，并生成一系列图像的元数据，主要用于图像的裁剪和数据集构建。
    方法将图像的元数据（如图像名称、裁剪区域坐标）存储在一个字典 place_ids 中，并返回这个字典以及图像的总数量
    """
    def __getdataframes(self):

        place_ids = {} #存储图像元数据的字典
        total_nb_images = 0 #图像的总数量
        #如果是纯图像文件
        if os.path.isfile(self.base_path): 
            
            img_path =  self.base_path

            self.base_im = self.image_loader(img_path)
            #获取尺寸和名字
            image_name = os.path.basename(img_path)
            width, height = self.base_im.size 
            #获取一系列patch的坐标
            patches_list = self.__generate_crop_corrs((height, width))  
            #累加裁剪区域的数量
            total_nb_images += len(patches_list)
            #获取当前 place_ids 字典的长度，作为新图像的起始 place_id，初始值为0
            len_place = len(place_ids)
            
            for i, patch_coor in enumerate(patches_list):
                place_ids[len_place+i] = [image_name, patch_coor]
       
        else: #如果是图像信息文件
            with open(self.base_info_path, 'r') as file: 
                for line in file:
                    #每行包含图像的名称、宽度和高度，使用 strip 去除多余的空白，split 按空格分隔每个字段
                    #由此可见base_map_info的格式样例： 1.jpg 512（height） 512（width）
                    parts = line.strip().split() 
                    #解析txt数据
                    image_name, height, width  = parts[0], int(parts[1]), int(parts[2])
                    patches_list = self.__generate_crop_corrs((height, width))  
                    total_nb_images += len(patches_list)
                    len_place = len(place_ids)
                    for i, patch_coor in enumerate(patches_list):
                        place_ids[len_place+i] = [image_name, patch_coor]
              
        return place_ids, total_nb_images
    
    def __getitem__(self, index):
        #获取图像名字和裁剪区域坐标
        place_id_im_name = self.places_ids[index][0]
        patch_corr = self.places_ids[index][1]
        
        imgs = [] 
        """
        如果 self.base_path 是文件，则 base_im 使用类属性 self.base_im 作为基本图像，
        假设该属性在初始化时已经加载过一次（例如在 __getdataframes 方法中）
        
        如果 self.base_path 不是文件路径，则 img_path 使用 place_id_im_name 拼接成一个完整的文件路径，
        之后通过 self.image_loader(img_path) 加载该图像。
        """
        if os.path.isfile( self.base_path ):  
            base_im = self.base_im 
        else: 
            img_path = os.path.join(self.base_path, place_id_im_name) 
            base_im = self.image_loader(img_path)
              
        base_im_place = base_im.crop(patch_corr) 
        for i in range(self.img_per_place):  
            img = base_im_place
            
            if self.transform is not None:
                img = self.transform(img) 
                imgs.append(self.transform2(img)) 
                img2 = self.apply_transform_and_crop(img)
                imgs.append(self.transform2(img2))
               
        
        return torch.stack(imgs), torch.tensor(index).repeat(self.img_per_place*2)
    
    #有疑问？
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
        
        
    """
    去除图像中的黑色边框
    """
    def remove_black_borders_pil(self, image):
        #将图像转化为（H,W,C）数组
        img_np = np.array(image)

        """
        检测非黑色区域

        np.sum(img_np, axis=-1)：沿着最后一个轴（即颜色通道 C）对图像的 RGB 通道进行求和。
        对于每个像素点，RGB 三个通道的值会被加总，这样我们就得到了一个二维数组，其中每个元素是该像素点的 RGB 总和。
        例如，如果一个像素的 RGB 值为 (0, 0, 0)（黑色），它的总和就是 0；而对于颜色较亮的像素，RGB 总和会更大。

        mask = np.sum(img_np, axis=-1) > 30：通过阈值 30 来判断一个像素点是否为“非黑色”像素。
        如果该像素的 RGB 总和大于 30，则认为该像素不是黑色的，mask 数组中该位置的值为 True。
        否则，认为该像素是黑色的，mask 中该位置的值为 False
        """
        mask = np.sum(img_np, axis=-1) > 30  

        """
        获取非黑色区域的行列坐标

        mask.sum(axis=1)：对 mask 数组沿着列方向（即垂直方向）求和，得到每一行是否包含至少一个非黑色像素。
        如果某行包含至少一个非黑色像素，其值就大于 0。

        np.where(mask.sum(axis=1) > 0)[0]：返回所有非黑色行的索引。
        non_black_rows 存储了所有包含非黑色像素的行的索引。

        mask.sum(axis=0)：对 mask 数组沿着行方向（即水平方向）求和，得到每一列是否包含至少一个非黑色像素。

        np.where(mask.sum(axis=0) > 0)[0]：返回所有非黑色列的索引。
        non_black_cols 存储了所有包含非黑色像素的列的索引。
        """
        non_black_rows = np.where(mask.sum(axis=1) > 0)[0]
        non_black_cols = np.where(mask.sum(axis=0) > 0)[0]

        """
        non_black_rows.min() 和 non_black_rows.max()：找到包含非黑色像素的行的最小和最大索引，分别表示非黑色区域的顶部和底部。

        non_black_cols.min() 和 non_black_cols.max()：找到包含非黑色像素的列的最小和最大索引，分别表示非黑色区域的左边界和右边界。

        如果找到了非黑色区域（即 non_black_rows.size > 0 和 non_black_cols.size > 0），
        就用这些边界坐标定义出一个矩形框，即非黑色区域的“裁剪框”。
        """
        if non_black_rows.size > 0 and non_black_cols.size > 0:
            top, bottom = non_black_rows.min(), non_black_rows.max()
            left, right = non_black_cols.min(), non_black_cols.max()

            """
            image.crop((left, top, right, bottom))：使用 PIL 的 crop 方法将图像
            裁剪为由 (left, top, right, bottom) 所定义的区域。
            这里的 (left, top, right, bottom) 是之前计算得到的非黑色区域的边界框
            """
            cropped_img = image.crop((left, top, right, bottom))
        else:
            #若无非黑色区域，返回原图
            cropped_img = image

        return cropped_img

    """
    应用透视变换后再去除黑色边框

    透视变换可能会导致一些图像区域的内容被拉伸或扭曲，造成图像的某些部分变为空白，通常这些空白区域会显示为黑色。

    调用 remove_black_borders_pil 后，图像中不包含实际内容的黑色区域会被裁剪掉，只保留图像中有实际内容的部分。
    """
    def apply_transform_and_crop(self, image): 
        # 在init中已经定义
        transformed_img = self.perspective_transform(image)

        # Remove black borders from the transformed image
        cropped_img = self.remove_black_borders_pil(transformed_img)

        return cropped_img  