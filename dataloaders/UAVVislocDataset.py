import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import re

IMAGENET_MEAN_STD = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class UAVVislocDataset(Dataset):
    def __init__(self, im_path='', image_size=None, mean_std=IMAGENET_MEAN_STD):
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.input_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        self.image_size = image_size
        # 遍历文件夹获取数据，同时解析 xlsx 文件中 UAV 图中心点经纬度坐标
        self.data = self.__getdata__(im_path)
    
    def __getdata__(self, root_dir):
        """
        遍历 root_dir 下的所有 place 文件夹（例如 01, 02, 03, ...）。
        对每个 place 文件夹：
        - 定位 drone 文件夹，里面保存所有 UAV 图像
        - 定位 CSV 文件（例如 01.csv），其中包含每张无人机图像中心点的经纬度坐标（字段：'lat' 和 'lon'）以及图像文件名（字段：'filename'）
        - 为 CSV 中的每一行记录构建一个样本，样本的数据结构为：
                {
                "uav": [图片路径],
                "label": {
                    "coords": (lat, lon),
                    "base_img": place 文件夹名称（代表底图或区分标识）
                }
                }
        """
        data = {}
        index = 0
        # 遍历所有 place 文件夹（建议使用排序保持顺序一致）
        for place in sorted(os.listdir(root_dir)):
            place_path = os.path.join(root_dir, place)
            if not os.path.isdir(place_path):
                continue
            
            # 定位 CSV 文件，文件名与 place 文件夹名称相同，如 01.csv
            csv_file = os.path.join(place_path, f"{place}.csv")
            if not os.path.exists(csv_file):
                continue

            # 定位 drone 文件夹
            drone_dir = os.path.join(place_path, "drone")
            if not os.path.isdir(drone_dir):
                continue
            
            # 读取 CSV 文件，期望包含 'filename', 'lat', 'lon' 三个字段
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                filename = row['filename']
                lat = row['lat']
                lon = row['lon']
                image_path = os.path.join(drone_dir, filename)
                sample = {
                    "uav": [image_path],
                    "label": {
                        "coords": (lat, lon),
                        "base_img": place
                    }
                }
                data[index] = sample
                index += 1
        return data

    def __getitem__(self, index):
        """
        返回样本，包含：
        - 单张 UAV 图片（经过数据变换）
        - label 信息，其中包含 UAV 图中心点经纬度坐标与 place 标识
        """
        sample = self.data[index]
        imgs = []
        for img_path in sample["uav"]:
            img = Image.open(img_path).convert('RGB')
            if self.input_transform:
                img = self.input_transform(img)
            imgs.append(img)
        label = sample["label"]
        return imgs, label

    def __len__(self):
        return len(self.data)