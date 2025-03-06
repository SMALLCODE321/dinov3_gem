#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch.nn as nn
from torchvision import transforms

# ---------------------------
# 定义图像预处理 (例如使用 ImageNet 的均值和方差)
# ---------------------------
input_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------------
# 1. 定义 TestImageCropDataset 用于切割底图，同时返回patch坐标
# ----------------------------------------------------------
class TestImageCropDataset(Dataset):
    def __init__(self, im_path, crop_size, step_size, input_transform=input_transform):
        """
        初始化数据集
        :param im_path: 底图路径
        :param crop_size: (高度, 宽度)形式的裁剪尺寸
        :param step_size: (垂直步长, 水平步长)形式的步长。如果为空，则默认步长等于裁剪尺寸
        :param input_transform: 图像预处理方法
        """
        self.crop_size = crop_size
        if step_size is None:
            self.step_size = crop_size
        else:
            self.step_size = step_size
            
        self.input_transform = input_transform  
        self.im = self.image_loader(im_path)
        # 生成所有 patch 的坐标列表，格式为 (left, top, right, bottom)
        self.patches_corr = self.__getdataframes() 
         
    def __generate_crop_corrs(self, image_size):
        """
        根据图像大小、裁剪尺寸、步长生成所有裁剪区域的坐标
        :param image_size: 图像尺寸 (高度, 宽度)
        :return: patch 坐标列表，每个 patch 用 (left, top, right, bottom) 表示
        """
        H, W = image_size  # H：高度，W：宽度
        crop_h, crop_w = self.crop_size
        step_h, step_w = self.step_size
        patches = []
        for top in range(0, H, step_h):
            for left in range(0, W, step_w):
                bottom = min(top + crop_h, H)
                right  = min(left + crop_w, W)
                new_top = top
                new_left = left
                if bottom - top < crop_h:
                    new_top = max(bottom - crop_h, 0)
                if right - left < crop_w:
                    new_left = max(right - crop_w, 0)
                    
                patch = (new_left, new_top, right, bottom)
                patches.append(patch)
        return patches 
    
    def __getdataframes(self):
        """
        获取图像尺寸并生成所有的裁剪区域坐标
        """   
        width, height = self.im.size  # 注意 PIL 的 im.size 返回 (宽, 高)
        patches_corr = self.__generate_crop_corrs((height, width)) 
        return patches_corr
          
    
    def __getitem__(self, index):
        """
        根据索引获取裁剪后的图像 patch，同时返回该 patch 的位置信息
        """
        corrs = self.patches_corr[index]    
        # 注意 PIL.crop 的参数顺序就是 (left, top, right, bottom)
        img = self.im.crop(corrs)
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, corrs
    
    def __len__(self):
        return len(self.patches_corr)
     
    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

# ----------------------------------------------------------
# 2. 辅助函数：计算 IoU 和 Average Precision
# ----------------------------------------------------------
def compute_iou(boxA, boxB):
    """
    计算两个边界框之间的 Intersection over Union (IoU)
    boxA, boxB: (left, top, right, bottom)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter_area = interW * interH
    if inter_area == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

def average_precision(ranked_relevance):
    """
    计算单个 query 的 Average Precision (AP)
    :param ranked_relevance: 按相似度排序后的二值列表（1 表示正例，0 表示负例）
    """
    num_relevant = sum(ranked_relevance)
    if num_relevant == 0:
        return 0.0
    ap = 0.0
    correct = 0
    for i, rel in enumerate(ranked_relevance):
        if rel:
            correct += 1
            ap += correct / (i + 1)
    return ap / num_relevant

# ----------------------------------------------------------
# 3. 详细的 evaluate 函数
# ----------------------------------------------------------
def evaluate(model, query_loader, gallery_loader, query_bboxes, gallery_patch_coords, device='cuda', iou_thresh=0.5, topks=[1, 3, 5]):
    """
    对景象匹配任务进行评估：
      1. 对 query（无人机视图）和 gallery（底图 patch）分别提取特征
      2. 利用点积计算相似度，对 gallery 进行排序
      3. 对于每个 query，根据其真实 bbox 与 gallery patch 坐标计算 IoU，
         当 IoU >= iou_thresh 时认为该 patch 正确匹配
      4. 按排序结果计算 Recall@1、Recall@3、Recall@5 以及 Average Precision (AP)
      
    参数:
      model: 用于特征提取的模型
      query_loader: 用于加载无人机视图图片的 DataLoader（假设只返回图像张量）
      gallery_loader: 用于加载底图 patch 的 DataLoader（返回 (image, coords) 形式的 batch）
      query_bboxes: list，每个元素为 (left, top, right, bottom)，表示无人机图像的真实位置
      gallery_patch_coords: list，每个元素为 (left, top, right, bottom)，来自 TestImageCropDataset.patches_corr
      device: 'cuda' 或 'cpu'
      iou_thresh: IoU 阈值，默认 0.5
      topks: list，要计算 Recall 的 k 值，例如 [1, 3, 5]
      
    返回:
      dict，包含 "Recall@1", "Recall@3", "Recall@5", "mAP"
    """
    model.eval()
    # ---------------------------
    # 提取 query 特征
    # ---------------------------
    query_features = []
    with torch.no_grad():
        for batch in query_loader:
            # 如果 DataLoader 直接返回 image 张量
            if isinstance(batch, dict):
                images = batch['image'].to(device)
            elif isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)
            feats = model(images)
            query_features.append(feats.cpu())
    query_features = torch.cat(query_features, dim=0)
    
    # ---------------------------
    # 提取 gallery 特征
    # ---------------------------
    gallery_features = []
    with torch.no_grad():
        for batch in gallery_loader:
            # gallery_loader 返回 (image, coords) 两部分
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)
            feats = model(images)
            gallery_features.append(feats.cpu())
    gallery_features = torch.cat(gallery_features, dim=0)
    
    # ---------------------------
    # 计算相似度矩阵（点积）并评价检索效果
    # ---------------------------
    sim_matrix = torch.mm(query_features, gallery_features.t())  # 形状: [num_query, num_gallery]
    num_query = query_features.shape[0]
    if len(query_bboxes) != num_query:
        raise ValueError("query_bboxes 的数量与 query images 数量不匹配.")
    
    all_AP = []
    correct_at_k = {k: 0 for k in topks}
    
    # 遍历每一个 query
    for i in range(num_query):
        gt_bbox = query_bboxes[i]
        # 根据 IoU 判断每个 gallery patch 是否为正例（relevance）
        relevances = []
        for patch_coord in gallery_patch_coords:
            iou = compute_iou(gt_bbox, patch_coord)
            relevances.append(1 if iou >= iou_thresh else 0)
            
        # 如果当前 query 没有正例，则该 query 的 AP 置为 0
        if sum(relevances) == 0:
            ranked_relevance = [0] * len(relevances)
            query_AP = 0.0
        else:
            scores = sim_matrix[i]
            sorted_indices = torch.argsort(scores, descending=True)
            ranked_relevance = [relevances[idx] for idx in sorted_indices]
            query_AP = average_precision(ranked_relevance)
            
        all_AP.append(query_AP)
        
        # 计算 Recall@k：如果前 k 个中至少出现一个正例，则认为匹配正确
        for k in topks:
            if any(ranked_relevance[:k]):
                correct_at_k[k] += 1
                
    recall_at_k = {k: correct_at_k[k] / num_query for k in topks}
    mAP = sum(all_AP) / num_query
    
    print("Evaluation Results:")
    for k in topks:
        print(f"Recall@{k}: {recall_at_k[k]*100:.2f}%")
    print(f"Mean Average Precision (mAP): {mAP*100:.2f}%")
    
    return {"Recall@1": recall_at_k.get(1, 0),
            "Recall@3": recall_at_k.get(3, 0),
            "Recall@5": recall_at_k.get(5, 0),
            "mAP": mAP}

# ----------------------------------------------------------
# 4. 示例主函数：如何加载模型、数据并调用 evaluate 函数
# ----------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ---------------------------
    # 定义并加载模型
    # ---------------------------
    # 这里举例一个简单的 dummy 模型，实际应使用你的匹配/描述子模型
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            # L2 归一化，可方便采用点积作为相似度
            x = x / x.norm(dim=1, keepdim=True)
            return x
    
    model = DummyModel().to(device)
    
    # 加载预训练权重（你可以直接使用 torch.load，不需要使用 Configuration 类）
    checkpoint_path = 'path/to/your/model_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        model_state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(model_state_dict, strict=False)
        print("Loaded model weights.")
    else:
        print("Checkpoint not found, using randomly initialized model.")
    
    # ---------------------------
    # 准备 Query 数据集（无人机视图）
    # ---------------------------
    # 假设你的 CSV 文件格式为：image_path, left, top, right, bottom
    query_csv_path = 'path/to/drone_bboxes.csv'
    query_df = pd.read_csv(query_csv_path)
    query_image_paths = query_df['image_path'].tolist()
    # 生成每张无人机图片在底图上对应的真实 bbox（左上和右下坐标）
    query_bboxes = list(query_df[['left', 'top', 'right', 'bottom']].itertuples(index=False, name=None))
    
    # 定义一个简单的 DroneDataset，只返回图像张量（查询时 bbox 信息由外部传入）
    class DroneDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, index):
            img = Image.open(self.image_paths[index]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
            
    drone_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    query_dataset = DroneDataset(query_image_paths, drone_transform)
    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # ---------------------------
    # 准备 Gallery 数据集（底图 patches）
    # ---------------------------
    base_image_path = 'path/to/base_map.jpg'
    crop_size = (384, 384)
    step_size = (384, 384)  # 可根据需要调整重叠率、步长等参数
    gallery_dataset = TestImageCropDataset(base_image_path, crop_size, step_size, input_transform)
    # 从 gallery_dataset 中提取每个 patch 的坐标
    gallery_patch_coords = gallery_dataset.patches_corr
    
    # 因为 __getitem__ 返回 (image, coords)，所以需要自定义 collate_fn
    def collate_fn(batch):
        images, coords = zip(*batch)
        images = torch.stack(images, 0)
        return images, coords
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # ---------------------------
    # 调用 evaluate 函数进行评估
    # ---------------------------
    metrics = evaluate(model=model, 
                       query_loader=query_loader, 
                       gallery_loader=gallery_loader, 
                       query_bboxes=query_bboxes, 
                       gallery_patch_coords=gallery_patch_coords, 
                       device=device, 
                       iou_thresh=0.5,
                       topks=[1, 3, 5])
    
    print("Final evaluation metrics:", metrics)