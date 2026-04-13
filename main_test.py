'''

import torch
import torch.nn.functional as F
import os
import csv
import numpy as np
import glob
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import OrderedDict
from vpr_model import VPRModel
from math import radians, cos, sin, asin, sqrt

# ==========================================
# 1. 地理距离计算 (Haversine 公式)
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    """计算两个经纬度点之间的地面直线距离（米）"""
    R = 6371000  # 地球半径 (米)
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return 2 * asin(sqrt(a)) * R

# ==========================================
# 2. 数据集定义
# ==========================================
class SimpleImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, path

# ==========================================
# 3. 核心评估流程
# ==========================================
def evaluate_robustness():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 路径配置 ---
    data_root = "/data/xulj/dinov3-salad/datasets/processed_VPR_dist_test"
    ckpt_path = '/data/xulj/dinov3-salad/train_result/model/14_40epoch.pth'
    
    # --- 模型初始化 ---
    model = VPRModel(
        backbone_arch='dinov3_vitb16', 
        backbone_config={'return_token': True, 'norm_layer': True, 'pretrained': False}, 
        agg_arch='GEM', 
        agg_config={'p': 3.0, 'eps': 1e-6}, 
        lr=6e-5, loss_name='MultiSimilarityLoss'
    )
    
    # --- 核心修改：手动加载权重并处理 DDP 前缀 ---
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # 如果你之前保存的是整个 model 对象而非 state_dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        # 处理 DDP 保存时产生的 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
            
        # 加载到模型中
        # strict=False 可以防止因为某些优化器参数不在 state_dict 里而报错
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}")
    else:
        print("Warning: No checkpoint found. Running with initial weights.")

    model.to(device)
    model.eval()

    # 统一转换尺寸到模型输入 (336)
    transform = transforms.Compose([
        transforms.CenterCrop(1024),
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- 第一步：提取 Gallery 特征 (卫星切片) ---
    print(">>> 提取卫星库 (Gallery) 特征...")
    # 只提取含有区域 03 和 04 的图片
    sat_files = []
    for r in ["03", "04"]:
        sat_files.extend(glob.glob(os.path.join(data_root, "gallery", f"sat_{r}_*.png")))
    sat_files = sorted(sat_files)
    
    sat_loader = DataLoader(SimpleImageDataset(sat_files, transform), batch_size=64, shuffle=False)
    sat_features, sat_coords = [], []

    with torch.no_grad():
        for imgs, paths in tqdm(sat_loader):
            feat = model(imgs.to(device))
            feat = F.normalize(feat, p=2, dim=1) # 必须做 L2 归一化
            sat_features.append(feat.cpu())
            for p in paths:
                fname = os.path.basename(p)
                # 从文件名解析: sat_03_x..._y..._lat_30.1_lon_120.1.png
                parts = fname.replace('.png', '').split('_')
                sat_coords.append([float(parts[parts.index('lat')+1]), float(parts[parts.index('lon')+1])])

    sat_features = torch.cat(sat_features, dim=0).to(device)
    sat_coords = np.array(sat_coords)

    # --- 第二步：准备 Query 数据 (无人机) ---
    uav_tasks = []
    with open(os.path.join(data_root, "query_gt.csv"), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['region'] in ["03", "04"]:
                uav_tasks.append((os.path.join(data_root, "query", row['filename']), float(row['lat']), float(row['lon'])))

    # --- 第三步：计算相似度与 Recall ---
    print(f">>> 开始评估 (Query 总数: {len(uav_tasks)})...")
    errors = []
    
    with torch.no_grad():
        for i, (path, gt_lat, gt_lon) in enumerate(tqdm(uav_tasks)):
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            q_feat = model(img_tensor)
            q_feat = F.normalize(q_feat, p=2, dim=1)
            
            # 矩阵乘法计算余弦相似度
            sim = torch.matmul(q_feat, sat_features.T)
            top1_idx = torch.argmax(sim, dim=1).item()
            
            pred_lat, pred_lon = sat_coords[top1_idx]
            dist = haversine(gt_lat, gt_lon, pred_lat, pred_lon)
            errors.append(dist)

            # 调试：打印前几个样本，确认坐标逻辑
            if i < 3:
                print(f"\n[Debug] Q: {os.path.basename(path)} | GT: ({gt_lat:.4f}, {gt_lon:.4f}) | Pred: ({pred_lat:.4f}, {pred_lon:.4f}) | Dist: {dist:.2f}m")

    # --- 第四步：统计并展示结果 ---
    errors = np.array(errors)
    thresholds = [20,50,100,200,500]
    
    print("\n" + "="*45)
    print(f" 鲁棒性测试结果 | 区域: 03, 04 ")
    print("-" * 45)
    print(f" Mean Error:    {np.mean(errors):.2f} m")
    print(f" Median Error:  {np.median(errors):.2f} m")
    print("-" * 45)
    for t in thresholds:
        recall = np.mean(errors <= t) * 100
        print(f" Recall @ {t:3}m :  {recall:.2f}%")
    print("="*45)

if __name__ == "__main__":
    evaluate_robustness()
'''
from glob import glob

import torch
import torch.nn as nn
from torchvision import transforms
from vpr_model import VPRModel, VPREvaluator
import os
from collections import OrderedDict
import glob
def zero_shot_test():
    # 1. 环境准备
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 定义模型路径
    ckpt_path = '/data/xulj/dinov3-salad/train_result/model/9_40epoch.pth'

    # 3. 实例化模型（必须先定义结构，再填入权重）
    model = VPRModel(
        backbone_arch='dinov3_vitb16',
        backbone_config={
            'num_trainable_blocks': 0, # 测试时固定 backbone
            'return_token': True,
            'norm_layer': True,
            'pretrained': False, # 关闭内部自动加载，改为下面手动加载
        },
        agg_arch='GEM',
        agg_config={
            'p': 3.0, 
            'eps': 1e-6
        },
        # 以下参数在评估模式下不生效，但实例化必须提供
        lr = 6e-5,
        optimizer='adamw',
        weight_decay=9.5e-9,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {'start_factor': 1, 'end_factor': 0.2, 'total_iters': 1000},
        loss_name='MultiSimilarityLoss'
    )

    # --- 核心修改：手动加载权重并处理 DDP 前缀 ---
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # 如果你之前保存的是整个 model 对象而非 state_dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        # 处理 DDP 保存时产生的 'module.' 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
            
        # 加载到模型中
        # strict=False 可以防止因为某些优化器参数不在 state_dict 里而报错
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}")
    else:
        print("Warning: No checkpoint found. Running with initial weights.")

    model.to(device)
    model.eval()

    # 4. 执行评估
    print(">>> 启动 Zero-shot 评估 (Drone -> Satellite)...")
    with torch.no_grad():
        evaluator = VPREvaluator(
            model=model,
            gallery_path="/data/xulj/dinov3-salad/datasets/University-1652/test/gallery_satellite",
            query_path="/data/xulj/dinov3-salad/datasets/University-1652/test/query_drone",
            batch_size=64, 
        )
        stats = evaluator.evaluate()

    print("\n" + "="*30)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    print("="*30)

if __name__ == '__main__':
    zero_shot_test()