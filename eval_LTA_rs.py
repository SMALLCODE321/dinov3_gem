import os
import cv2
import torch
import faiss
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from openpyxl import load_workbook
from vpr_model import VPRModel 
from PIL import Image


# patch_size (160,140)

# 加载模型
def load_model(model_path, device):
    if os.path.exists(model_path):
        model = torch.load(model_path)
        model.eval()
        model.to(device)
        print("Loaded model weights.")
        return model
    else:
        print("Checkpoint not found, using randomly initialized model.")

# 图像预处理
def preprocess(img_cv):
    # OpenCV 读入是 BGR，需要转为 RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((322, 322)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    return transform(img_pil)

# Patch配置
PATCH_SIZE = (200, 200)
STRIDE = (100, 100)

# ===================
# 1. 构建 FAISS satellite patch特征库
# ===================
print("began to create gallery patches faiss index...")
gallery_path = "/data/qiaoq/Project/salad_tz/datasets/real_scene_test01/basemap"
basemap_info_path = os.path.join(gallery_path, "basemap_info.txt")

# 读取 basemap 尺寸
with open(basemap_info_path, "r") as f:
    basemap_info = dict()
    for line in f:
        name, w, h = line.strip().split()
        basemap_info[name] = (int(w), int(h))

faiss_feats = []
patch_infos = []
patch_index = 0

#加载模型
model = load_model('/data/qiaoq/Project/salad_tz/train_result/model/project6e-5-5epoch.pth', 'cuda')

# for basemap_name, (width, height) in basemap_info.items():
#     #单张底图
#     image_path = os.path.join(gallery_path, basemap_name)
#     img = cv2.imread(image_path)
#     #根据patch大小和步长切patch
#     for y in range(0, height - PATCH_SIZE[1] + 1, STRIDE[1]):
#         for x in range(0, width - PATCH_SIZE[0] + 1, STRIDE[0]):
#             #单个patch的位置
#             patch = img[y:y + PATCH_SIZE[1], x:x + PATCH_SIZE[0]]
#             #预处理，resize成（322，322）
#             patch_resized = preprocess(patch).unsqueeze(0).to('cuda')
#             with torch.no_grad():
#                 feat = model(patch_resized).cpu().numpy().flatten()
#             #加到feature列表中
#             faiss_feats.append(feat)
#             #记录一个patch的信息：索引号，对应底图，（x1,y1）,(x2,y2)
#             patch_infos.append([patch_index, basemap_name, x, y, x + PATCH_SIZE[0], y + PATCH_SIZE[1]])
#             patch_index += 1

# 保存 CSV信息
csv_path = "/data/qiaoq/Project/salad_tz/datasets/real_scene_test01/patch_info.csv"
# pd.DataFrame(patch_infos, columns=["index", "basemap", "x1", "y1", "x2", "y2"]).to_csv(csv_path, index=False)

# 构建 FAISS index
faiss_index_path = "/data/qiaoq/Project/salad_tz/datasets/real_scene_test01/faiss_index.index"
index = faiss.read_index(faiss_index_path)
print(f"Loaded FAISS index, ntotal = {index.ntotal}")
# faiss_feats = np.stack(faiss_feats).astype('float32')
# index = faiss.IndexFlatL2(faiss_feats.shape[1])
# index.add(faiss_feats)
# 保存索引
# faiss.write_index(index, "/data/qiaoq/Project/salad_tz/datasets/real_scene_test01/faiss_index.index")
print("successfully create faiss vector index")

# ===================
# 2. 对每张 query 检索 Top5
# ===================
query_root = "/data/qiaoq/Project/salad_tz/datasets/real_scene_test01"
# 加载 ground_truth 文件
gt_path = os.path.join(query_root, "Coordinate Information.xlsx")
gt_info = pd.read_excel(gt_path, engine='openpyxl', dtype={'index': str})
gt_info.set_index("index", inplace=True)

retrieval_results = dict()

for area_id in tqdm(os.listdir(query_root)):
    if not area_id.isdigit():
        continue
    area_folder = os.path.join(query_root, area_id)
    # 对于每个 i，从 area_folder 中匹配所有可能后缀
    for i in range(1, 9):
        # 使用 glob 匹配所有扩展名
        pattern = os.path.join(area_folder, f"{area_id}_{i}.*")
        matches = glob.glob(pattern)
        if not matches:
            print(f"Warning: 未找到 {area_id}_{i} 对应的图像文件")
            continue
        # 取第一个匹配项
        img_path = matches[0]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # 同样的预处理
        img_tensor = preprocess(img).unsqueeze(0).to('cuda')
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().astype('float32')

        # 寻找 top5 相似的 patch 索引
        D, I = index.search(feat, 5)
        # 存储列表
        retrieval_results[f"{area_id}_{i}"] = I[0]

# ===================
# 3. 评估 Recall 和 mAP
# ===================

#计算iou
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

#获取相应patch的位置信息
patch_info_df = pd.read_csv(csv_path)

#统计recall和ap
recalls = {1: 0, 3: 0, 5: 0}
average_precisions = []

#获取真实坐标，跟patch坐标一起计算iou
for key, pred_indices in retrieval_results.items():
    area_id = key.split('_')[0]
    gt_row = gt_info.loc[area_id]
    gt_basemap = gt_row['basemap']
    gt_box = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]

    correct = []
    for idx in pred_indices:
        row = patch_info_df.iloc[idx]
        pred_basemap = row['basemap']
        pred_box = [row['x1'], row['y1'], row['x2'], row['y2']]
        #匹配成功的条件：1)底图名一致 2)iou大于0.14
        if pred_basemap == gt_basemap and iou(gt_box, pred_box) > 0.39:
            correct.append(1)
        else:
            correct.append(0)

    for k in [1, 3, 5]:
        if any(correct[:k]):
            recalls[k] += 1

    # mAP
    precision_at_i = [sum(correct[:i+1]) / (i+1) for i, c in enumerate(correct) if c]
    ap = np.mean(precision_at_i) if precision_at_i else 0.0
    average_precisions.append(ap)

total = len(retrieval_results)
print(f"Recall@1: {recalls[1]/total:.4f}")
print(f"Recall@3: {recalls[3]/total:.4f}")
print(f"Recall@5: {recalls[5]/total:.4f}")
print(f"mAP: {np.mean(average_precisions):.4f}")



