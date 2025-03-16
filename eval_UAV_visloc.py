import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import os
import cv2
import glob
import faiss
import pandas as pd
from vpr_model import VPRModel 
# 使用新数据集的 Dataloader
from dataloaders.UAVVislocDataset import UAVVislocDataset  
from torchvision import transforms

# 输入图像预处理（根据模型输入需要做相应的处理）
input_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_distance(coordA, coordB):
    """
    计算两个地理坐标之间的距离，单位为米（采用 Haversine 公式）
    
    参数:
        coordA -- 第一个坐标，格式为 (latitude, longitude)，单位为度
        coordB -- 第二个坐标，格式为 (latitude, longitude)，单位为度
        
    返回:
        两点之间的距离（单位：米）
    """
    # 拆分坐标
    lat1, lon1 = coordA
    lat2, lon2 = coordB
    
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # 计算经纬度差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # 使用 Haversine 公式计算 a 的值
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # 地球半径（单位：米）
    earth_radius = 6371000
    distance = earth_radius * c
    return distance

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

def load_model(ckpt_path):
    # model = VPRModel(
    #     backbone_arch='dinov2_vitb14',
    #     backbone_config={
    #         'num_trainable_blocks': 4,
    #         'return_token': True,
    #         'norm_layer': True,
    #     },
    #     agg_arch='SALAD',
    #     agg_config={
    #         'num_channels': 768,
    #         'num_clusters': 64,
    #         'cluster_dim': 128,
    #         'token_dim': 256,
    #     },
    # )
    # loaded_state_dict = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
    # if "state_dict" in loaded_state_dict.keys():
    #     loaded_state_dict = loaded_state_dict["state_dict"]
    # model.load_state_dict(loaded_state_dict)
    # model = model.eval()
    # model = model.to('cuda')
    # print(f"Loaded model from {ckpt_path} Successfully!")
    model = torch.load(ckpt_path) 
    model.eval() 
    model = model.to('cuda')
    return model

def custom_collate_fn(batch):
    """
    自定义 collate_fn：
    - 对图片部分（Tensor）进行合并
    - 对 label 部分保持原状，不转换成 Tensor
    假设每个 sample 中只返回一张图片（即 imgs 列表只有一个元素）
    """
    for sample in batch:
        sample_imgs, sample_label = sample

    return sample_imgs, sample_label

def convert_label_coords(label):
    """
    对于 UAVVislocDataset，新数据集的 label 已经是 UAV 图中心点的经纬度坐标
    假设格式为:
       - tuple: (latitude, longitude)
       - 或者字典格式: {'center': (latitude, longitude)}
    """
    if "coords" in label:
        coords = label["coords"]
        try:
            def get_val(val):
                return int(val.item()) if hasattr(val, "item") else int(val)
            x1 = get_val(coords[0][0])
            y1 = get_val(coords[1][0])
            label["coords"] = (x1, y1)
        except Exception as e:
            print("转换 coords 时出错：", e)
    return label

def get_query_descriptors(model, dataloader, device):
    """
    对于每个样本，不对多张 UAV 图片做平均处理，而是每张图片独立检索。
    返回：
      - descriptors: [N, feature_dim]，N 为所有 UAV 图片数量
      - labels_all: 长度 N 的列表，每个元素为对应的标签（中心点经纬度）
    """
    descriptors = []
    labels_all = []
    # 使用自动混合精度（若 device 为 cuda，则采用 float16，否则 float32）
    dtype = torch.float16 if device == 'cuda' else torch.float32
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            # dataloader 每个 batch 返回 (imgs, label)
            for batch in tqdm(dataloader, desc='Calculating descriptors...'):
                batch_imgs, batch_labels = batch
                # label = convert_label_coords(batch_labels)
                # 遍历当前 batch 内的每个 UAV 图片
                for img in batch_imgs:
                    # img 已经过 transform 处理，shape 如 (3, H, W)
                    img_tensor = img.to(device)
                    desc = model(img_tensor.unsqueeze(0))  # 增加 batch 维度
                    descriptors.append(desc.cpu())
                    # 每张 UAV 图片都使用相同 label，保存在列表中
                    labels_all.append(batch_labels)
    descriptors = torch.cat(descriptors, dim=0)
    return descriptors, labels_all

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 测试文件夹路径，每个子文件夹代表一个底图，可按需调整
    parser.add_argument("--val_path", type=str, default='./data/test_images', 
                        help="测试文件夹路径")
    parser.add_argument('--output_file', type=str, default='/data/qiaoq/Project/salad_tz/train_result/model/result-distance_threshold-tzb.txt', 
                        help="结果保存文件")
    parser.add_argument("--ckpt_path", type=str, default='./dino_salad.ckpt', 
                        help="模型 checkpoint 路径")
    parser.add_argument("--index_path", type=str, default='./index.faiss', 
                        help="保存好的 FAISS 索引文件位置")
    parser.add_argument("--index_info_csv", type=str, default='./index_info.csv', 
                        help="包含 patch 信息的 CSV 文件，字段至少包含: image_name, center_lat, center_lon")
    parser.add_argument('--image_size', type=lambda s: tuple(map(int, s.split(','))), default=None, 
                        help='全局特征提取器的输入尺寸，例如 "384,384"')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch 大小')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='worker 数量')
    parser.add_argument('--topk_index', type=int, default=5, 
                        help='FAISS 检索返回前 K 个候选 patch')
    # 这里原参数名为 iou_threshold，但现在表示中心点之间的距离阈值
    parser.add_argument('--distance_threshold', type=float, default=500, 
                        help='距离阈值（单位与经纬度一致），若 patch 与无人机图中心点之间的距离小于此值，认为匹配成功')
    args = parser.parse_args()
    return args 

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型（全局特征提取模型）
    model = load_model(args.ckpt_path)
    if model is None:
        return

    # 构造新数据集，使用 UAVVislocDataset（__getitem__ 返回 (imgs, label)）
    val_dataset = UAVVislocDataset(im_path=args.val_path, image_size=args.image_size)
    data_loader = DataLoader(val_dataset, 
                             num_workers=args.num_workers, 
                             batch_size=args.batch_size,
                             shuffle=False, 
                             drop_last=False, 
                             pin_memory=True,
                             collate_fn=custom_collate_fn)
    
    # 针对每个 UAV 图片单独提取描述符，同时保留对应标签（中心点坐标）
    print("Extracting query descriptors ...")
    query_descriptors, query_labels = get_query_descriptors(model, data_loader, device)

    # 加载 FAISS 索引及其信息 CSV 文件
    print("Loading FAISS index from:", args.index_path)
    index = faiss.read_index(args.index_path) 
    index_info_csv = pd.read_csv(args.index_info_csv)  
    # 假设 CSV 中包含字段：image_name, center_lat, center_lon

    # 利用 FAISS 对所有查询描述符进行搜索，返回 topk_index 个候选 patch
    print("Performing FAISS search ...")
    query_descriptors_np = query_descriptors.numpy()
    distances, indices = index.search(query_descriptors_np, args.topk_index)
    index.reset()
    del index

    # 对每个查询计算匹配情况，并记录指标
    recall1, recall3, recall5 = 0, 0, 0
    ap_list = []
    num_valid_queries = 0

    # 遍历每个查询（每个 UAV 图片）
    for i, label in tqdm(enumerate(query_labels), total=len(query_labels), desc="Evaluating queries"):
        # 获取 UAV 图片的中心点坐标
        if isinstance(label, dict):
            query_center = label.get("coords", None)
        else:
            query_center = label
        # 若中心点无效，则跳过
        if query_center is None or len(query_center) != 2:
            continue

        num_valid_queries += 1
        ranked_relevance = []
        # 对当前查询的候选 patch 进行遍历
        for j in range(args.topk_index):
            patch_idx = indices[i][j]
            row = index_info_csv.iloc[patch_idx]
            patch_center = (row["center_lat"], row["center_lon"])
            dist = compute_distance(query_center, patch_center)
            # 如果两个中心点之间的距离小于阈值，则认为匹配成功
            relevance = 1 if dist < args.distance_threshold else 0
            ranked_relevance.append(relevance)
        # Recall 评价：只要候选列表中前 k 个中至少有一个符合匹配条件则算命中
        if ranked_relevance[0] == 1:
            recall1 += 1
        if sum(ranked_relevance[:3]) > 0:
            recall3 += 1
        if sum(ranked_relevance[:5]) > 0:
            recall5 += 1
        ap = average_precision(ranked_relevance)
        ap_list.append(ap)

    if num_valid_queries == 0:
        print("没有有效的 query 标注，可能图片没有中心点信息，无法计算指标！")
        return

    overall_recall1 = recall1 / num_valid_queries
    overall_recall3 = recall3 / num_valid_queries
    overall_recall5 = recall5 / num_valid_queries
    mAP = np.mean(ap_list) if len(ap_list) > 0 else 0.0

    # 将结果保存到输出文件中
    with open(args.output_file, 'w') as result_writer:
        result_writer.write(f"Number of valid queries: {num_valid_queries}\n")
        result_writer.write(f"Recall@1: {overall_recall1:.4f}\n")
        result_writer.write(f"Recall@3: {overall_recall3:.4f}\n")
        result_writer.write(f"Recall@5: {overall_recall5:.4f}\n")
        result_writer.write(f"mAP: {mAP:.4f}\n")

    print("Evaluation complete.")
    print(f"Valid queries: {num_valid_queries}")
    print(f"Recall@1: {overall_recall1:.4f}")
    print(f"Recall@3: {overall_recall3:.4f}")
    print(f"Recall@5: {overall_recall5:.4f}")
    print(f"mAP: {mAP:.4f}")

if __name__ == '__main__':
    main()