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
from dataloaders.LTA_Dataset import ValDataset
from torchvision import transforms

input_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

def load_model(model_path, device):
    if os.path.exists(model_path):
        model = torch.load(model_path)
        model.eval()
        model.to(device)
        print("Loaded model weights.")
        return model
    else:
        print("Checkpoint not found, using randomly initialized model.")

def convert_label_coords(label):
    """
    转换 label 中 'coords' 字段，假设格式为：
      'coords': [[[x1], [y1]], [[x2], [y2]]]
    转换为：((x1, y1), (x2, y2))，确保 x1, y1, x2, y2 均为整型。
    """
    if "coords" in label:
        coords = label["coords"]
        try:
            def get_val(val):
                return int(val.item()) if hasattr(val, "item") else int(val)
            x1 = get_val(coords[0][0][0])
            y1 = get_val(coords[0][1][0])
            x2 = get_val(coords[1][0][0])
            y2 = get_val(coords[1][1][0])
            label["coords"] = ((x1, y1), (x2, y2))
        except Exception as e:
            print("转换 coords 时出错：", e)
    return label

def get_query_descriptors(model, dataloader, device):
    """
    对于每个样本，不对多张 UAV 图片做平均处理，
    而是每张图片独立检索，所有 UAV 图片共享同一个 label。
    返回：
      - descriptors: [N, feature_dim]，N 为所有 UAV 图片数量
      - labels_all: 长度 N 的列表，每个元素为字典，包含 label 信息
    """
    descriptors = []
    labels_all = []
    # 使用自动混合精度（若 device 为 cuda 则采用 float16，否则保持 float32）
    dtype = torch.float16 if device == 'cuda' else torch.float32
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            # dataloader 每个 batch 返回 (list of imgs, list of label)，
            # 其中每个 imgs 为一个列表（该样本中的所有 UAV 图片）
            for batch in tqdm(dataloader, desc='Calculating descriptors...'):
                batch_imgs, batch_labels = batch
                label = convert_label_coords(batch_labels)
                # 遍历当前 batch 内的每个样本
                for img in batch_imgs:
                    # 对该样本中的每张 UAV 图片单独计算描述符
                        # img 已经过 transform 处理，shape 如 (3, H, W)
                        img_tensor = img.to(device)  # shape: (1, 3, H, W)
                        desc = model(img_tensor)  # shape: (1, feature_dim)
                        descriptors.append(desc.cpu())
                        # 每张 UAV 图片都使用相同 label，保存到列表中
                        labels_all.append(label)
    descriptors = torch.cat(descriptors, dim=0)
    return descriptors, labels_all

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--val_path", type=str, default='./data/test_images', 
                        help="测试文件夹路径，每个子文件夹代表一个底图")
    parser.add_argument('--output_file', type=str, default='/data/qiaoq/Project/salad_tz/train_result/model/result-iou0.1-model6e-5-20epoch.txt', 
                        help="结果保存文件")
    parser.add_argument("--ckpt_path", type=str, default='./dino_salad.ckpt', 
                        help="模型 checkpoint 路径")
    parser.add_argument("--index_path", type=str, default='./index.faiss', 
                        help="保存好的 FAISS 索引文件位置")
    parser.add_argument("--index_info_csv", type=str, default='./index_info.csv', 
                        help="包含 patch 信息的 CSV 文件，字段至少包括: image_name, left, top, right, bottom")
    parser.add_argument('--image_size', type=lambda s: tuple(map(int, s.split(','))), default=None, 
                        help='全局特征提取器的输入尺寸，例如 "384,384"')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch 大小')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='worker 数量')
    parser.add_argument('--topk_index', type=int, default=5, 
                        help='FAISS 检索返回前 K 个候选 patch')
    parser.add_argument('--iou_threshold', type=float, default=0.39, 
                        help='IoU 阈值，若 patch 与标注框 IoU 大于此值，认为匹配成功')
    args = parser.parse_args()
    return args 

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型（全局特征提取模型）
    model = load_model(args.ckpt_path, device)

    # 构造验证数据集，使用更新后的 ValDataset（每个样本返回 (imgs, label)）
    val_dataset = ValDataset(im_path=args.val_path, image_size=args.image_size)
    data_loader = DataLoader(val_dataset, 
                             num_workers=args.num_workers, 
                             batch_size=args.batch_size,
                             shuffle=False, 
                             drop_last=False, 
                             pin_memory=True)
    
    # 针对每个 UAV 图片单独提取描述符，同时保留对应 label 信息
    print("Extracting query descriptors ...")
    query_descriptors, query_labels = get_query_descriptors(model, data_loader, device)

    # 加载 FAISS 索引及其信息 CSV 文件
    print("Loading FAISS index from:", args.index_path)
    index = faiss.read_index(args.index_path) 
    index_info_csv = pd.read_csv(args.index_info_csv)  
    # 假设 CSV 中包含字段：image_name, left, top, right, bottom

    # 利用 FAISS 对所有查询描述符进行搜索，返回 topk_index 个候选 patch
    print("Performing FAISS search ...")
    query_descriptors_np = query_descriptors.numpy()
    distances, indices = index.search(query_descriptors_np, args.topk_index)
    index.reset()
    del index

    # 对每个查询计算 IoU 得分，并记录是否命中
    recall1, recall3, recall5 = 0, 0, 0
    ap_list = []
    num_valid_queries = 0

    # 遍历每个查询（每个 UAV 图片）
    for i, label in tqdm(enumerate(query_labels), total=len(query_labels), desc="Evaluating queries"):
        # 每个 label 为字典，包含 "coords": ((d1_x, d1_y), (d2_x, d2_y)) 和 "base_img"
        bndbox = label.get("bndbox", None)
        # 若标注框无效，则跳过（例如未正确标注或解析失败）
        if bndbox is None:
            continue

        num_valid_queries += 1
        # 将标注框转换为 (left, top, right, bottom)
        query_box = (int(bndbox[0]), int(bndbox[1]), int(bndbox[2]), int(bndbox[3]))
        ranked_relevance = []
        # 针对当前查询得到 FAISS 检索返回的前 topk_index 个候选 patch
        for j in range(args.topk_index):
            patch_idx = indices[i][j]
            row = index_info_csv.iloc[patch_idx]
            patch_box = (row["x1"], row["y1"], row["x2"], row["y2"])
            iou = compute_iou(query_box, patch_box)
            relevance = 1 if iou >= args.iou_threshold else 0
            ranked_relevance.append(relevance)
        # Recall 评价：只要候选列表中前 k 个中至少有一个匹配（即 relevance 为 1）则算命中
        if ranked_relevance[0] == 1:
            recall1 += 1
        if sum(ranked_relevance[:3]) > 0:
            recall3 += 1
        if sum(ranked_relevance[:5]) > 0:
            recall5 += 1
        ap = average_precision(ranked_relevance)
        ap_list.append(ap)

    if num_valid_queries == 0:
        print("没有有效的 query 标注（可能标注框为空），无法计算指标！")
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