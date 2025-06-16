import os
import glob
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from vpr_model import VPRModel  # 请确保你的模型接口是 model(img) -> descriptor
import torch.nn.functional as F

# -------------------------------------------------------------------
# 通用工具
# -------------------------------------------------------------------
def average_precision(ranked_relevance):
    """
    计算单个 query 的 Average Precision (AP)
    ranked_relevance: 按相似度排序后的二值列表（1 表示正例，0 表示负例）
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

def load_model(ckpt_path, device):
    """
    加载模型权重并切到 eval 模式
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = torch.load(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    print(f"[Model] loaded from {ckpt_path} -> {device}")
    return model

# -------------------------------------------------------------------
# 特征提取与 FAISS 构建
# -------------------------------------------------------------------
def extract_descriptors(image_paths, model, device, input_transform, batch_size=32):
    """
    批量读取 image_paths，输出 descriptors (N, D) 及对应的原始 id 列表。
    id_fn(image_path) 应返回该图对应的整数 id（如 文件名 '123.jpg' -> 123）。
    """
    descriptors = []
    ids = []
    model_dtype = torch.float16 if device.startswith('cuda') else torch.float32

    def id_from_path(p):
        parent = os.path.basename(os.path.dirname(p))
        return int(parent)

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting gallery/query"):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                imgs.append(input_transform(img))
                ids.append(id_from_path(p))
            x = torch.stack(imgs, dim=0).to(device)
            with torch.autocast(device_type=device.split(':')[0], dtype=model_dtype):
                feats = model(x)  # (B, D)
                # 如果模型不归一化向量，需要自行归一：
                feats = F.normalize(feats, dim=1, p=2)
            descriptors.append(feats.cpu().float())
    descriptors = torch.cat(descriptors, dim=0).numpy()
    return descriptors, ids

def build_faiss_index(descriptors, use_gpu=False):
    """
    使用 L2 距离构建一个 Flat 索引
    """
    d = descriptors.shape[1]
    index = faiss.IndexFlatL2(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(descriptors)
    return index

# -------------------------------------------------------------------
# 主流程
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery_path", type=str, default='/data/qiaoq/Project/salad_tz/datasets/SUES-200-512x512/Testing/150/gallery_satellite',
                        help="卫星图 patch 目录，文件命名：1.jpg, 2.jpg, ...")
    parser.add_argument("--query_path", type=str, default='/data/qiaoq/Project/salad_tz/datasets/SUES-200-512x512/Testing/150/query_drone' ,
                        help="UAV 图目录，内部若干子文件夹，子文件夹名对应 patch 编号")
    parser.add_argument("--ckpt_path", type=str, default='/data/qiaoq/Project/salad_tz/train_result/model/model6e-5-15epoch-final.pth',
                        help="全局特征模型 checkpoint")
    parser.add_argument("--topk", type=int, default=5,
                        help="检索前 K")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_gpu_index", action="store_true",
                        help="是否将 FAISS 索引放到 GPU上")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 构造输入变换
    input_transform = transforms.Compose([
        transforms.Resize((322, 322)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载模型
    model = load_model(args.ckpt_path, device)

    # -------------------------------------------------------------------
    # 3. 处理 Gallery（卫星 patch）
    # -------------------------------------------------------------------
    #University-1652
    gallery_paths = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for sub in os.listdir(args.gallery_path):
        subdir = os.path.join(args.gallery_path, sub)
        if not os.path.isdir(subdir):
            continue
        for e in exts:
            gallery_paths += glob.glob(os.path.join(subdir, e))   #如果是仿射变换，这块是args.gallery_path
    print(f"[Gallery] found {len(gallery_paths)} patches.")
    gallery_descs, gallery_ids = extract_descriptors(
        gallery_paths, model, device, input_transform, batch_size=args.batch_size
    )

    # 4. 构建 FAISS 索引
    print("[FAISS] building index ...")
    index = build_faiss_index(gallery_descs, use_gpu=args.use_gpu_index)
    print(f"[FAISS] index size: {index.ntotal}")

    # -------------------------------------------------------------------
    # 5. 处理 Query（UAV 图）
    # -------------------------------------------------------------------
    # 扫描每个子文件夹，收集所有 query 图及其对应的真值 id
    query_image_paths = []
    query_ids = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    for sub in sorted(os.listdir(args.query_path)):
        subdir = os.path.join(args.query_path, sub)
        if not os.path.isdir(subdir):
            continue
        try:
            true_id = int(sub)
        except:
            continue
        imgs = glob.glob(os.path.join(subdir, "*.jpg"))
        for p in imgs:
            query_image_paths.append(p)
            query_ids.append(true_id)
    print(f"[Query] found {len(query_image_paths)} images across {len(set(query_ids))} folders.")

    query_descs, _ = extract_descriptors(
        query_image_paths, model, device, input_transform, batch_size=args.batch_size
    )

    # -------------------------------------------------------------------
    # 6. 在 FAISS 上检索
    # -------------------------------------------------------------------
    topk = args.topk
    print(f"[Search] running top-{topk} search for {len(query_descs)} queries ...")
    D, I = index.search(query_descs, topk)  # I: (nq, topk) 的索引到 gallery_descs

    # -------------------------------------------------------------------
    # 7. 计算 Recall@1/3/5 和 mAP
    # -------------------------------------------------------------------
    recall_counts = {1: 0, 3: 0, 5: 0}
    ap_list = []
    nq = len(query_ids)

    for qi in range(nq):
        true_id = query_ids[qi]
        retrieved_ids = [gallery_ids[idx] for idx in I[qi]]
        # 构造二值相关性列表
        rel = [1 if rid == true_id else 0 for rid in retrieved_ids]
        # Recall@K
        for k in recall_counts:
            if sum(rel[:k]) > 0:
                recall_counts[k] += 1
        # AP
        ap_list.append(average_precision(rel))

    recall1 = recall_counts[1] / nq
    recall3 = recall_counts[3] / nq
    recall5 = recall_counts[5] / nq
    mAP = np.mean(ap_list)

    # -------------------------------------------------------------------
    # 8. 输出
    # -------------------------------------------------------------------
    out = []
    out.append(f"Number of queries: {nq}")
    out.append(f"Recall@1: {recall1:.4f}")
    out.append(f"Recall@3: {recall3:.4f}")
    out.append(f"Recall@5: {recall5:.4f}")
    out.append(f"mAP: {mAP:.4f}")
    result_str = "\n".join(out)

    print("\n===== Evaluation Results =====")
    print(result_str)

if __name__ == "__main__":
    main()