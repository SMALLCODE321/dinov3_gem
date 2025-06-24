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
import gc

# -------------------------------------------------------------------
# 评测相关函数：compute_mAP + eval_query
# -------------------------------------------------------------------
def compute_mAP_cmc(index, good_index, junk_index):
    """
    输入：
      index      -- 排序后的 gallery 下标数组，shape (G,)
      good_index -- 正例下标数组，shape (ngood, )
      junk_index -- 噪声下标数组，shape (njunk, )
    返回：
      ap  -- 一个 query 的 Average Precision (float)
      cmc -- 该 query 的 CMC 向量，shape (G,) (0/1)
    """
    G = len(index)
    cmc = np.zeros(G, dtype=int)

    # 没有正例
    if good_index.size == 0:
        cmc[0] = -1
        return 0.0, cmc

    # 1. 去掉 junk
    if junk_index.size > 0:
        mask = ~np.in1d(index, junk_index)
        index = index[mask]

    # 2. 找到所有正例在排序中的位置
    mask_good = np.in1d(index, good_index)
    rows_good = np.where(mask_good)[0]  # e.g. [2, 7, 15]

    # 3. 构造 CMC（first-hit 之后都算命中）
    first_hit = rows_good[0]
    cmc[first_hit:] = 1

    # 4. 计算 AP（插值法）
    ngood = good_index.size
    ap = 0.0
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision_i = (i + 1) / (rows_good[i] + 1)
        if rows_good[i] != 0:
            precision_prev = i / rows_good[i]
        else:
            precision_prev = 1.0
        ap += d_recall * (precision_prev + precision_i) / 2.0

    return ap, cmc

def eval_query(query_desc, gallery_descs, q_id, gallery_ids):
    """
    针对单个 query 特征，返回 its AP & CMC
    query_desc: np.array, (D,)
    gallery_descs: np.array, (G, D)
    q_id: int
    gallery_ids: list[int] 长度 G
    """
    # 1. 计算相似度（内积）
    scores = gallery_descs.dot(query_desc)       # (G,)
    index = np.argsort(scores)[::-1]             # 从大到小

    gl = np.array(gallery_ids)
    good_index = np.argwhere(gl == q_id).flatten()
    junk_index = np.argwhere(gl == -1).flatten()  # 如果没有 -1，则返回空

    return compute_mAP_cmc(index, good_index, junk_index)

# -------------------------------------------------------------------
# 通用工具
# -------------------------------------------------------------------
def load_model(ckpt_path, device):
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
    descriptors = []
    ids = []
    model_dtype = torch.float16 if device.startswith('cuda') else torch.float32

    def id_from_path(p):
        parent = os.path.basename(os.path.dirname(p))
        return int(parent)

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting"):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                imgs.append(input_transform(img))
                ids.append(id_from_path(p))
            x = torch.stack(imgs, dim=0).to(device)
            with torch.autocast(device_type=device.split(':')[0], dtype=model_dtype):
                feats = model(x)               # (B, D)
                feats = F.normalize(feats, dim=1, p=2)
            descriptors.append(feats.cpu().float())
    descriptors = torch.cat(descriptors, dim=0).numpy()
    return descriptors, ids

def build_faiss_index(descriptors, use_gpu=False):
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
    parser.add_argument("--gallery_path", type=str, default='/data/qiaoq/Project/salad_tz/datasets/University-1652/test/gallery_drone',
                        help="卫星图 patch 目录，文件命名：1.jpg, 2.jpg, ...")
    parser.add_argument("--query_path", type=str, default='/data/qiaoq/Project/salad_tz/datasets/University-1652/test/query_satellite',
                        help="UAV 图目录，内部若干子文件夹，子文件夹名对应 patch 编号")
    parser.add_argument("--ckpt_path", type=str, default='/data/qiaoq/Project/salad_tz/train_result/model/University-6e-5-dino-salad-10epoch.pth',
                        help="全局特征模型 checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_gpu_index", action="store_true",
                        help="是否将 FAISS 索引放到 GPU上")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 输入变换
    input_transform = transforms.Compose([
        transforms.Resize((322, 322)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])

    # 2. 加载模型
    model = load_model(args.ckpt_path, device)

    # 3. Gallery 特征
    gallery_paths = []
    for sub in os.listdir(args.gallery_path):
        subdir = os.path.join(args.gallery_path, sub)
        if not os.path.isdir(subdir): continue
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            gallery_paths += glob.glob(os.path.join(subdir, ext))
    print(f"[Gallery] found {len(gallery_paths)} images.")
    gallery_descs, gallery_ids = extract_descriptors(
        gallery_paths, model, device, input_transform, batch_size=args.batch_size
    )

    # 4. FAISS 索引
    print("[FAISS] building index ...")
    index = build_faiss_index(gallery_descs, use_gpu=args.use_gpu_index)
    G = len(gallery_ids)
    print(f"[FAISS] index size: {G}")

    # 5. Query 特征
    query_paths, query_ids = [], []
    for sub in sorted(os.listdir(args.query_path)):
        subdir = os.path.join(args.query_path, sub)
        if not os.path.isdir(subdir): continue
        try:
            qid = int(sub)
        except:
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for imgf in glob.glob(os.path.join(subdir, ext)):
                query_paths.append(imgf)
                query_ids.append(qid)
    print(f"[Query] found {len(query_paths)} images across {len(set(query_ids))} ids.")
    query_descs, _ = extract_descriptors(
        query_paths, model, device, input_transform, batch_size=args.batch_size
    )
    Q = len(query_ids)

    # 6. 对每个 query 完整排序并累加 CMC / AP
    cmc_sum = np.zeros(G, dtype=int)
    ap_sum  = 0.0

    print("[Eval] computing CMC and mAP on full gallery ...")
    for qi in tqdm(range(Q)):
        q_desc = query_descs[qi]
        q_id   = query_ids[qi]
        ap, cmc = eval_query(q_desc, gallery_descs, q_id, gallery_ids)
        if cmc[0] == -1:
            continue  # 跳过无正例的 query
        cmc_sum += cmc
        ap_sum  += ap

    # 7. 归一化并输出
    cmc_avg = cmc_sum.astype(float) / Q      # 平均 CMC
    mAP     = ap_sum / Q * 100                # 百分制

    # 常见 Recall@1/3/5
    out = []
    for k in (1,3,5):
        out.append(f"Recall@{k}: {cmc_avg[k-1]*100:.4f}")
    # top 1%
    top1 = round(G * 0.01)
    out.append(f"Recall@top1%({top1}): {cmc_avg[top1]*100:.4f}")
    out.append(f"mAP: {mAP:.4f}")

    print("\n===== Evaluation Results =====")
    print("\n".join(out))

    # 清理
    del gallery_descs, gallery_ids, query_descs, query_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()