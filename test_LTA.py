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
from torchvision import transforms
from PIL import Image

# -- keep the same normalization / resize as before
input_transform = transforms.Compose([
    transforms.Resize((322, 322)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path, device):
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=device)
        model.eval()
        model.to(device)
        print("Loaded model weights.")
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--ckpt_path", type=str, required=True, 
                        help="模型 checkpoint 路径")
    parser.add_argument("--index_path", type=str, required=True, 
                        help="保存好的 FAISS 索引文件路径")
    parser.add_argument("--index_info_csv", type=str, required=True, 
                        help="包含 patch 信息的 CSV 文件，字段: image_name, x1, y1, x2, y2")
    parser.add_argument("--query_folder", type=str, required=True,
                        help="存放 query 图像的文件夹路径")
    parser.add_argument("--base_image_dir", type=str, required=True,
                        help="存放原始底图的文件夹路径，用于裁剪 patch")
    parser.add_argument("--output_crop_dir", type=str, default="./output_crops",
                        help="保存裁剪后 top-K patch 的文件夹")
    parser.add_argument("--topk_crop", type=int, default=3,
                        help="裁剪并保存最相似的前 K 个 patch")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu, 默认为自动检测")
    return parser.parse_args()

def test_and_crop(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    model = load_model(args.ckpt_path, device)

    # 2. Load FAISS index & CSV
    print("Loading FAISS index from:", args.index_path)
    index = faiss.read_index(args.index_path)
    index_info = pd.read_csv(args.index_info_csv)
    # Expect columns: ['image_name', 'x1', 'y1', 'x2', 'y2']

    # 3. Prepare output directory
    os.makedirs(args.output_crop_dir, exist_ok=True)

    # 4. Iterate over query images
    query_paths = glob.glob(os.path.join(args.query_folder, "*.*"))
    print(f"Found {len(query_paths)} query images in {args.query_folder}")

    for qpath in tqdm(query_paths, desc="Processing queries"):
        qname = os.path.splitext(os.path.basename(qpath))[0]
        # load & preprocess
        img = Image.open(qpath).convert("RGB")
        img_t = input_transform(img).unsqueeze(0).to(device)  # (1,3,H,W)
        
        # forward
        with torch.no_grad():
            desc = model(img_t)  # (1, D)
        desc_np = desc.cpu().numpy().astype('float32')

        # search
        D, I = index.search(desc_np, args.topk_crop)  # shapes (1, K)
        top_idxs = I[0]  # [k]

        # create a subfolder for this query
        out_folder = os.path.join(args.output_crop_dir, qname)
        os.makedirs(out_folder, exist_ok=True)

        # for each top-k
        for rank, patch_idx in enumerate(top_idxs, start=1):
            row = index_info.iloc[patch_idx]
            base_name = row["image_name"]
            x1, y1, x2, y2 = map(int, (row["x1"], row["y1"], row["x2"], row["y2"]))

            # load base image
            base_path = os.path.join(args.base_image_dir, base_name)
            if not os.path.exists(base_path):
                print(f"Warning: base image not found: {base_path}")
                continue
            base_img = cv2.imread(base_path)  # BGR
            crop = base_img[y1:y2, x1:x2]     # note cv2 uses [y1:y2, x1:x2]

            # save crop
            patch_base = os.path.splitext(base_name)[0]
            out_name = f"{qname}_rank{rank}_from_{patch_base}.jpg"
            out_path = os.path.join(out_folder, out_name)
            cv2.imwrite(out_path, crop)

        # optionally, save the query itself
        qsave = os.path.join(out_folder, f"{qname}_query.jpg")
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(qsave, img_np)

    print("All queries processed. Cropped patches are saved under:", args.output_crop_dir)

def main():
    args = parse_args()
    test_and_crop(args)

if __name__ == '__main__':
    main()