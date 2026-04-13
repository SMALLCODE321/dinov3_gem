import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

# ==========================================
# 1. 评估
# ==========================================
def evaluate(sim_matrix, query_ids, gallery_ids, topk=[1, 5, 10]):
    indices = np.argsort(-sim_matrix, axis=1)
    pred_ids = gallery_ids[indices]
    correct = (pred_ids == query_ids[:, np.newaxis])

    print("\n" + "="*30 + "\n      VFM-Loc 测试结果\n" + "="*30)
    for k in topk:
        recall_k = np.any(correct[:, :k], axis=1).mean() * 100
        print(f"Recall@{k:<2}: {recall_k:.2f}%")

    all_ap = []
    for i in range(len(query_ids)):
        pos = np.where(correct[i])[0]
        if len(pos) > 0:
            precision = np.arange(1, len(pos) + 1) / (pos + 1)
            all_ap.append(np.mean(precision))
        else:
            all_ap.append(0)
    print(f"mAP      : {np.mean(all_ap) * 100:.2f}%\n" + "="*30)


# ==========================================
# 2. 流形对齐模块 (Progressive Alignment)
# ==========================================
class ManifoldAligner:
    def __init__(self, n_components=512):
        self.n_components = n_components
        self.pca_d = PCA(n_components=n_components, whiten=True)
        self.pca_s = PCA(n_components=n_components, whiten=True)
        self.R = None

    def _l2(self, x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    def fit(self, drone_feats, sat_feats):
        print(">>> 执行统计流形对齐 (SMA)...")
        drone_feats = self._l2(drone_feats)
        sat_feats = self._l2(sat_feats)

        d_red = self.pca_d.fit_transform(drone_feats)
        s_red = self.pca_s.fit_transform(sat_feats)

        # Orthogonal Procrustes: 求解 R 使得 ||d_red @ R - s_red|| 最小
        M = s_red.T @ d_red
        U, _, Vt = np.linalg.svd(M)
        self.R = Vt.T @ U.T 
        print(">>> 流形空间映射矩阵 R 计算完成")

    def transform_drone(self, x):
        return self.pca_d.transform(self._l2(x)) @ self.R

    def transform_sat(self, x):
        return self.pca_s.transform(self._l2(x))

    def save_calibration(self, path):
        with open(path, "wb") as f:
            pickle.dump({"pca_d": self.pca_d, "pca_s": self.pca_s, "R": self.R}, f)

    def load_calibration(self, path):
        if not os.path.exists(path): return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.pca_d, self.pca_s, self.R = data["pca_d"], data["pca_s"], data["R"]
        return True


# ==========================================
# 3. Backbone (Visual Hierarchies - Concat)
# ==========================================
class VFMBackbone(nn.Module):
    def __init__(self, model_name='dinov3_vitl16', pretrained_path=None):
        super().__init__()
        self.model = torch.hub.load('/data/xulj/dinov3-salad/models/backbones/facebookresearch/dinov3',
                                    model_name, source='local', pretrained=False)
        if pretrained_path:
            sd = torch.load(pretrained_path, map_location='cpu')
            if 'model' in sd: sd = sd['model']
            self.model.load_state_dict(sd, strict=False)
        self.num_reg = getattr(self.model, "n_storage_tokens", 0)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        B, _, H, W = x.shape
        x, hw = self.model.prepare_tokens_with_masks(x)
        rope = self.model.rope_embed(H=hw[0], W=hw[1]) if self.model.rope_embed else None

        outputs = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x, rope)
            if i >= len(self.model.blocks) - 4:
                outputs.append(x)
        
        # 拼接最后四层特征 (论文核心：保留判别性细节)
        x_concat = torch.cat(outputs, dim=-1)
        patch_tokens = x_concat[:, 1 + self.num_reg:, :]
        C = patch_tokens.shape[-1]
        return patch_tokens.reshape(B, hw[0], hw[1], C).permute(0, 3, 1, 2)


# ==========================================
# 4. 层次特征提取 (SW-RMAC, alpha=6.0)
# ==========================================
class HierarchicalClueExtractor(nn.Module):
    def __init__(self, p=3.0, alpha=6.0):
        super().__init__()
        self.p = p
        self.alpha = alpha

    def gem(self, x):
        return F.avg_pool2d(x.clamp(min=1e-6).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p).flatten(1)

    def sw_rmac(self, x, levels=2):
        B, C, H, W = x.shape
        final_feat = torch.zeros(B, C).to(x.device)
        for l in range(1, levels + 1):
            weight = 1.0 / (l ** self.alpha)
            k = (max(1, H // (l + 1) * 2), max(1, W // (l + 1) * 2))
            s = (max(1, k[0] // 2), max(1, k[1] // 2))
            pooled = F.max_pool2d(x, kernel_size=k, stride=s)
            final_feat += weight * pooled.mean(dim=(2, 3))
        return F.normalize(final_feat, p=2, dim=1)

    def forward(self, x):
        g = F.normalize(self.gem(x), p=2, dim=1)
        l = self.sw_rmac(x)
        return F.normalize(g + l, p=2, dim=1)


# ==========================================
# 5. 数据集与特征提取函数
# ==========================================
class GeoDataset(Dataset):
    def __init__(self, root, transform):
        self.imgs, self.ids, self.transform = [], [], transform
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    p = os.path.join(r, f)
                    self.imgs.append(p)
                    self.ids.append(os.path.relpath(p, root).split(os.sep)[0])
        print(f"[INFO] 加载完成: {root} -> {len(self.imgs)} images")

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return self.transform(Image.open(self.imgs[i]).convert('RGB')), self.ids[i]

def get_feats(path, backbone, hce, device, transform):
    ds = GeoDataset(path, transform)
    loader = DataLoader(ds, batch_size=24, shuffle=False, num_workers=4)
    feats, ids = [], []
    with torch.no_grad():
        for imgs, id_ in loader:
            f = hce(backbone(imgs.to(device)))
            feats.append(f.cpu().numpy())
            ids.extend(id_)
    return np.concatenate(feats), np.array(ids)

def get_paired_calib(root, backbone, hce, device, transform, max_ids=1000):
    d_base, s_base = os.path.join(root, "drone"), os.path.join(root, "satellite")
    common_ids = sorted(list(set(os.listdir(d_base)) & set(os.listdir(s_base))))[:max_ids]
    d_feats, s_feats = [], []
    print(f">>> 提取校准特征中...")
    with torch.no_grad():
        for i, obj in enumerate(common_ids):
            # Drone 均值采样
            d_dir = os.path.join(d_base, obj)
            d_p = [os.path.join(d_dir, f) for f in os.listdir(d_dir)][:4]
            d_batch = torch.stack([transform(Image.open(p).convert('RGB')) for p in d_p]).to(device)
            d_feats.append(hce(backbone(d_batch)).mean(0, keepdim=True).cpu().numpy())
            # Satellite 单采样
            s_dir = os.path.join(s_base, obj)
            s_p = os.path.join(s_dir, os.listdir(s_dir)[0])
            s_batch = transform(Image.open(s_p).convert('RGB')).unsqueeze(0).to(device)
            s_feats.append(hce(backbone(s_batch)).cpu().numpy())
    return np.concatenate(d_feats), np.concatenate(s_feats)


# ==========================================
# 6. Main
# ==========================================
def main():
    BASE = "/data/xulj/dinov3-salad/datasets/University-1652"
    WEIGHTS = "/data/xulj/dinov3-salad/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    backbone = VFMBackbone(model_name='dinov3_vitl16', pretrained_path=WEIGHTS).to(device)
    hce = HierarchicalClueExtractor(alpha=6.0).to(device)
    aligner = ManifoldAligner(n_components=512)

    calib_file = "align.pkl"
    if not aligner.load_calibration(calib_file):
        # 传入 BASE 路径下的 train 目录进行校准
        d, s = get_paired_calib(os.path.join(BASE, "train"), backbone, hce, device, transform)
        aligner.fit(d, s)
        aligner.save_calibration(calib_file)

    # 2. 提取测试集特征 (修正原本报错的地方)
    # 提取 Query (Drone) 特征
    q_raw, q_ids = get_feats(os.path.join(BASE, "test/query_drone"), backbone, hce, device, transform)
    
    # 提取 Gallery (Satellite) 特征
    g_raw, g_ids = get_feats(os.path.join(BASE, "test/gallery_satellite"), backbone, hce, device, transform)

    # 3. 流形空间对齐
    q_a = aligner.transform_drone(q_raw)
    g_a = aligner.transform_sat(g_raw)

    # 4. 再次 L2 归一化以计算余弦相似度
    q_final = q_a / (np.linalg.norm(q_a, axis=1, keepdims=True) + 1e-9)
    g_final = g_a / (np.linalg.norm(g_a, axis=1, keepdims=True) + 1e-9)

    # 5. 计算相似度矩阵并评估
    sim = q_final @ g_final.T
    evaluate(sim, q_ids, g_ids)

if __name__ == "__main__":
    main()