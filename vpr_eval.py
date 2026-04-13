import os
import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import faiss
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random


class VPREvaluator:
    """
    FAISS-based Visual Place Recognition evaluator.
    可以通过传入一个已加载好的 model 实例进行评测。
    输出 Recall@1/3/5, Recall@top1%, mAP（%制）。
    """
    def __init__(
        self,
        model: torch.nn.Module,
        gallery_path: str,
        query_path: str,
        batch_size: int = 32,
        use_gpu_index: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            model:          已加载好的 torch model，forward(imgs)->descriptors
            gallery_path:   gallery 根目录，子文件夹名为 int id
            query_path:     query 根目录，子文件夹名为 int id
            batch_size:     特征提取批大小
            use_gpu_index:  是否将 FAISS 索引放到 GPU
            device:         "cuda" / "cpu"，默认自动选 cuda
        """
        self.gallery_path = gallery_path
        self.query_path = query_path
        self.batch_size = batch_size
        self.use_gpu_index = use_gpu_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 输入预处理
        self.input_transform = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.430, 0.411, 0.296],
                std=[0.213, 0.156, 0.143]
            ),
        ])
        '''
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
        '''
        # Model
        self.model = model.to(self.device).eval()
        print(f"[Evaluator] using provided model on {self.device}")

        # Placeholders
        self.gallery_descs: Optional[np.ndarray] = None
        self.gallery_ids:   Optional[List[int]] = None
        self.index:         Optional[faiss.Index] = None

    @staticmethod
    def compute_mAP_cmc(index: np.ndarray,
                        good_index: np.ndarray,
                        junk_index: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        输入：
            index      -- 排序后的 gallery 下标数组，shape (G,)
            good_index -- 正例下标数组，shape (ngood,)
            junk_index -- 噪声下标数组，shape (njunk,)
        返回：
            ap  -- Average Precision (float)
            cmc -- CMC 向量，shape (G,) (0/1)
        """
        G = len(index)
        cmc = np.zeros(G, dtype=int)

        if good_index.size == 0:
            cmc[0] = -1
            return 0.0, cmc

        # 1. 去掉 junk
        if junk_index.size > 0:
            mask = ~np.in1d(index, junk_index)
            index = index[mask]

        # 2. 找到所有正例在排序中的位置
        mask_good = np.in1d(index, good_index)
        rows_good = np.where(mask_good)[0]
        if rows_good.size == 0:
            return 0.0, cmc

        # 3. 构造 CMC
        first_hit = rows_good[0]
        cmc[first_hit:] = 1

        # 4. 计算 AP
        ngood = good_index.size
        ap = 0.0
        for i, rg in enumerate(rows_good):
            d_recall = 1.0 / ngood
            precision_i = (i + 1) / (rg + 1)
            if rg != 0:
                precision_prev = i / rg
            else:
                precision_prev = 1.0
            ap += d_recall * (precision_prev + precision_i) / 2.0

        return ap, cmc

    def eval_query(self,
                   query_desc: np.ndarray,
                   q_id: int) -> Tuple[float, np.ndarray]:
        """
        对单个 query 计算 AP & CMC。
        query_desc: (D,)
        q_id:       int
        """
        # 相似度：gallery_descs dot query_desc
        scores = self.gallery_descs.dot(query_desc)
        index = np.argsort(scores)[::-1]

        gl = np.array(self.gallery_ids)
        good_index = np.argwhere(gl == q_id).flatten()
        junk_index = np.argwhere(gl == -1).flatten()  # 若无 -1，则为空

        return self.compute_mAP_cmc(index, good_index, junk_index)

    def _id_from_path(self, p: str) -> int:
        return int(os.path.basename(os.path.dirname(p)))

    def _extract_descriptors(
        self,
        image_paths: List[str]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        通过路径自动识别 drone/satellite 并执行非对称分辨率处理
        """
        all_feats = []
        all_ids = []
        model_dtype = torch.float16 if "cuda" in self.device else torch.float32
        
        # 设定你想测试的无人机降质分辨率（如果是正常评估，设为 336）
        # 汇报时可以手动改这个值，或者从 self 的某个属性读取
        target_drone_res = 128  # 比如你想测试 128px 的鲁棒性

        with torch.no_grad():
            for i in tqdm(
                range(0, len(image_paths), self.batch_size),
                desc="Extracting descriptors"
            ):
                batch = image_paths[i:i + self.batch_size]
                imgs = []
                for p in batch:
                    im = Image.open(p).convert("RGB")
                    # 根据路径判断是无人机图还是卫星图
                    if "drone" in p:
                        im = transforms.Resize((target_drone_res, target_drone_res))(im)    
                    im = self.input_transform(im)
                    
                    imgs.append(im)
                    all_ids.append(self._id_from_path(p))

                x = torch.stack(imgs, 0).to(self.device)
                with torch.autocast(self.device.split(":")[0], model_dtype):
                    feats = self.model(x)
                    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
                all_feats.append(feats.cpu())

        feats = torch.cat(all_feats, 0).float().numpy()
        return feats, all_ids

    def build_gallery_index(self):
        """
        收集 gallery 图、提取特征、构建 FAISS 索引（仅用于快速检索，不参与 mAP 计算）。
        """
        sat_non_tif_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        sat_tif_exts = ("*.tif", "*.tiff")
        g_paths = []
        for sub in os.listdir(self.gallery_path):
            subdir = os.path.join(self.gallery_path, sub)
            if not os.path.isdir(subdir):
                continue

            # 先找非 tif
            non_tif = []
            for e in sat_non_tif_exts:
                non_tif += glob.glob(os.path.join(subdir, e))
            # 再找 tif
            tif = []
            for e in sat_tif_exts:
                tif += glob.glob(os.path.join(subdir, e))

            if non_tif:
                chosen = sorted(non_tif)
            elif tif:
                chosen = [random.choice(tif)]
            else:
                # 此子文件夹下没找到任何支持的图
                continue

            for p in chosen:
                g_paths.append(p)

        # exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        # g_paths = []
        # for sub in os.listdir(self.gallery_path):
        #     d = os.path.join(self.gallery_path, sub)
        #     if not os.path.isdir(d):
        #         continue
        #     for e in exts:
        #         g_paths += glob.glob(os.path.join(d, e))
        print(f"[Gallery] found {len(g_paths)} images")

        descs, ids = self._extract_descriptors(g_paths)

        # FAISS index
        dim = descs.shape[1]
        idx = faiss.IndexFlatL2(dim)
        if self.use_gpu_index:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
        idx.add(descs)

        self.gallery_descs = descs
        self.gallery_ids = ids
        self.index = idx
        print(f"[FAISS] index built, ntotal = {idx.ntotal}")

    def _gather_query(self) -> Tuple[List[str], List[int]]:
        """
        收集 query 图路径及对应 id
        """
        exts = ("*.jpg", "*.JPG" , "*.jpeg", "*.png", "*.bmp")
        q_paths, q_ids = [], []
        for sub in sorted(os.listdir(self.query_path)):
            d = os.path.join(self.query_path, sub)
            if not os.path.isdir(d):
                continue
            try:
                tid = int(sub)
            except ValueError:
                continue
            for e in exts:
                for p in glob.glob(os.path.join(d, e)):
                    q_paths.append(p)
                    q_ids.append(tid)
        print(f"[Query] found {len(q_paths)} images, {len(set(q_ids))} classes")
        return q_paths, q_ids

    def evaluate(self) -> Dict[str, float]:
        """
        运行完整的评测：
          - 全 gallery 排序，累加 CMC & AP
          - 输出 Recall@1/3/5, Recall@top1%, mAP（%制）
        """
        if self.index is None:
            self.build_gallery_index()

        q_paths, q_ids = self._gather_query()
        q_descs, _ = self._extract_descriptors(q_paths)
        Q = len(q_ids)
        G = len(self.gallery_ids)

        cmc_sum = np.zeros(G, dtype=int)
        ap_sum = 0.0
        valid_q = 0

        print(f"[Eval] computing CMC and mAP on full gallery for {Q} queries ...")
        for i in tqdm(range(Q)):
            ap, cmc = self.eval_query(q_descs[i], q_ids[i])
            # 跳过无正例查询
            if cmc[0] == -1:
                continue
            cmc_sum += cmc
            ap_sum += ap
            valid_q += 1

        if valid_q == 0:
            raise RuntimeError("No valid queries with positive examples found.")

        cmc_avg = cmc_sum.astype(float) / valid_q
        mAP = ap_sum / valid_q * 100

        # 收集指标
        results: Dict[str, float] = {}
        for k in (1, 3, 5):
            if G >= k:
                results[f"recall@{k}"] = cmc_avg[k - 1] * 100
        top1 = max(1, int(G * 0.01))
        results[f"recall@top1%({top1})"] = cmc_avg[top1] * 100
        results["mAP"] = mAP
        results["num_queries"] = valid_q

        print("===== Evaluation Results =====")
        for k, v in results.items():
            print(f"{k}: {v:.4f}" if k != "num_queries" else f"{k}: {v}")

        return results