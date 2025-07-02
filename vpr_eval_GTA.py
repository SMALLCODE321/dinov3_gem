import os
import json
import gc
from typing import List, Dict, Optional
from sklearn.metrics import average_precision_score

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def sdm(query_loc, sdmk_list, index, gallery_loc_xy_list, s=0.001):
    query_x, query_y = query_loc
    sdm_list = []
    for k in sdmk_list:
        sdm_nom = 0.0
        sdm_den = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            d = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            sdm_nom += (k - i) / np.exp(s * d)
            sdm_den += (k - i)
        sdm_list.append(sdm_nom / sdm_den)
    return sdm_list

def get_dis(query_loc, index, gallery_loc_xy_list, disk_list, match_loc=None):
    query_x, query_y = query_loc
    dis_list = []
    for k in disk_list:
        dis_sum = 0.0
        for i in range(k):
            idx = index[i]
            gallery_x, gallery_y = gallery_loc_xy_list[idx]
            dis = np.sqrt((query_x - gallery_x)**2 + (query_y - gallery_y)**2)
            dis_sum += dis
        # For matcher estimated location
        if k == 1 and match_loc is not None:
            match_x, match_y = match_loc
            dis_list.append(np.sqrt((query_x - match_x)**2 + (query_y - match_y)**2))
        else:
            dis_list.append(dis_sum / k)
    return dis_list

class GTAEvaluator:
    """
    用于 GTAUAV 数据集的检索评估：
      - 按 test.json 构建 query/gallery 列表
      - 提取特征、计算相似度
      - 计算 Recall@1/3/5, Recall@top1%, mAP, SDM@k, Dis@d
    """
    def __init__(
        self,
        model: torch.nn.Module,
        test_json: str,
        root_dir: str,
        batch_size: int           = 32,
        device: Optional[str]     = None,
        normalize_features: bool  = True,
        topk_ranks: List[int]     = [1,3,5],
        sdmk_list: List[int]      = [3],
        disk_list: List[int]      = [1],
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = model.to(self.device).eval()
        self.batch_size = batch_size
        self.normalize  = normalize_features
        self.topk_ranks = topk_ranks
        self.sdmk_list  = sdmk_list
        self.disk_list  = disk_list

        self.transform = transforms.Compose([
            transforms.Resize((322, 322)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.root_dir  = root_dir
        self.test_json = test_json
        with open(self.test_json, 'r') as f:
            self.records = json.load(f)

        self._build_io_lists()

    def _build_io_lists(self):
        q_list, q_centers = [], []
        gal_map = {}    # gallery basename -> (fullpath, center)
        pairs = {}
        for rec in self.records:
            qname = rec["drone_img_name"]
            qpath = os.path.join(self.root_dir, rec["drone_img_dir"], qname)
            q_list.append(qpath)
            q_centers.append(rec["drone_loc_x_y"])

            pos    = rec.get("pair_pos_sate_img_list", [])
            pos_l  = rec.get("pair_pos_sate_loc_x_y_list", [])
            semi   = rec.get("pair_pos_semipos_sate_img_list", [])
            semi_l = rec.get("pair_pos_semipos_sate_loc_x_y_list", [])
            all_imgs = pos + semi
            all_locs = pos_l + semi_l
            pairs[qname] = all_imgs

            for img_name, loc in zip(all_imgs, all_locs):
                if img_name not in gal_map:
                    gpath = os.path.join(self.root_dir, rec["sate_img_dir"], img_name)
                    gal_map[img_name] = (gpath, loc)

        self.gallery_list    = []
        self.gallery_centers = []
        for img_name, (gpath, loc) in gal_map.items():
            self.gallery_list.append(gpath)
            self.gallery_centers.append(loc)

        self.query_list    = q_list
        self.query_centers = q_centers
        self.pairs_dict    = pairs
        self._gal_idx = { os.path.basename(p): i for i, p in enumerate(self.gallery_list) }

    def _extract_descriptors(self, paths: List[str]) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(paths), self.batch_size),
                          desc="Extracting features"):
                batch = paths[i:i+self.batch_size]
                imgs = []
                for p in batch:
                    im = Image.open(p).convert("RGB")
                    imgs.append(self.transform(im))
                x = torch.stack(imgs, 0).to(self.device)
                with torch.autocast(self.device, torch.float16):
                    out = self.model(x)
                if self.normalize:
                    out = F.normalize(out, dim=1, p=2)
                feats.append(out.cpu())
        return torch.cat(feats, 0).numpy()

    def compute_mAP_cmc(
        self,
        index: np.ndarray,
        good_index: np.ndarray
    ):
        """
        计算单个 query 的 AP 和 CMC 向量
        输入:
          index      -- 排序后的 gallery 下标数组，shape (G,)
          good_index -- 正例下标数组，shape (ngood,)
        输出:
          ap  -- Average Precision (float)
          cmc -- CMC 向量，shape (G,) (0/1 或 -1 表示跳过)
        """
        G = len(index)
        cmc = np.zeros(G, dtype=int)

        if good_index.size == 0:
            cmc[0] = -1
            return 0.0, cmc

        mask_good = np.in1d(index, good_index)
        rows_good = np.where(mask_good)[0]
        if rows_good.size == 0:
            return 0.0, cmc

        first_hit = rows_good[0]
        cmc[first_hit:] = 1

        # 计算 AP
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

    def evaluate(
        self,
        top10_log: bool = False
    ) -> Dict[str, float]:
        """
        运行评估：
          - Recall@1/3/5, Recall@top1%, mAP, SDM@k, Dis@d
        """
        q_feats = self._extract_descriptors(self.query_list)    # (Q, D)
        g_feats = self._extract_descriptors(self.gallery_list)  # (G, D)
        Q, G = q_feats.shape[0], g_feats.shape[0]
        scores = q_feats @ g_feats.T                            # (Q, G)

        # 初始化累加器
        cmc_sum = np.zeros(G, dtype=int)
        ap_list  = []       # 存放每个 query 的 AP
        valid_q  = 0        # 有正样本的 query 数
        sdm_sum  = np.zeros(len(self.sdmk_list), dtype=float)
        dis_sum  = np.zeros(len(self.disk_list), dtype=float)

        top10_all = [] if top10_log else None

        print(f"[GTAEval] {Q} queries, {G} gallery -> computing metrics ...")
        for i in range(Q):
            # 排序
            idx_sorted = np.argsort(scores[i])[::-1]
            # 找到正例的 gallery 下标
            qbase = os.path.basename(self.query_list[i])
            pos_names = self.pairs_dict[qbase]
            good_idx = np.array([self._gal_idx[n] for n in pos_names], dtype=int)

            # 构造 y_true, y_score（降序排列后）
            mask_good = np.in1d(idx_sorted, good_idx)     # bool array, length G
            rows_good = np.where(mask_good)[0]
            if rows_good.size == 0:
                # 该 query 无正例，跳过
                continue

            # 计算 CMC
            first_hit = rows_good[0]
            cmc_sum[first_hit:] += 1

            # 计算 AP
            y_true   = mask_good.astype(int)
            y_scores = scores[i][idx_sorted]
            ap_i = average_precision_score(y_true, y_scores)
            ap_list.append(ap_i)

            valid_q += 1

            # 计算 SDM
            sdm_vals = sdm(self.query_centers[i],
                           self.sdmk_list,
                           idx_sorted,
                           self.gallery_centers)
            sdm_sum += np.array(sdm_vals, dtype=float)

            # 计算 Dis
            dis_vals = get_dis(self.query_centers[i],
                               idx_sorted,
                               self.gallery_centers,
                               self.disk_list,
                               match_loc=None)
            dis_sum += np.array(dis_vals, dtype=float)

            # 记录 Top10（可选）
            if top10_log:
                top10 = [os.path.basename(self.gallery_list[j])
                         for j in idx_sorted[:10]]
                top10_all.append((qbase, top10))

        if valid_q == 0:
            raise RuntimeError("No valid queries with positives found.")

        # 归一化
        cmc_avg = cmc_sum.astype(float) / valid_q
        mAP     = np.mean(ap_list) * 100
        sdm_avg = sdm_sum / valid_q
        dis_avg = dis_sum / valid_q

        # 构造输出
        results: Dict[str, float] = {}
        for k in self.topk_ranks:
            if G >= k:
                results[f"recall@{k}"] = cmc_avg[k-1] * 100
        top1p = max(1, int(0.01 * G))
        results[f"recall@top1%({top1p})"] = cmc_avg[top1p] * 100
        results["mAP"] = mAP
        for idx, k in enumerate(self.sdmk_list):
            results[f"SDM@{k}"] = float(sdm_avg[idx])
        for idx, d in enumerate(self.disk_list):
            results[f"Dis@{d}"] = float(dis_avg[idx])
        results["num_queries"] = valid_q

        # 打印
        print("===== GTAUAV Evaluation Results =====")
        for k, v in results.items():
            if k != "num_queries":
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        if top10_log:
            for qbase, top10 in top10_all:
                print(f"Query {qbase} -> Top10: {top10}")

        # 清理
        del q_feats, g_feats, scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results