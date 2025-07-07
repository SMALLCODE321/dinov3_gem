import os
import glob
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import faiss
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random


def compute_mAP_cmc(index: np.ndarray,
                    good_index: np.ndarray,
                    junk_index: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute average precision and CMC.
    index: sorted gallery indices (G,)
    good_index: indices of correct matches
    junk_index: indices to ignore
    """
    G = len(index)
    cmc = np.zeros(G, dtype=int)
    if good_index.size == 0:
        cmc[0] = -1
        return 0.0, cmc
    # remove junk
    if junk_index.size > 0:
        mask = ~np.in1d(index, junk_index)
        index = index[mask]
    # find good
    mask_good = np.in1d(index, good_index)
    rows_good = np.where(mask_good)[0]
    if rows_good.size == 0:
        return 0.0, cmc
    first_hit = rows_good[0]
    cmc[first_hit:] = 1
    # AP
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


def iou(box1: Tuple[int,int,int,int],
        box2: Tuple[int,int,int,int]) -> float:
    """
    Compute IoU of two boxes (xmin, ymin, xmax, ymax).
    """
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    w = max(0, xi2 - xi1 + 1)
    h = max(0, yi2 - yi1 + 1)
    inter = w * h
    area1 = (xa2 - xa1 + 1) * (ya2 - ya1 + 1)
    area2 = (xb2 - xb1 + 1) * (yb2 - yb1 + 1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class LTAEvaluator:
    """
    FAISS-based evaluator for LTA dataset.
    Query: small drone images with place boxes in XML.
    Gallery: large panorama images, indexed via sliding windows.
    Matching by IoU > threshold.
    Reports Recall@1/3/5, Recall@top1%, mAP.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        test_path: str,
        window_size: Tuple[int,int] = (322, 322),
        stride: Tuple[int,int] = (161, 161),
        batch_size: int = 32,
        iou_threshold: float = 0.14,
        use_gpu_index: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            model:         torch model, forward(imgs)->descriptors
            test_path:     root of LTA test, with 'gallery' and 'query' subdirs
            window_size:   (h, w) for sliding window on gallery
            stride:        (sh, sw) stride for sliding window
            batch_size:    batch size for feature extraction
            iou_threshold: IoU threshold to consider a match
            use_gpu_index: whether to put FAISS index on GPU
            device:        'cuda' or 'cpu'
        """
        self.model = model
        self.test_path = test_path
        self.gallery_path = os.path.join(test_path, "gallery")
        self.query_path = os.path.join(test_path, "query")
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.iou_thr = iou_threshold
        self.use_gpu_index = use_gpu_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # input transform
        self.transform = transforms.Compose([
            transforms.Resize((322,322)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        self.model = self.model.to(self.device).eval()
        print(f"[LTAEvaluator] using model on {self.device}")

        # placeholders
        self.patch_feats: Optional[np.ndarray] = None  # (N_patches, D)
        self.patch_coords: Optional[List[Tuple[int,int,int,int]]] = None
        self.index: Optional[faiss.Index] = None

    def build_gallery_index(self):
        """
        Slide a window over each large gallery image, extract patch features, build FAISS index.
        """
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
        patch_specs = []  # each: (img_path, coords)
        for sub in sorted(os.listdir(self.gallery_path)):
            subdir = os.path.join(self.gallery_path, sub)
            if not os.path.isdir(subdir):
                continue
            imgs = []
            for e in exts:
                imgs += glob.glob(os.path.join(subdir, e))
            if not imgs:
                continue
            img_path = sorted(imgs)[0]
            # open to get size
            with Image.open(img_path) as im:
                W, H = im.size
            win_h, win_w = self.window_size
            stride_h, stride_w = self.stride
            # sliding
            ys = list(range(0, H - win_h + 1, stride_h))
            xs = list(range(0, W - win_w + 1, stride_w))
            if ys[-1] + win_h < H:
                ys.append(H - win_h)
            if xs[-1] + win_w < W:
                xs.append(W - win_w)
            for y in ys:
                for x in xs:
                    coords = (x, y, x + win_w - 1, y + win_h - 1)
                    patch_specs.append((img_path, coords))
        print(f"[Gallery] total patches: {len(patch_specs)}")

        # extract features
        feats = []
        coords_list = []
        model_dtype = torch.float16 if "cuda" in self.device else torch.float32
        with torch.no_grad():
            for i in tqdm(range(0, len(patch_specs), self.batch_size),
                          desc="Extracting gallery patches"):
                batch = patch_specs[i:i+self.batch_size]
                imgs = []
                for img_path, (x1,y1,x2,y2) in batch:
                    with Image.open(img_path) as im:
                        im = im.crop((x1, y1, x2+1, y2+1)).convert("RGB")
                    imgs.append(self.transform(im))
                    coords_list.append((x1,y1,x2,y2))
                x = torch.stack(imgs, 0).to(self.device)
                with torch.autocast(self.device.split(':')[0], model_dtype):
                    f = self.model(x)
                    f = F.normalize(f, dim=1, p=2)
                feats.append(f.cpu())
        feats = torch.cat(feats, 0).float().numpy()
        self.patch_feats = feats
        self.patch_coords = coords_list

        # FAISS
        dim = feats.shape[1]
        idx = faiss.IndexFlatL2(dim)
        if self.use_gpu_index:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
        idx.add(feats)
        self.index = idx
        print(f"[FAISS] gallery index built, total = {idx.ntotal}")

    def _parse_queries(self) -> List[Tuple[str, str, Tuple[int,int,int,int]]]:
        """
        Parse each query image, return list of (img_path, place_name, bbox)
        """
        queries = []
        exts = ("*.jpg","*.jpeg","*.png","*.bmp")
        for region in sorted(os.listdir(self.query_path)):
            rd = os.path.join(self.query_path, region)
            if not os.path.isdir(rd):
                continue
            # find xml
            xmls = glob.glob(os.path.join(rd, "*.xml"))
            if not xmls:
                continue
            xml_path = xmls[0]
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # map place -> bbox
            gt = {}
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                bb = obj.find("bndbox")
                x1 = int(bb.find("xmin").text)
                y1 = int(bb.find("ymin").text)
                x2 = int(bb.find("xmax").text)
                y2 = int(bb.find("ymax").text)
                gt[name] = (x1, y1, x2, y2)
            # find query images
            for e in exts:
                for qp in glob.glob(os.path.join(rd, e)):
                    fn = os.path.basename(qp)
                    pname = fn.split('-')[0]
                    if pname in gt:
                        queries.append((qp, pname, gt[pname]))
        print(f"[Query] total queries: {len(queries)}")
        return queries

    def evaluate(self) -> Dict[str, float]:
        """
        Full evaluation: build index, extract query features, compute Recall@1/3/5, top1%, mAP.
        """
        if self.index is None:
            self.build_gallery_index()

        queries = self._parse_queries()
        Q = len(queries)
        G = len(self.patch_feats)

        # extract query descriptors
        q_feats = []
        q_bboxes = []
        for qp, pname, bbox in queries:
            q_bboxes.append(bbox)
        # batch extract
        model_dtype = torch.float16 if "cuda" in self.device else torch.float32
        with torch.no_grad():
            for i in tqdm(range(0, Q, self.batch_size),
                          desc="Extracting query descriptors"):
                batch = queries[i:i+self.batch_size]
                imgs = []
                for qp, _, _ in batch:
                    im = Image.open(qp).convert("RGB")
                    imgs.append(self.transform(im))
                x = torch.stack(imgs,0).to(self.device)
                with torch.autocast(self.device.split(':')[0], model_dtype):
                    f = self.model(x)
                    f = F.normalize(f, dim=1, p=2)
                q_feats.append(f.cpu())
        q_feats = torch.cat(q_feats, 0).float().numpy()

        cmc_sum = np.zeros(G, dtype=int)
        ap_sum = 0.0

        print("[Eval] computing metrics ...")
        for i in tqdm(range(Q)):
            qf = q_feats[i]
            bbox = q_bboxes[i]
            # full ranking by dot product
            scores = self.patch_feats.dot(qf)
            idx_sorted = np.argsort(scores)[::-1]
            # good indices: those patches with IoU > thr
            good = []
            for j, pc in enumerate(self.patch_coords):
                if iou(pc, bbox) >= self.iou_thr:
                    good.append(j)
            good = np.array(good, dtype=int)
            junk = np.array([], dtype=int)
            ap, cmc = compute_mAP_cmc(idx_sorted, good, junk)
            if cmc[0] == -1:
                # no ground truth
                continue
            cmc_sum += cmc
            ap_sum += ap

        valid_q = Q  # we keep only queries with gt; above skip if none, but Q includes all
        cmc_avg = cmc_sum.astype(float) / valid_q
        mAP = ap_sum / valid_q * 100

        results: Dict[str, float] = {}
        for k in (1,3,5):
            if G >= k:
                results[f"recall@{k}"] = cmc_avg[k-1] * 100
        top1 = max(1, int(G * 0.01))
        results[f"recall@top1%({top1})"] = cmc_avg[top1] * 100
        results["mAP"] = mAP
        results["num_queries"] = valid_q

        print("===== LTA Evaluation =====")
        for k,v in results.items():
            if k != "num_queries":
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        return results