from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parent
VFM_LOC_ROOT = Path("/data/qq/Project/qq/VFM-Loc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VFM_LOC_ROOT) not in sys.path:
    sys.path.insert(0, str(VFM_LOC_ROOT))

from vfm_loc.engine import format_metrics  # noqa: E402
from vfm_loc.utils import canonical_query_labels, label_sets  # noqa: E402
from vfm_loc.zero_shot import mean_by_id, pca_fit, pca_project, procrustes_align  # noqa: E402
from vpr_model import VPRModel  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageListDataset(Dataset):
    def __init__(self, items: Sequence[tuple[Path, int, str]], transform) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label, _ = self.items[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DINOv3-GeM checkpoint on DenseUAV test split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "train_result/model/university-1652-epoch02-92.51.ckpt",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/qq/Project/qq/data/DenseUAV"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "train_result/eval_denseuav/denseuav_metrics.json")
    parser.add_argument("--query-heights", nargs="+", default=["H80", "H90", "H100"])
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-procrustes", action="store_true")
    parser.add_argument("--max-query", type=int, default=None, help="Debug only: limit query images.")
    parser.add_argument("--max-gallery", type=int, default=None, help="Debug only: limit gallery images.")
    parser.add_argument(
        "--debug-gallery-from-query-labels",
        action="store_true",
        help="Debug only: keep gallery images whose IDs appear in the sampled queries.",
    )
    return parser.parse_args()


def build_transform(image_size: int):
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def dense_label(place_dir: Path) -> int:
    return int(place_dir.name)


def collect_gallery(data_root: Path, max_gallery: int | None = None) -> list[tuple[Path, int, str]]:
    gallery_root = data_root / "test" / "gallery_satellite"
    if not gallery_root.exists():
        raise FileNotFoundError(f"Missing DenseUAV gallery folder: {gallery_root}")

    items: list[tuple[Path, int, str]] = []
    for place_dir in sorted(path for path in gallery_root.iterdir() if path.is_dir()):
        label = dense_label(place_dir)
        for image_path in sorted(path for path in place_dir.iterdir() if is_image(path)):
            items.append((image_path, label, image_path.stem))
            if max_gallery is not None and len(items) >= max_gallery:
                return items
    if not items:
        raise RuntimeError(f"No gallery images found under {gallery_root}")
    return items


def collect_queries(data_root: Path, query_heights: Sequence[str], max_query: int | None = None) -> list[tuple[Path, int, str]]:
    query_root = data_root / "test" / "query_drone"
    if not query_root.exists():
        raise FileNotFoundError(f"Missing DenseUAV query folder: {query_root}")

    wanted = {height.upper() for height in query_heights}
    items: list[tuple[Path, int, str]] = []
    for place_dir in sorted(path for path in query_root.iterdir() if path.is_dir()):
        label = dense_label(place_dir)
        for image_path in sorted(path for path in place_dir.iterdir() if is_image(path)):
            height = image_path.stem.upper()
            if height not in wanted:
                continue
            items.append((image_path, label, height))
            if max_query is not None and len(items) >= max_query:
                return items
    if not items:
        raise RuntimeError(f"No query images found under {query_root} for heights {sorted(wanted)}")
    return items


def subset_by_height(items: Sequence[tuple[Path, int, str]], height: str) -> list[tuple[Path, int, str]]:
    height = height.upper()
    return [item for item in items if item[2].upper() == height]


def filter_gallery_from_query_labels(
    gallery_items: Sequence[tuple[Path, int, str]],
    query_items: Sequence[tuple[Path, int, str]],
    max_gallery: int | None,
) -> list[tuple[Path, int, str]]:
    query_labels = {label for _, label, _ in query_items}
    filtered = [item for item in gallery_items if item[1] in query_labels]
    if max_gallery is not None:
        filtered = filtered[:max_gallery]
    if not filtered:
        raise RuntimeError("Debug gallery filtering removed every image. Check DenseUAV label parsing.")
    return filtered


@torch.no_grad()
def extract_features(model: VPRModel, loader: DataLoader, device: torch.device, apply_adapter: bool):
    mixed_precision = device.type == "cuda"
    features, labels = [], []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            feats = model(images, apply_adapter=apply_adapter)
        features.append(feats.to(torch.float32))
        labels.append(batch_labels.to(device))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def evaluate_multi_positive_retrieval(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    ranks: Iterable[int],
    step_size: int = 256,
) -> dict[str, float]:
    ranks = sorted(set(int(rank) for rank in ranks))
    top1p = max(1, reference_features.size(0) // 100)
    max_k = max(max(ranks), top1p)
    positives = label_sets(query_labels)
    ref_labels_list = reference_labels.cpu().tolist()
    ref_label_set = set(ref_labels_list)
    matchable_queries = sum(bool(pos.intersection(ref_label_set)) for pos in positives)
    if matchable_queries == 0:
        raise RuntimeError("No query labels overlap with gallery labels. Check dataset ID matching.")

    correct = {rank: 0 for rank in ranks}
    top1p_correct = 0
    ap_sum = 0.0
    hit_rate = 0.0
    total = query_features.size(0)

    for start in range(0, total, step_size):
        end = min(total, start + step_size)
        sim = query_features[start:end] @ reference_features.T
        sorted_idx = torch.argsort(sim, dim=1, descending=True)
        topk_idx = sorted_idx[:, :max_k]
        topk_labels = reference_labels[topk_idx].cpu().tolist()

        for offset, ranked_indices in enumerate(sorted_idx.cpu().tolist()):
            query_pos = positives[start + offset]
            ranked_labels = [ref_labels_list[idx] for idx in ranked_indices]
            matches = [label in query_pos for label in ranked_labels]
            for rank in ranks:
                if any(matches[:rank]):
                    correct[rank] += 1
            if any(matches[:top1p]):
                top1p_correct += 1

            relevant_positions = [idx + 1 for idx, matched in enumerate(matches) if matched]
            if relevant_positions:
                precision_sum = 0.0
                for position in relevant_positions:
                    precision_sum += sum(matches[:position]) / position
                ap_sum += precision_sum / len(relevant_positions)

            local_topk = topk_labels[offset]
            negatives = [label for label in local_topk if label not in query_pos]
            if not negatives:
                hit_rate += 1.0

    metrics = {f"recall@{rank}": correct[rank] / total * 100.0 for rank in ranks}
    metrics["recall@top1%"] = top1p_correct / total * 100.0
    metrics["mAP"] = ap_sum / total * 100.0
    metrics["hit_rate"] = hit_rate / total * 100.0
    return metrics


@torch.no_grad()
def vfm_loc_align_and_score(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    pca_dim: int,
    ranks: Iterable[int],
    step_size: int,
    use_procrustes: bool,
):
    dim = min(pca_dim, query_features.shape[-1], gallery_features.shape[-1])
    q_mean, q_proj = pca_fit(query_features)
    g_mean, g_proj = pca_fit(gallery_features)
    query_vectors = pca_project(query_features, q_mean, q_proj, dim)
    gallery_vectors = pca_project(gallery_features, g_mean, g_proj, dim)

    if use_procrustes:
        query_ids = canonical_query_labels(query_labels)
        q_unique, q_anchors = mean_by_id(query_vectors, query_ids)
        g_unique, g_anchors = mean_by_id(gallery_vectors, gallery_labels)
        common = sorted(set(q_unique.cpu().tolist()).intersection(set(g_unique.cpu().tolist())))
        if len(common) >= 2:
            q_map = {int(label.item()): idx for idx, label in enumerate(q_unique.cpu())}
            g_map = {int(label.item()): idx for idx, label in enumerate(g_unique.cpu())}
            q_pairs = torch.stack([q_anchors[q_map[item]] for item in common], dim=0)
            g_pairs = torch.stack([g_anchors[g_map[item]] for item in common], dim=0)
            rotation = procrustes_align(q_pairs, g_pairs)
            query_vectors = query_vectors @ rotation

    query_vectors = F.normalize(query_vectors, dim=-1)
    gallery_vectors = F.normalize(gallery_vectors, dim=-1)
    return evaluate_multi_positive_retrieval(
        query_features=query_vectors,
        reference_features=gallery_vectors,
        query_labels=query_labels,
        reference_labels=gallery_labels,
        ranks=ranks,
        step_size=step_size,
    )


def make_loader(items: Sequence[tuple[Path, int, str]], transform, args, device: torch.device):
    return DataLoader(
        ImageListDataset(items, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def evaluate_split(
    name: str,
    model: VPRModel,
    query_items: Sequence[tuple[Path, int, str]],
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    transform,
    args,
    device: torch.device,
):
    split_start = time.perf_counter()
    query_loader = make_loader(query_items, transform, args, device)
    query_features, query_labels = extract_features(model, query_loader, device, apply_adapter=True)
    metrics = vfm_loc_align_and_score(
        query_features=query_features,
        query_labels=query_labels,
        gallery_features=gallery_features,
        gallery_labels=gallery_labels,
        pca_dim=args.pca_dim,
        ranks=(1, 5, 10),
        step_size=args.step_size,
        use_procrustes=not args.no_procrustes,
    )
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics["num_queries"] = len(query_items)
    metrics["elapsed_sec"] = time.perf_counter() - split_start
    print(f"[DenseUAV][{name}] {format_metrics(metrics)} | queries: {len(query_items)} | elapsed: {metrics['elapsed_sec']:.2f}s")
    return metrics


def main():
    args = parse_args()
    started_at = time.perf_counter()
    device = torch.device(args.device)
    transform = build_transform(args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[DenseUAV] checkpoint: {args.checkpoint}")
    print(f"[DenseUAV] data_root: {args.data_root}")
    print(f"[DenseUAV] device: {device}")
    print(f"[DenseUAV] query heights: {', '.join(args.query_heights)}")
    print("[DenseUAV] gallery positives are label-based: any same-ID satellite height/old variant is correct.")

    model = VPRModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.to(device)
    model.eval()

    query_items = collect_queries(args.data_root, args.query_heights, max_query=args.max_query)
    gallery_limit = None if args.debug_gallery_from_query_labels else args.max_gallery
    gallery_items = collect_gallery(args.data_root, max_gallery=gallery_limit)
    if args.debug_gallery_from_query_labels:
        gallery_items = filter_gallery_from_query_labels(gallery_items, query_items, args.max_gallery)
    print(f"[DenseUAV] gallery images: {len(gallery_items)}")
    print(f"[DenseUAV] query images: {len(query_items)}")

    gallery_start = time.perf_counter()
    gallery_loader = make_loader(gallery_items, transform, args, device)
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device, apply_adapter=False)
    print(f"[DenseUAV] gallery feature extraction: {time.perf_counter() - gallery_start:.2f}s")

    results = {}
    for height in args.query_heights:
        height_items = subset_by_height(query_items, height)
        if not height_items:
            print(f"[DenseUAV][{height}] no query images, skipped.")
            continue
        results[height] = evaluate_split(
            name=height,
            model=model,
            query_items=height_items,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
            transform=transform,
            args=args,
            device=device,
        )
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    results["all"] = evaluate_split(
        name="all",
        model=model,
        query_items=query_items,
        gallery_features=gallery_features,
        gallery_labels=gallery_labels,
        transform=transform,
        args=args,
        device=device,
    )

    results["_meta"] = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "image_size": args.image_size,
        "pca_dim": args.pca_dim,
        "use_procrustes": not args.no_procrustes,
        "query_heights": list(args.query_heights),
        "num_gallery": len(gallery_items),
        "num_query": len(query_items),
        "debug_gallery_from_query_labels": args.debug_gallery_from_query_labels,
        "total_elapsed_sec": time.perf_counter() - started_at,
    }
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[DenseUAV] saved metrics to {args.output}")
    print(f"[DenseUAV] total elapsed: {results['_meta']['total_elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
