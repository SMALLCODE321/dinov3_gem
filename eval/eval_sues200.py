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

from vfm_loc.engine import evaluate_retrieval, format_metrics  # noqa: E402
from vfm_loc.utils import canonical_query_labels  # noqa: E402
from vfm_loc.zero_shot import mean_by_id, pca_fit, pca_project, procrustes_align  # noqa: E402
from vpr_model import VPRModel  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageListDataset(Dataset):
    def __init__(self, items: Sequence[tuple[Path, int]], transform) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label = self.items[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DINOv3-GeM checkpoint on SUES-200.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "train_result/checkpoints/orth_adapter-epoch01-r192.04.ckpt",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/qq/Project/qq/data/SUES-200"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "train_result/eval_sues200/sues200_metrics.json")
    parser.add_argument("--heights", nargs="+", default=["150", "200", "250", "300"])
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-procrustes", action="store_true")
    parser.add_argument("--max-query-per-height", type=int, default=None, help="Debug only: limit queries per height.")
    parser.add_argument("--max-gallery", type=int, default=None, help="Debug only: limit gallery images.")
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


def collect_gallery(data_root: Path, max_gallery: int | None = None) -> list[tuple[Path, int]]:
    gallery_root = data_root / "satellite-view"
    if not gallery_root.exists():
        raise FileNotFoundError(f"Missing SUES-200 gallery folder: {gallery_root}")

    items: list[tuple[Path, int]] = []
    for place_dir in sorted(path for path in gallery_root.iterdir() if path.is_dir()):
        images = sorted(path for path in place_dir.iterdir() if is_image(path))
        if not images:
            continue
        items.append((images[0], int(place_dir.name)))
        if max_gallery is not None and len(items) >= max_gallery:
            break
    if not items:
        raise RuntimeError(f"No gallery images found under {gallery_root}")
    return items


def collect_queries(data_root: Path, height: str, max_query: int | None = None) -> list[tuple[Path, int]]:
    query_root = data_root / "drone_view_512"
    if not query_root.exists():
        raise FileNotFoundError(f"Missing SUES-200 query folder: {query_root}")

    items: list[tuple[Path, int]] = []
    for place_dir in sorted(path for path in query_root.iterdir() if path.is_dir()):
        height_dir = place_dir / str(height)
        if not height_dir.exists():
            continue
        for image_path in sorted(path for path in height_dir.iterdir() if is_image(path)):
            items.append((image_path, int(place_dir.name)))
            if max_query is not None and len(items) >= max_query:
                return items
    if not items:
        raise RuntimeError(f"No query images found for height {height} under {query_root}")
    return items


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
    return evaluate_retrieval(
        query_features=query_vectors,
        reference_features=gallery_vectors,
        query_labels=query_labels,
        reference_labels=gallery_labels,
        ranks=ranks,
        step_size=step_size,
    )


def main():
    args = parse_args()
    started_at = time.perf_counter()
    device = torch.device(args.device)
    transform = build_transform(args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SUES-200] checkpoint: {args.checkpoint}")
    print(f"[SUES-200] data_root: {args.data_root}")
    print(f"[SUES-200] device: {device}")
    print(f"[SUES-200] heights: {', '.join(args.heights)}")

    model = VPRModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.to(device)
    model.eval()

    gallery_items = collect_gallery(args.data_root, max_gallery=args.max_gallery)
    gallery_loader = DataLoader(
        ImageListDataset(gallery_items, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"[SUES-200] gallery images: {len(gallery_items)}")
    gallery_start = time.perf_counter()
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device, apply_adapter=False)
    print(f"[SUES-200] gallery feature extraction: {time.perf_counter() - gallery_start:.2f}s")

    all_results = {}
    for height in args.heights:
        height_start = time.perf_counter()
        query_items = collect_queries(args.data_root, height, max_query=args.max_query_per_height)
        query_loader = DataLoader(
            ImageListDataset(query_items, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        print(f"[SUES-200][{height}m] query images: {len(query_items)}")
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
        metrics["num_gallery"] = len(gallery_items)
        metrics["elapsed_sec"] = time.perf_counter() - height_start
        all_results[str(height)] = metrics
        print(f"[SUES-200][{height}m] {format_metrics(metrics)} | elapsed: {metrics['elapsed_sec']:.2f}s")

        args.output.write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")

    total_elapsed = time.perf_counter() - started_at
    all_results["_meta"] = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "image_size": args.image_size,
        "pca_dim": args.pca_dim,
        "use_procrustes": not args.no_procrustes,
        "total_elapsed_sec": total_elapsed,
    }
    args.output.write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[SUES-200] saved metrics to {args.output}")
    print(f"[SUES-200] total elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
