from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VFM_LOC_ROOT = Path("/data/qq/Project/qq/VFM-Loc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VFM_LOC_ROOT) not in sys.path:
    sys.path.insert(0, str(VFM_LOC_ROOT))

from vfm_loc.zero_shot import mean_by_id, pca_fit, pca_project, procrustes_align  # noqa: E402
from vpr_model import VPRModel  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/qq/Project/qq/data/CVUSA")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "train_result/model/orth_adapter-epoch19-r193.77.ckpt"
DEFAULT_OUTPUT = PROJECT_ROOT / "train_result/eval_cvusa/orth_adapter_epoch19_cvusa.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class CVUSAEvalDataset(Dataset):
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
    parser = argparse.ArgumentParser(description="Evaluate a trained DINOv3-GeM checkpoint on CVUSA.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-file", type=Path, default=None, help="Defaults to <data-root>/splits/val-19zl.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sat-size", type=int, default=336)
    parser.add_argument("--ground-width", type=int, default=None)
    parser.add_argument("--ground-height", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=512)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-procrustes", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Debug only: limit split rows.")
    return parser.parse_args()


def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def resolve_ground_size(sat_size: int, ground_width: int | None, ground_height: int | None) -> tuple[int, int]:
    width = ground_width or sat_size * 2
    if ground_height is None:
        ground_height = round((224 / 1232) * width)
    return round_up_to_multiple(int(ground_height), 16), round_up_to_multiple(int(width), 16)


def build_transform(image_size: tuple[int, int]):
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def label_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    match = re.search(r"(\d+)", stem)
    if not match:
        raise ValueError(f"Cannot parse CVUSA label from path: {path}")
    return int(match.group(1))


def resolve_image_path(data_root: Path, rel_path: str) -> Path:
    rel_path = rel_path.strip()
    path = data_root / rel_path
    if path.exists():
        return path

    rel = Path(rel_path)
    digits = re.search(r"(\d+)", rel.stem)
    if digits:
        folder = data_root / rel.parent
        for suffix in IMAGE_EXTENSIONS:
            for name in (f"{digits.group(1)}{suffix}", f"input{digits.group(1)}{suffix}"):
                candidate = folder / name
                if candidate.exists():
                    return candidate

    raise FileNotFoundError(f"Missing CVUSA image: {data_root / rel_path}")


def read_split(data_root: Path, split_file: Path, max_samples: int | None = None):
    sat_items: list[tuple[Path, int]] = []
    ground_items: list[tuple[Path, int]] = []

    with split_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            sat_rel, ground_rel = row[0], row[1]
            label = label_from_path(sat_rel)
            sat_items.append((resolve_image_path(data_root, sat_rel), label))
            ground_items.append((resolve_image_path(data_root, ground_rel), label))
            if max_samples is not None and len(sat_items) >= max_samples:
                break

    if not sat_items:
        raise RuntimeError(f"No CVUSA rows found in split file: {split_file}")
    return sat_items, ground_items


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
def vfm_loc_align(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    reference_features: torch.Tensor,
    reference_labels: torch.Tensor,
    pca_dim: int,
    use_procrustes: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    dim = min(pca_dim, query_features.shape[-1], reference_features.shape[-1])
    q_mean, q_proj = pca_fit(query_features)
    r_mean, r_proj = pca_fit(reference_features)
    query_vectors = pca_project(query_features, q_mean, q_proj, dim)
    reference_vectors = pca_project(reference_features, r_mean, r_proj, dim)

    info = {"enabled": use_procrustes, "num_pairs": 0}
    if use_procrustes:
        q_unique, q_anchors = mean_by_id(query_vectors, query_labels)
        r_unique, r_anchors = mean_by_id(reference_vectors, reference_labels)
        common = sorted(set(q_unique.cpu().tolist()).intersection(set(r_unique.cpu().tolist())))
        if len(common) >= 2:
            q_map = {int(label.item()): idx for idx, label in enumerate(q_unique.cpu())}
            r_map = {int(label.item()): idx for idx, label in enumerate(r_unique.cpu())}
            q_pairs = torch.stack([q_anchors[q_map[item]] for item in common], dim=0)
            r_pairs = torch.stack([r_anchors[r_map[item]] for item in common], dim=0)
            rotation = procrustes_align(q_pairs, r_pairs)
            query_vectors = query_vectors @ rotation
            info["num_pairs"] = len(common)

    return F.normalize(query_vectors, dim=-1), F.normalize(reference_vectors, dim=-1), info


@torch.no_grad()
def evaluate_cvusa_retrieval(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    query_labels: torch.Tensor,
    reference_labels: torch.Tensor,
    ranks: Iterable[int],
    step_size: int,
) -> dict[str, float]:
    ranks = sorted(set(int(rank) for rank in ranks))
    total = query_features.size(0)
    top1p = max(1, reference_features.size(0) // 100)
    correct = {rank: 0 for rank in ranks}
    top1p_correct = 0
    ap_sum = 0.0

    ref_labels = reference_labels.cpu().tolist()
    ref_index_by_label = {int(label): index for index, label in enumerate(ref_labels)}
    missing = [int(label) for label in query_labels.cpu().tolist() if int(label) not in ref_index_by_label]
    if missing:
        raise RuntimeError(f"{len(missing)} query labels have no matching CVUSA reference image.")

    for start in range(0, total, step_size):
        end = min(total, start + step_size)
        sim = query_features[start:end] @ reference_features.T
        sorted_idx = torch.argsort(sim, dim=1, descending=True).cpu().tolist()
        for offset, ranked_indices in enumerate(sorted_idx):
            label = int(query_labels[start + offset].item())
            gt_index = ref_index_by_label[label]
            rank = ranked_indices.index(gt_index) + 1
            for cutoff in ranks:
                if rank <= cutoff:
                    correct[cutoff] += 1
            if rank <= top1p:
                top1p_correct += 1
            ap_sum += 1.0 / rank

    metrics = {f"recall@{rank}": correct[rank] / total * 100.0 for rank in ranks}
    metrics["recall@top1%"] = top1p_correct / total * 100.0
    metrics["mAP"] = ap_sum / total * 100.0
    return metrics


def format_metrics(metrics: dict, ranks: Sequence[int]) -> str:
    parts = [f"R@{rank} {metrics[f'recall@{rank}']:.2f}" for rank in ranks]
    parts.append(f"R@top1% {metrics['recall@top1%']:.2f}")
    parts.append(f"mAP {metrics['mAP']:.2f}")
    return " | ".join(parts)


def make_loader(items: Sequence[tuple[Path, int]], transform, args, device: torch.device):
    return DataLoader(
        CVUSAEvalDataset(items, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def main():
    args = parse_args()
    started_at = time.perf_counter()
    device = torch.device(args.device)
    split_file = args.split_file or args.data_root / "splits/val-19zl.csv"
    ground_size = resolve_ground_size(args.sat_size, args.ground_width, args.ground_height)
    sat_size = (args.sat_size, args.sat_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[CVUSA] checkpoint: {args.checkpoint}")
    print(f"[CVUSA] data_root: {args.data_root}")
    print(f"[CVUSA] split_file: {split_file}")
    print(f"[CVUSA] sat_size: {sat_size}")
    print(f"[CVUSA] ground_size: {ground_size}")
    print(f"[CVUSA] device: {device}")

    sat_items, ground_items = read_split(args.data_root, split_file, max_samples=args.max_samples)
    print(f"[CVUSA] reference satellite images: {len(sat_items)}")
    print(f"[CVUSA] query ground images: {len(ground_items)}")

    model = VPRModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.to(device)
    model.eval()

    sat_loader = make_loader(sat_items, build_transform(sat_size), args, device)
    ground_loader = make_loader(ground_items, build_transform(ground_size), args, device)

    sat_start = time.perf_counter()
    reference_features, reference_labels = extract_features(model, sat_loader, device, apply_adapter=False)
    print(f"[CVUSA] satellite feature extraction: {time.perf_counter() - sat_start:.2f}s")

    ground_start = time.perf_counter()
    query_features, query_labels = extract_features(model, ground_loader, device, apply_adapter=True)
    print(f"[CVUSA] ground feature extraction: {time.perf_counter() - ground_start:.2f}s")

    align_start = time.perf_counter()
    query_vectors, reference_vectors, procrustes_info = vfm_loc_align(
        query_features=query_features,
        query_labels=query_labels,
        reference_features=reference_features,
        reference_labels=reference_labels,
        pca_dim=args.pca_dim,
        use_procrustes=not args.no_procrustes,
    )
    print(f"[CVUSA] VFM-Loc alignment: {time.perf_counter() - align_start:.2f}s")
    print(f"[CVUSA] procrustes: {procrustes_info}")

    metrics = evaluate_cvusa_retrieval(
        query_features=query_vectors,
        reference_features=reference_vectors,
        query_labels=query_labels,
        reference_labels=reference_labels,
        ranks=args.ranks,
        step_size=args.step_size,
    )
    metrics = {key: float(value) for key, value in metrics.items()}
    print(f"[CVUSA][all] {format_metrics(metrics, args.ranks)}")

    results = {
        "all": metrics,
        "_meta": {
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "split_file": str(split_file),
            "sat_size": list(sat_size),
            "ground_size": list(ground_size),
            "pca_dim": args.pca_dim,
            "use_procrustes": not args.no_procrustes,
            "procrustes": procrustes_info,
            "num_samples": len(ground_items),
            "total_elapsed_sec": time.perf_counter() - started_at,
            "positive_definition": "ground query and satellite reference share the numeric CVUSA image id",
        },
    }
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[CVUSA] saved metrics to {args.output}")
    print(f"[CVUSA] total elapsed: {results['_meta']['total_elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
