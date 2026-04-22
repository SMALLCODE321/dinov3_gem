from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parent
VFM_LOC_ROOT = Path("/data/qq/Project/qq/VFM-Loc")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(VFM_LOC_ROOT) not in sys.path:
    sys.path.insert(0, str(VFM_LOC_ROOT))

from vfm_loc.zero_shot import pca_fit, pca_project, procrustes_align  # noqa: E402
from vpr_model import VPRModel  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_DATA_ROOT = Path("/data/qq/Project/qq/data/UAV_VisLoc_dataset")
DEFAULT_PATCH_INDEX = PROJECT_ROOT / "train_result/eval_uav_visloc/patch_indices/uav_visloc_crop1536x1024_stride768x512.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "train_result/model/university-1652-epoch02-92.51.ckpt"
DEFAULT_OUTPUT = PROJECT_ROOT / "train_result/eval_uav_visloc/uav_visloc_metrics.json"
EARTH_RADIUS_M = 6371008.8


class QueryImageDataset(Dataset):
    def __init__(self, items: Sequence[dict], transform) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path = Path(self.items[index]["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(index, dtype=torch.long)


class SatellitePatchDataset(Dataset):
    def __init__(self, patches: Sequence[dict], transform) -> None:
        self.patches = list(patches)
        self.transform = transform
        self._image_cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.patches)

    def _get_satellite(self, path: str) -> Image.Image:
        image = self._image_cache.get(path)
        if image is None:
            image = Image.open(path)
            self._image_cache[path] = image
        return image

    def __getitem__(self, index: int):
        patch = self.patches[index]
        image = self._get_satellite(patch["satellite_path"])
        crop = image.crop((patch["x1"], patch["y1"], patch["x2"], patch["y2"])).convert("RGB")
        return self.transform(crop), torch.tensor(index, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DINOv3-GeM checkpoint on UAV-VisLoc.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--patch-index", type=Path, default=DEFAULT_PATCH_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument(
        "--distance-thresholds",
        nargs="+",
        type=float,
        default=[],
        help="Optional distance thresholds in meters. Disabled by default.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-procrustes", action="store_true")
    parser.add_argument("--max-query", type=int, default=None, help="Debug only: limit query images.")
    parser.add_argument("--max-patches", type=int, default=None, help="Debug only: limit gallery patches.")
    parser.add_argument("--save-rankings", action="store_true", help="Save top-k ranking details for debugging.")
    return parser.parse_args()


def build_transform(image_size: int):
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def read_patch_index(path: Path, max_patches: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing patch index: {path}. Build it first with utils/build_patch_index.py."
        )

    patches: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patch = dict(row)
            patch["x1"] = int(float(patch["x1"]))
            patch["y1"] = int(float(patch["y1"]))
            patch["x2"] = int(float(patch["x2"]))
            patch["y2"] = int(float(patch["y2"]))
            patch["center_lat"] = float(patch["center_lat"])
            patch["center_lon"] = float(patch["center_lon"])
            patch["lat_top"] = float(patch["lat_top"])
            patch["lat_bottom"] = float(patch["lat_bottom"])
            patch["lon_left"] = float(patch["lon_left"])
            patch["lon_right"] = float(patch["lon_right"])
            patch["area_id"] = str(patch["area_id"]).zfill(2)
            patches.append(patch)
            if max_patches is not None and len(patches) >= max_patches:
                break

    if not patches:
        raise RuntimeError(f"No patches found in {path}")
    return patches


def read_queries(data_root: Path, max_query: int | None = None) -> list[dict]:
    queries: list[dict] = []
    for area_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        area_id = area_dir.name.zfill(2)
        csv_path = area_dir / f"{area_id}.csv"
        drone_dir = area_dir / "drone"
        if not csv_path.exists() or not drone_dir.exists():
            continue

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = drone_dir / row["filename"]
                if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                queries.append(
                    {
                        "query_id": f"{area_id}_{row['filename']}",
                        "area_id": area_id,
                        "filename": row["filename"],
                        "image_path": str(image_path),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "height": float(row["height"]),
                    }
                )
                if max_query is not None and len(queries) >= max_query:
                    return queries

    if not queries:
        raise RuntimeError(f"No UAV-VisLoc query images found under {data_root}")
    return queries


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def patch_contains_query(patch: dict, query: dict) -> bool:
    lat_min = min(patch["lat_top"], patch["lat_bottom"])
    lat_max = max(patch["lat_top"], patch["lat_bottom"])
    lon_min = min(patch["lon_left"], patch["lon_right"])
    lon_max = max(patch["lon_left"], patch["lon_right"])
    return lat_min <= query["lat"] <= lat_max and lon_min <= query["lon"] <= lon_max


def build_area_to_patch_indices(patches: Sequence[dict]) -> dict[str, list[int]]:
    area_to_indices: dict[str, list[int]] = {}
    for index, patch in enumerate(patches):
        area_to_indices.setdefault(patch["area_id"], []).append(index)
    return area_to_indices


def nearest_patch_index(query: dict, patches: Sequence[dict], candidate_indices: Sequence[int]) -> tuple[int, float]:
    best_index = -1
    best_distance = float("inf")
    for patch_index in candidate_indices:
        patch = patches[patch_index]
        distance = haversine_m(query["lat"], query["lon"], patch["center_lat"], patch["center_lon"])
        if distance < best_distance:
            best_index = patch_index
            best_distance = distance
    return best_index, best_distance


def positive_patch_indices(
    query: dict,
    patches: Sequence[dict],
    area_to_indices: dict[str, list[int]],
) -> tuple[set[int], int | None]:
    candidates = area_to_indices.get(query["area_id"], [])
    positives = {index for index in candidates if patch_contains_query(patches[index], query)}
    if positives:
        return positives, None

    fallback, _ = nearest_patch_index(query, patches, candidates)
    if fallback < 0:
        return set(), None
    return {fallback}, fallback


@torch.no_grad()
def extract_features(model: VPRModel, loader: DataLoader, device: torch.device, apply_adapter: bool):
    mixed_precision = device.type == "cuda"
    features, indices = [], []
    for images, batch_indices in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            feats = model(images, apply_adapter=apply_adapter)
        features.append(feats.to(torch.float32))
        indices.append(batch_indices)

    features = torch.cat(features, dim=0)
    indices = torch.cat(indices, dim=0)
    order = torch.argsort(indices)
    return features[order]


@torch.no_grad()
def align_features(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    queries: Sequence[dict],
    patches: Sequence[dict],
    area_to_indices: dict[str, list[int]],
    pca_dim: int,
    use_procrustes: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    dim = min(pca_dim, query_features.shape[-1], gallery_features.shape[-1])
    q_mean, q_proj = pca_fit(query_features)
    g_mean, g_proj = pca_fit(gallery_features)
    query_vectors = pca_project(query_features, q_mean, q_proj, dim)
    gallery_vectors = pca_project(gallery_features, g_mean, g_proj, dim)

    procrustes_info = {"enabled": use_procrustes, "num_pairs": 0, "num_missing_anchor": 0}
    if use_procrustes:
        q_pairs, g_pairs = [], []
        for query_index, query in enumerate(queries):
            anchor_index, _ = nearest_patch_index(query, patches, area_to_indices.get(query["area_id"], []))
            if anchor_index < 0:
                procrustes_info["num_missing_anchor"] += 1
                continue
            q_pairs.append(query_vectors[query_index])
            g_pairs.append(gallery_vectors[anchor_index])

        if len(q_pairs) >= 2:
            q_pair_tensor = torch.stack(q_pairs, dim=0)
            g_pair_tensor = torch.stack(g_pairs, dim=0)
            rotation = procrustes_align(q_pair_tensor, g_pair_tensor)
            query_vectors = query_vectors @ rotation
            procrustes_info["num_pairs"] = len(q_pairs)

    return F.normalize(query_vectors, dim=-1), F.normalize(gallery_vectors, dim=-1), procrustes_info


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def evaluate_geoloc_retrieval(
    query_vectors: torch.Tensor,
    gallery_vectors: torch.Tensor,
    queries: Sequence[dict],
    patches: Sequence[dict],
    area_to_indices: dict[str, list[int]],
    ranks: Iterable[int],
    distance_thresholds: Iterable[float],
    step_size: int,
    save_rankings: bool = False,
):
    ranks = sorted(set(int(rank) for rank in ranks))
    max_rank = min(max(ranks), gallery_vectors.size(0))
    distance_thresholds = sorted(float(threshold) for threshold in distance_thresholds)
    correct = {rank: 0 for rank in ranks}
    topk_min_distances = {rank: [] for rank in ranks}
    top1_distances: list[float] = []
    ap_sum = 0.0
    fallback_positive_count = 0
    no_positive_count = 0
    rankings = []
    total = len(queries)

    positive_cache: list[set[int]] = []
    for query in queries:
        positives, fallback = positive_patch_indices(query, patches, area_to_indices)
        if fallback is not None:
            fallback_positive_count += 1
        if not positives:
            no_positive_count += 1
        positive_cache.append(positives)

    for start in range(0, total, step_size):
        end = min(total, start + step_size)
        sim = query_vectors[start:end] @ gallery_vectors.T
        sorted_idx = torch.argsort(sim, dim=1, descending=True)

        for offset, ranked_indices in enumerate(sorted_idx.cpu().tolist()):
            query_index = start + offset
            query = queries[query_index]
            positives = positive_cache[query_index]
            matches = [patch_index in positives for patch_index in ranked_indices]

            for rank in ranks:
                rank_limit = min(rank, len(matches))
                if any(matches[:rank_limit]):
                    correct[rank] += 1

            relevant_positions = [idx + 1 for idx, matched in enumerate(matches) if matched]
            if relevant_positions:
                precision_sum = 0.0
                for position in relevant_positions:
                    precision_sum += sum(matches[:position]) / position
                ap_sum += precision_sum / len(relevant_positions)

            top_indices = ranked_indices[:max_rank]
            top_distances = [
                haversine_m(
                    query["lat"],
                    query["lon"],
                    patches[patch_index]["center_lat"],
                    patches[patch_index]["center_lon"],
                )
                for patch_index in top_indices
            ]
            if top_distances:
                top1_distances.append(top_distances[0])
            for rank in ranks:
                rank_distances = top_distances[: min(rank, len(top_distances))]
                topk_min_distances[rank].append(min(rank_distances) if rank_distances else float("inf"))

            if save_rankings:
                rankings.append(
                    {
                        "query_id": query["query_id"],
                        "area_id": query["area_id"],
                        "lat": query["lat"],
                        "lon": query["lon"],
                        "top_patches": [
                            {
                                "rank": rank + 1,
                                "patch_id": patches[patch_index]["patch_id"],
                                "area_id": patches[patch_index]["area_id"],
                                "center_lat": patches[patch_index]["center_lat"],
                                "center_lon": patches[patch_index]["center_lon"],
                                "distance_m": top_distances[rank],
                                "contains_gt": patch_index in positives,
                            }
                            for rank, patch_index in enumerate(top_indices)
                        ],
                    }
                )

    metrics = {f"recall@{rank}": correct[rank] / total * 100.0 for rank in ranks}
    metrics["mAP"] = ap_sum / total * 100.0
    metrics["top1_mean_distance_m"] = mean(top1_distances)
    metrics["top1_median_distance_m"] = median(top1_distances)
    metrics["top1_min_distance_m"] = float(min(top1_distances)) if top1_distances else float("nan")
    metrics["top1_max_distance_m"] = float(max(top1_distances)) if top1_distances else float("nan")

    for threshold in distance_thresholds:
        metrics[f"top1_distance_recall@{threshold:g}m"] = (
            sum(distance <= threshold for distance in top1_distances) / total * 100.0
        )
    for rank in ranks:
        distances = topk_min_distances[rank]
        metrics[f"top{rank}_min_mean_distance_m"] = mean(distances)
        metrics[f"top{rank}_min_median_distance_m"] = median(distances)
        for threshold in distance_thresholds:
            metrics[f"top{rank}_distance_recall@{threshold:g}m"] = (
                sum(distance <= threshold for distance in distances) / total * 100.0
            )

    metrics["num_queries"] = total
    metrics["num_gallery_patches"] = len(patches)
    metrics["num_fallback_positive_queries"] = fallback_positive_count
    metrics["num_no_positive_queries"] = no_positive_count
    return metrics, rankings


def format_metrics(metrics: dict, ranks: Sequence[int], distance_thresholds: Sequence[float]) -> str:
    parts = [f"R@{rank} {metrics[f'recall@{rank}']:.2f}" for rank in ranks]
    parts.append(f"mAP {metrics['mAP']:.2f}")
    parts.append(f"top1_med {metrics['top1_median_distance_m']:.2f}m")
    parts.append(f"top1_mean {metrics['top1_mean_distance_m']:.2f}m")
    for threshold in distance_thresholds:
        key = f"top1_distance_recall@{threshold:g}m"
        parts.append(f"R@{threshold:g}m {metrics[key]:.2f}")
    return " | ".join(parts)


def make_loader(dataset: Dataset, args, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def main():
    args = parse_args()
    started_at = time.perf_counter()
    device = torch.device(args.device)
    transform = build_transform(args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[UAV-VisLoc] checkpoint: {args.checkpoint}")
    print(f"[UAV-VisLoc] data_root: {args.data_root}")
    print(f"[UAV-VisLoc] patch_index: {args.patch_index}")
    print(f"[UAV-VisLoc] device: {device}")

    patches = read_patch_index(args.patch_index, max_patches=args.max_patches)
    queries = read_queries(args.data_root, max_query=args.max_query)
    area_to_indices = build_area_to_patch_indices(patches)
    print(f"[UAV-VisLoc] query images: {len(queries)}")
    print(f"[UAV-VisLoc] gallery patches: {len(patches)}")

    model = VPRModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.to(device)
    model.eval()

    gallery_start = time.perf_counter()
    gallery_loader = make_loader(SatellitePatchDataset(patches, transform), args, device)
    gallery_features = extract_features(model, gallery_loader, device, apply_adapter=False)
    print(f"[UAV-VisLoc] gallery feature extraction: {time.perf_counter() - gallery_start:.2f}s")

    query_start = time.perf_counter()
    query_loader = make_loader(QueryImageDataset(queries, transform), args, device)
    query_features = extract_features(model, query_loader, device, apply_adapter=True)
    print(f"[UAV-VisLoc] query feature extraction: {time.perf_counter() - query_start:.2f}s")

    align_start = time.perf_counter()
    query_vectors, gallery_vectors, procrustes_info = align_features(
        query_features=query_features,
        gallery_features=gallery_features,
        queries=queries,
        patches=patches,
        area_to_indices=area_to_indices,
        pca_dim=args.pca_dim,
        use_procrustes=not args.no_procrustes,
    )
    print(f"[UAV-VisLoc] feature alignment: {time.perf_counter() - align_start:.2f}s")
    print(f"[UAV-VisLoc] procrustes: {procrustes_info}")

    metrics, rankings = evaluate_geoloc_retrieval(
        query_vectors=query_vectors,
        gallery_vectors=gallery_vectors,
        queries=queries,
        patches=patches,
        area_to_indices=area_to_indices,
        ranks=args.ranks,
        distance_thresholds=args.distance_thresholds,
        step_size=args.step_size,
        save_rankings=args.save_rankings,
    )
    print(f"[UAV-VisLoc][all] {format_metrics(metrics, args.ranks, args.distance_thresholds)}")

    results = {
        "all": metrics,
        "_meta": {
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "patch_index": str(args.patch_index),
            "image_size": args.image_size,
            "pca_dim": args.pca_dim,
            "use_procrustes": not args.no_procrustes,
            "procrustes": procrustes_info,
            "ranks": args.ranks,
            "distance_thresholds_m": args.distance_thresholds,
            "num_queries": len(queries),
            "num_gallery_patches": len(patches),
            "positive_definition": "query center latitude/longitude is inside the satellite patch bbox",
            "distance_definition": "top1 distance is the haversine distance from query center to the top1 retrieved patch center; top1_med is its median over all queries",
            "total_elapsed_sec": time.perf_counter() - started_at,
        },
    }
    if args.save_rankings:
        results["rankings"] = rankings

    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[UAV-VisLoc] saved metrics to {args.output}")
    print(f"[UAV-VisLoc] total elapsed: {results['_meta']['total_elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
