from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image


Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/data/qq/Project/qq/data/UAV_VisLoc_dataset")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "train_result/eval_uav_visloc/patch_indices"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a reusable UAV-VisLoc satellite patch index.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--crop-size", type=int, default=None, help="Backward-compatible square crop size.")
    parser.add_argument("--stride", type=int, default=None, help="Backward-compatible square stride.")
    parser.add_argument("--crop-width", type=int, default=1536)
    parser.add_argument("--crop-height", type=int, default=1024)
    parser.add_argument("--stride-x", type=int, default=None)
    parser.add_argument("--stride-y", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to train_result/eval_uav_visloc/patch_indices/.",
    )
    return parser.parse_args()


def find_satellite_range_csv(data_root: Path) -> Path:
    candidates = sorted(data_root.glob("*coordinates_range*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No satellite coordinate range csv found under {data_root}")
    return candidates[0]


def area_id_from_satellite_name(name: str) -> str:
    match = re.search(r"(\d+)", name)
    if not match:
        raise ValueError(f"Cannot infer area id from satellite filename: {name}")
    return match.group(1).zfill(2)


def axis_starts(length: int, crop_length: int, stride: int) -> list[int]:
    if crop_length <= 0:
        raise ValueError("crop length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= crop_length:
        return [0]

    starts = list(range(0, length - crop_length + 1, stride))
    last = length - crop_length
    if starts[-1] != last:
        starts.append(last)
    return starts


def pixel_to_lat_lon(
    x: float,
    y: float,
    width: int,
    height: int,
    lt_lat: float,
    lt_lon: float,
    rb_lat: float,
    rb_lon: float,
) -> tuple[float, float]:
    lon = lt_lon + (x / width) * (rb_lon - lt_lon)
    lat = lt_lat + (y / height) * (rb_lat - lt_lat)
    return lat, lon


def build_patch_rows(
    data_root: Path,
    crop_width: int,
    crop_height: int,
    stride_x: int,
    stride_y: int,
) -> tuple[list[dict], dict]:
    range_csv = find_satellite_range_csv(data_root)
    rows: list[dict] = []
    per_area_counts: dict[str, int] = {}

    with range_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for sat_row in reader:
            mapname = sat_row["mapname"]
            area_id = area_id_from_satellite_name(mapname)
            satellite_path = data_root / area_id / mapname
            if not satellite_path.exists():
                matches = sorted((data_root / area_id).glob("satellite*.tif"))
                if not matches:
                    raise FileNotFoundError(f"Missing satellite image for area {area_id}: {satellite_path}")
                satellite_path = matches[0]

            with Image.open(satellite_path) as image:
                width, height = image.size

            lt_lat = float(sat_row["LT_lat_map"])
            lt_lon = float(sat_row["LT_lon_map"])
            rb_lat = float(sat_row["RB_lat_map"])
            rb_lon = float(sat_row["RB_lon_map"])

            area_count = 0
            for y1 in axis_starts(height, crop_height, stride_y):
                y2 = min(y1 + crop_height, height)
                for x1 in axis_starts(width, crop_width, stride_x):
                    x2 = min(x1 + crop_width, width)
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    center_lat, center_lon = pixel_to_lat_lon(
                        center_x,
                        center_y,
                        width,
                        height,
                        lt_lat,
                        lt_lon,
                        rb_lat,
                        rb_lon,
                    )
                    lat_top, lon_left = pixel_to_lat_lon(
                        x1,
                        y1,
                        width,
                        height,
                        lt_lat,
                        lt_lon,
                        rb_lat,
                        rb_lon,
                    )
                    lat_bottom, lon_right = pixel_to_lat_lon(
                        x2,
                        y2,
                        width,
                        height,
                        lt_lat,
                        lt_lon,
                        rb_lat,
                        rb_lon,
                    )

                    rows.append(
                        {
                            "patch_id": f"{area_id}_{area_count:06d}",
                            "area_id": area_id,
                            "satellite_path": str(satellite_path),
                            "satellite_name": satellite_path.name,
                            "satellite_width": width,
                            "satellite_height": height,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "center_x": f"{center_x:.3f}",
                            "center_y": f"{center_y:.3f}",
                            "center_lat": f"{center_lat:.10f}",
                            "center_lon": f"{center_lon:.10f}",
                            "lat_top": f"{lat_top:.10f}",
                            "lon_left": f"{lon_left:.10f}",
                            "lat_bottom": f"{lat_bottom:.10f}",
                            "lon_right": f"{lon_right:.10f}",
                            "crop_width": crop_width,
                            "crop_height": crop_height,
                            "stride_x": stride_x,
                            "stride_y": stride_y,
                        }
                    )
                    area_count += 1

            per_area_counts[area_id] = area_count

    metadata = {
        "data_root": str(data_root),
        "satellite_range_csv": str(range_csv),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "num_patches": len(rows),
        "per_area_counts": per_area_counts,
    }
    return rows, metadata


def resolve_crop_and_stride(args) -> tuple[int, int, int, int]:
    crop_width = args.crop_size if args.crop_size is not None else args.crop_width
    crop_height = args.crop_size if args.crop_size is not None else args.crop_height
    stride_x = args.stride if args.stride is not None else args.stride_x
    stride_y = args.stride if args.stride is not None else args.stride_y
    if stride_x is None:
        stride_x = max(1, crop_width // 2)
    if stride_y is None:
        stride_y = max(1, crop_height // 2)
    return crop_width, crop_height, stride_x, stride_y


def main():
    args = parse_args()
    crop_width, crop_height, stride_x, stride_y = resolve_crop_and_stride(args)
    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_DIR / (
            f"uav_visloc_crop{crop_width}x{crop_height}_stride{stride_x}x{stride_y}.csv"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    rows, metadata = build_patch_rows(args.data_root, crop_width, crop_height, stride_x, stride_y)
    if not rows:
        raise RuntimeError("No patches were generated.")

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[UAV-VisLoc] saved patch index: {output}")
    print(f"[UAV-VisLoc] saved metadata: {meta_path}")
    print(f"[UAV-VisLoc] patches: {metadata['num_patches']}")
    for area_id, count in sorted(metadata["per_area_counts"].items()):
        print(f"[UAV-VisLoc] area {area_id}: {count} patches")


if __name__ == "__main__":
    main()
