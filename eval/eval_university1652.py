from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vfm_loc_eval import University1652VfmLocEvaluator  # noqa: E402
from vpr_model import VPRModel  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/qq/Project/qq/data/University-Release")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "train_result/model/orth_adapter-epoch19-r193.77.ckpt"
DEFAULT_OUTPUT = PROJECT_ROOT / "train_result/eval_university1652/university1652_metrics.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DINOv3+GeM checkpoint on University-1652 using VFM-Loc.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--disable-pca", action="store_true")
    parser.add_argument("--disable-procrustes", action="store_true")
    parser.add_argument(
        "--disable-query-adapter",
        action="store_true",
        help="Do not apply OrthogonalAdapter on query/drone images during evaluation.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    print(f"[University-1652] checkpoint: {args.checkpoint}")
    print(f"[University-1652] data_root: {args.data_root}")
    print(f"[University-1652] query_apply_adapter: {not args.disable_query_adapter}")
    print(f"[University-1652] use_pca: {not args.disable_pca}")
    print(f"[University-1652] use_procrustes: {not args.disable_procrustes}")

    model = VPRModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.to(device)
    model.eval()

    evaluator = University1652VfmLocEvaluator(
        model=model,
        data_root=str(args.data_root),
        image_size=(args.image_size, args.image_size),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pca_dim=args.pca_dim,
        ranks=args.ranks,
        step_size=args.step_size,
        use_pca=not args.disable_pca,
        use_procrustes=not args.disable_procrustes,
        query_apply_adapter=not args.disable_query_adapter,
    )

    start = time.perf_counter()
    metrics = evaluator.evaluate()
    elapsed = time.perf_counter() - start
    metrics = {key: float(value) for key, value in metrics.items()}

    print(
        "[University-1652] "
        + " | ".join(
            [f"R@{rank} {metrics[f'recall@{rank}']:.2f}" for rank in args.ranks]
            + [f"mAP {metrics['mAP']:.2f}", f"elapsed {elapsed:.2f}s"]
        )
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "query_apply_adapter": not args.disable_query_adapter,
        "use_pca": not args.disable_pca,
        "use_procrustes": not args.disable_procrustes,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "pca_dim": args.pca_dim,
        "elapsed_sec": elapsed,
        "metrics": metrics,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[University-1652] saved metrics to {output}")


if __name__ == "__main__":
    main()
