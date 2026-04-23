from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from dataloaders.SatellitePatchDataset import SatelliteSmallDataModule
from vpr_model import VPRModel


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DATA_ROOT = Path("/data/qq/Project/qq/data/50w_image")
U1652_ROOT = Path("/data/qq/Project/qq/data/University-Release")
PRETRAINED_PATH = Path("/data/qq/Project/qq/dinov3_gem/weight/checkpoints/dinov3_vitb16_pretrain.pth")
CHECKPOINT_DIR = PROJECT_ROOT / "train_result" / "checkpoints"
MODEL_DIR = PROJECT_ROOT / "train_result" / "model"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3+GeM for VFM-Loc style cross-view geo-localization.")
    parser.add_argument("--run-name", default=None, help="Name used for checkpoint subfolder and model-state file.")
    parser.add_argument("--disable-orthogonal-adapter", action="store_true", help="Train strict DINOv3+GeM baseline without query adapter.")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--sat-aug-per-place", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--orthogonal-lambda", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true", help="Resume from this run's last.ckpt if it exists.")
    parser.add_argument("--no-eval", action="store_true", help="Disable University-1652 eval at epoch end.")
    return parser.parse_args()


def count_training_images(folder: Path) -> int:
    count = 0
    for root, _, files in os.walk(folder):
        count += sum(file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for file in files)
    return count


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    use_orthogonal_adapter = not args.disable_orthogonal_adapter
    run_name = args.run_name or ("orth_adapter" if use_orthogonal_adapter else "no_adapter")
    checkpoint_dir = CHECKPOINT_DIR / run_name

    batch_size = args.batch_size
    sat_aug_per_place = args.sat_aug_per_place
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    accumulate_grad_batches = args.accumulate_grad_batches
    if accumulate_grad_batches < 1:
        raise ValueError("--accumulate-grad-batches must be >= 1.")

    num_devices = max(1, torch.cuda.device_count())
    num_images = count_training_images(TRAIN_DATA_ROOT)
    local_num_images = (num_images + num_devices - 1) // num_devices
    steps_per_epoch = max(1, (local_num_images + batch_size - 1) // batch_size)
    optimizer_steps_per_epoch = max(1, (steps_per_epoch + accumulate_grad_batches - 1) // accumulate_grad_batches)
    total_iters = optimizer_steps_per_epoch * max_epochs
    print(
        "Training schedule: "
        f"{num_images} images | {num_devices} device(s) | "
        f"per-device batch {batch_size} | grad accumulation {accumulate_grad_batches} | "
        f"{optimizer_steps_per_epoch} optimizer steps/epoch | {total_iters} total iters"
    )
    print(f"Run name: {run_name} | use_orthogonal_adapter={use_orthogonal_adapter}")

    datamodule = SatelliteSmallDataModule(
        data_path=str(TRAIN_DATA_ROOT),
        batch_size=batch_size,
        image_size=(336, 336),
        sat_aug_per_place=sat_aug_per_place,
        num_workers=num_workers,
    )

    model = VPRModel(
        backbone_arch="dinov3_vitb16",
        backbone_config={
            "num_trainable_blocks": 2,
            "return_token": True,
            "norm_layer": True,
            "pretrained": True,
            "pretrained_path": str(PRETRAINED_PATH),
        },
        agg_arch="GEM",
        agg_config={
            "p": 3.0,
            "eps": 1e-6,
        },
        lr=args.lr,
        optimizer="adamw",
        weight_decay=args.weight_decay,
        momentum=0.9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1.0,
            "end_factor": 0.2,
            "total_iters": total_iters,
        },
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.3,
        faiss_gpu=False,
        use_orthogonal_adapter=use_orthogonal_adapter,
        orthogonal_lambda=args.orthogonal_lambda,
        eval_config={
            "enabled": not args.no_eval,
            "data_root": str(U1652_ROOT),
            "image_size": (336, 336),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "batch_size": 64,
            "num_workers": 4,
            "pca_dim": 256,
            "ranks": [1, 5, 10],
            "step_size": 256,
            "use_procrustes": True,
        },
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = checkpoint_dir / "last.ckpt"

    checkpoint_callback = ModelCheckpoint(
        monitor="u1652_vfmloc_R1",
        dirpath=str(checkpoint_dir),
        filename=f"{run_name}-epoch{{epoch:02d}}-r1{{u1652_vfmloc_R1:.2f}}",
        save_top_k=1,
        mode="max",
        save_last=True,
        every_n_epochs=1,
        auto_insert_metric_name=False,
    )

    strategy = (
        DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=True,
            timeout=timedelta(hours=4),
        )
        if num_devices > 1
        else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices,
        strategy=strategy,
        default_root_dir=str(PROJECT_ROOT / "train_result" / run_name),
        num_nodes=1,
        num_sanity_val_steps=0,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    resume_path = str(last_checkpoint_path) if args.resume and last_checkpoint_path.exists() else None
    if resume_path:
        print(f"Resuming from {resume_path}")
    else:
        print("Starting a fresh training run.")

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_path)

    torch.save(model.state_dict(), MODEL_DIR / f"{run_name}_last_state.pth")
    if checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
