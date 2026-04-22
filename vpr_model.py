from __future__ import annotations

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import lr_scheduler

import utils
from models import helper
from vfm_loc_eval import University1652VfmLocEvaluator


class OrthogonalAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

    def orthogonality_penalty(self) -> torch.Tensor:
        identity = torch.eye(self.weight.size(0), device=self.weight.device, dtype=self.weight.dtype)
        gram = self.weight.transpose(0, 1) @ self.weight
        return (gram - identity).pow(2).mean()


class VPRModel(pl.LightningModule):
    def __init__(
        self,
        backbone_arch: str = "resnet50",
        backbone_config: Optional[Dict] = None,
        agg_arch: str = "ConvAP",
        agg_config: Optional[Dict] = None,
        lr: float = 0.03,
        optimizer: str = "sgd",
        weight_decay: float = 1e-3,
        momentum: float = 0.9,
        lr_sched: str = "linear",
        lr_sched_args: Optional[Dict] = None,
        loss_name: str = "MultiSimilarityLoss",
        miner_name: str = "MultiSimilarityMiner",
        miner_margin: float = 0.1,
        faiss_gpu: bool = False,
        use_orthogonal_adapter: bool = True,
        orthogonal_lambda: float = 1e-3,
        eval_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.backbone_config = backbone_config or {}
        self.agg_config = agg_config or {}
        self.lr_sched_args = lr_sched_args or {
            "start_factor": 1.0,
            "end_factor": 0.2,
            "total_iters": 4000,
        }
        self.eval_config = eval_config or {}

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        self.faiss_gpu = faiss_gpu
        self.use_orthogonal_adapter = use_orthogonal_adapter
        self.orthogonal_lambda = orthogonal_lambda

        self.save_hyperparameters()

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

        self.backbone = helper.get_backbone(backbone_arch, self.backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, self.agg_config)

        if not hasattr(self.backbone, "num_channels"):
            raise AttributeError("Backbone must expose num_channels for the orthogonal baseline.")
        self.descriptor_dim = int(self.backbone.num_channels) * 2
        self.query_adapter = OrthogonalAdapter(self.descriptor_dim) if use_orthogonal_adapter else None

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, cls_token = self.backbone(x)
        pooled = self.aggregator(feature_map)
        if pooled.dim() == 4:
            pooled = pooled.flatten(1)
        return torch.cat([pooled, cls_token], dim=1)

    def apply_query_adapter(self, x: torch.Tensor) -> torch.Tensor:
        if self.query_adapter is None:
            return x
        return self.query_adapter(x)

    def forward(self, x: torch.Tensor, apply_adapter: bool = False) -> torch.Tensor:
        descriptors = self.encode_backbone(x)
        if apply_adapter:
            descriptors = self.apply_query_adapter(descriptors)
        return descriptors

    def configure_optimizers(self):
        optimizer_name = self.optimizer.lower()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        sched_name = self.lr_sched.lower()
        if sched_name == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_sched_args["milestones"],
                gamma=self.lr_sched_args["gamma"],
            )
        elif sched_name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args["T_max"])
        elif sched_name == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args["start_factor"],
                end_factor=self.lr_sched_args["end_factor"],
                total_iters=self.lr_sched_args["total_iters"],
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.lr_sched}")

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    def compute_metric_loss(self, descriptors: torch.Tensor, labels: torch.Tensor):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        return loss, batch_acc

    def training_step(self, batch, batch_idx):
        places, labels = batch
        batch_size, num_views, ch, h, w = places.shape

        images = places.view(batch_size * num_views, ch, h, w)
        flat_labels = labels.view(-1)
        raw_descriptors = self.forward(images, apply_adapter=False).view(batch_size, num_views, -1)

        if self.query_adapter is not None and num_views > 1:
            reference_descriptors = raw_descriptors[:, :1, :]
            query_descriptors = self.apply_query_adapter(raw_descriptors[:, 1:, :].reshape(-1, self.descriptor_dim))
            query_descriptors = query_descriptors.view(batch_size, num_views - 1, self.descriptor_dim)
            descriptors = torch.cat([reference_descriptors, query_descriptors], dim=1)
        else:
            descriptors = raw_descriptors

        descriptors = descriptors.view(batch_size * num_views, -1)
        metric_loss, batch_acc = self.compute_metric_loss(descriptors, flat_labels)
        orth_loss = (
            self.query_adapter.orthogonality_penalty()
            if self.query_adapter is not None
            else torch.zeros((), device=descriptors.device)
        )
        loss = metric_loss + self.orthogonal_lambda * orth_loss

        self.batch_acc.append(batch_acc)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("loss_metric", metric_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("loss_orth", orth_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("b_acc", sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=batch_size)
        return loss

    def on_train_epoch_end(self):
        self.batch_acc = []

        if not self.eval_config.get("enabled", True):
            return

        is_global_zero = getattr(self.trainer, "is_global_zero", True)
        metric_names = ("recall@1", "recall@5", "recall@10", "mAP")
        metric_tensor = torch.zeros(len(metric_names), device=self.device, dtype=torch.float32)

        if is_global_zero:
            was_training = self.training
            self.eval()

            evaluator = University1652VfmLocEvaluator(
                model=self,
                data_root=self.eval_config["data_root"],
                image_size=tuple(self.eval_config.get("image_size", (336, 336))),
                mean=self.eval_config.get("mean", [0.485, 0.456, 0.406]),
                std=self.eval_config.get("std", [0.229, 0.224, 0.225]),
                batch_size=int(self.eval_config.get("batch_size", 64)),
                num_workers=int(self.eval_config.get("num_workers", 4)),
                pca_dim=int(self.eval_config.get("pca_dim", 256)),
                ranks=self.eval_config.get("ranks", [1, 5, 10]),
                step_size=int(self.eval_config.get("step_size", 256)),
                use_procrustes=bool(self.eval_config.get("use_procrustes", True)),
            )
            metrics = evaluator.evaluate()
            metric_tensor = torch.tensor(
                [metrics[name] for name in metric_names],
                device=self.device,
                dtype=torch.float32,
            )

            print(
                f"[Epoch {self.current_epoch}] "
                f"VFM-Loc U1652 -> "
                f"R@1 {metrics['recall@1']:.2f} | "
                f"R@5 {metrics['recall@5']:.2f} | "
                f"R@10 {metrics['recall@10']:.2f} | "
                f"mAP {metrics['mAP']:.2f}"
            )

            if was_training:
                self.train()

        if self.trainer.world_size > 1:
            torch.distributed.broadcast(metric_tensor, src=0)

        metrics = {name: metric_tensor[idx].item() for idx, name in enumerate(metric_names)}
        self.log("u1652_vfmloc_R1", metrics["recall@1"], prog_bar=True, logger=True, sync_dist=True)
        self.log("u1652_vfmloc_R5", metrics["recall@5"], prog_bar=False, logger=True, sync_dist=True)
        self.log("u1652_vfmloc_R10", metrics["recall@10"], prog_bar=False, logger=True, sync_dist=True)
        self.log("u1652_vfmloc_mAP", metrics["mAP"], prog_bar=False, logger=True, sync_dist=True)
