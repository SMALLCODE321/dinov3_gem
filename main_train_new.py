import os
import types
import torch
import pytorch_lightning as pl

from vpr_model import VPRModel
from dataloaders.patch_Dataloader import PatchImageDataModule

# ----------------------------------------
#  Callback：前 N 个 epoch 冻结 backbone
# ----------------------------------------
class WarmupFreeze(pl.Callback):
    def __init__(self, freeze_epochs: int = 2):
        super().__init__()
        self.freeze_epochs = freeze_epochs

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.freeze_epochs:
            pl_module.backbone.eval()
            for p in pl_module.backbone.parameters():
                p.requires_grad = False
        else:
            pl_module.backbone.train()
            for p in pl_module.backbone.parameters():
                p.requires_grad = True

# --------------------------------------------------
#  Monkey‐patch：给 backbone/aggregator 设置不同的 lr
# --------------------------------------------------
def configure_optimizers_override(self):
    # 1) 分组参数
    backbone_params = list(self.backbone.parameters())
    agg_params = list(self.aggregator.parameters())
    # 如果有单独的 score head
    if hasattr(self, "score"):
        agg_params += list(self.score.parameters())

    # 2) 构造优化器
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": self.hparams.lr},
            {"params": agg_params,      "lr": self.hparams.agg_lr},
        ],
        weight_decay=self.hparams.weight_decay,
    )

    # 3) 线性 lr decay
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=self.hparams.lr_sched_args["start_factor"],
        end_factor=self.hparams.lr_sched_args["end_factor"],
        total_iters=self.hparams.lr_sched_args["total_iters"],
    )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }

# ================================
#    主函数
# ================================
if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # --------- DataModule ---------
    datamodule = PatchImageDataModule(
        train_satellite_path="/data/qiaoq/Project/salad_tz/datasets/University-1652/train/satellite",
        train_drone_path="/data/qiaoq/Project/salad_tz/datasets/University-1652/train/drone",
        val_set_names=["University-1652"],
        batch_size=32,
        image_size=(322, 322),
        num_workers=8,
    )

    # --------- Model ---------
    model = VPRModel(
        # Encoder (DINO‐ViT)
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        # Aggregator (SALAD)
        agg_arch="SALAD",
        agg_config={
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256,
        },

        # 优化器 & LR scheduler
        lr=6e-5,             # backbone lr
        optimizer="adamw",
        weight_decay=9.5e-9,
        lr_sched="linear",
        lr_sched_args={
            "start_factor": 1.0,
            "end_factor": 0.2,
            # place_num / batch_size * max_epochs
            "total_iters": (701 // 32) * 10,
        },

        # Loss
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=False,
    )

    # 覆盖 configure_optimizers
    model.configure_optimizers = types.MethodType(configure_optimizers_override, model)

    # --------- Trainer ---------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        default_root_dir="/data/qiaoq/Project/salad_tz/train_result",
        precision="16-mixed",
        max_epochs=10,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=1,
        callbacks=[
            WarmupFreeze(freeze_epochs=2),
        ],
    )

    # Load pretrained DINO backbone（strict=False 会自动忽略 SALAD 部分）
    dino_ckpt = torch.hub.load(
        "facebookresearch/dino:main",
        "dino_vitbase16_pretrain"
    )
    # 过滤并注入 ViT‐DINO 权重
    model_state = model.state_dict()
    for k, v in dino_ckpt.items():
        name = k.replace("module.", "")
        if name in model_state and v.shape == model_state[name].shape:
            model_state[name] = v
    model.load_state_dict(model_state)

    # --------- 训练 & 最后保存 ---------
    trainer.fit(model, datamodule=datamodule)
    torch.save(
        model,
        os.path.join("./train_result/model/", "University-dino_only-10epoch-2.pth"),
    )