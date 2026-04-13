import os
import pytorch_lightning as pl
import torch
from vpr_model import VPRModel
from dataloaders.SatellitePatchDataset import SatelliteSmallDataModule
from dataloaders.university1652pair import UniversityPairDataModule
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    # 配置路径
    checkpoint_dir = '/data/xulj/dinov3_salad/train_result/checkpoints'
    last_checkpoint_path = os.path.join(checkpoint_dir, 'last.ckpt')

    datamodule = SatelliteSmallDataModule(
            data_path='/data/xulj/dinov3-salad/datasets/image',
            batch_size=16,         # 每个 batch 的 place 数量
            image_size=(336, 336), # 图像尺寸
            sat_aug_per_place=5,   # 每个 place 的增强版本数量
            num_workers=4)

    model = VPRModel(
        #---- Encoder
        backbone_arch='dinov3_vitb16',
        backbone_config={
            'num_trainable_blocks': 6,
            'return_token': True,
            'norm_layer': True,
            'pretrained': True,
            'pretrained_path': '/data/xulj/dinov3-salad/checkpoints/dinov3_vitb16_pretrain.pth'
        },
        agg_arch='GEM',
        agg_config={
            'p': 3.0,
            'eps': 1e-6
        },
        lr = 5e-6,
        optimizer='adamw',
        weight_decay=9.5e-9, 
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': (1652 // 64) * 40,  
        },
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.3,
        faiss_gpu=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='loss',           
        dirpath=checkpoint_dir,
        filename='dinov3-vpr-{epoch:02d}-{loss:.3f}',
        save_top_k=3,             
        mode='min',
        save_last=True,           # 核心：这会始终生成/更新一个 last.ckpt
        every_n_epochs=1          
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy=DDPStrategy(process_group_backend="gloo", find_unused_parameters=True),
        default_root_dir='/data/xulj/salad_tz/train_result', 
        num_nodes=1,
        num_sanity_val_steps=0, 
        precision='16-mixed', 
        max_epochs=40,
        callbacks=[checkpoint_callback], 
        check_val_every_n_epoch=1,       
        reload_dataloaders_every_n_epochs=1, 
        log_every_n_steps=1,
        accumulate_grad_batches=4,
    )

    # --- 修改部分：自动检测是否有 checkpoint ---
    resume_path = None
    if os.path.exists(last_checkpoint_path):
        print(f"检测到未完成的训练，正在从 {last_checkpoint_path} 恢复...")
        resume_path = last_checkpoint_path
    else:
        print("未检测到上次训练的 checkpoint，开始全新训练。")

    # 传入 ckpt_path 即可实现断点续训
    # 它会自动恢复当前的 epoch 数、优化器状态、学习率调度器状态以及模型权重
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_path)
    
    # 训练结束后保存最终模型
    save_final_dir = './train_result/model/'
    os.makedirs(save_final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_final_dir, 'cls_patch_40epoch_state.pth'))