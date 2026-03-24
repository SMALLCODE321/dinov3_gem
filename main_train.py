import os
import pytorch_lightning as pl
import torch
from vpr_model import VPRModel
from dataloaders.SatellitePatchDataset import SatelliteSmallDataModule
from dataloaders.university1652pair import UniversityPairDataModule
from pytorch_lightning.strategies import DDPStrategy
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    datamodule = SatelliteSmallDataModule(
            data_path='/data/xulj/dinov3-salad/datasets/image',
            batch_size=16,         # 每个 batch 的 place 数量
            image_size=(336, 336), # 图像尺寸
            sat_aug_per_place=5,   # 每个 place 的增强版本数量
            num_workers=4)
    '''
    datamodule=UniversityPairDataModule(
        data_path='/data/xulj/dinov3-salad/datasets/University-1652/train',
        batch_size=8,
        drone_per_place=4,
        image_size=(336, 336),
        num_workers=4
    )
    '''
    model = VPRModel(
        #---- Encoder
        backbone_arch='dinov3_vitb16',
        backbone_config={
            'num_trainable_blocks': 8,
            'return_token': True,
            'norm_layer': True,
            'pretrained': True,
            'pretrained_path': '/data/xulj/dinov3-salad/checkpoints/dinov3_vitb16_pretrain.pth'
        },
        agg_arch='GEM',
        agg_config={
            #'num_channels': 768,
            #'num_clusters': 64,
            #'cluster_dim': 128,
            #'token_dim': 256,
            'p': 3.0,        # or learnable
            'eps': 1e-6
        },
        lr = 5e-6,#6e-5,#普通的salad是6e-5,#对于tzb是1e-6
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': (1652 // 64) * 50,  # place_num / batch_size * epochs
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.3,
        faiss_gpu=False
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy=DDPStrategy(process_group_backend="gloo", find_unused_parameters=True),
        default_root_dir=f'/data/xulj/salad_tz/train_result', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=50,
        check_val_every_n_epoch=100, # run validation every epoch
        # callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=1,
        accumulate_grad_batches=8,
    )
    trainer.fit(model=model, datamodule=datamodule)
    torch.save(model, os.path.join('./train_result/model/', 'DenseUAV-5e-6-50epoch.pth'))