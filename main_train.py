import pytorch_lightning as pl
import torch
from vpr_model import VPRModel
from dataloaders.LTA_Dataloader import ImageFolderDataModule
from dataloaders.LTA_pretrain_Dataloader import SatelliteImageDataModule
from dataloaders.patch_Dataloader import PatchImageDataModule
from dataloaders.GTA_Dataloader import GTAUAVDataModule
from dataloaders.GTA_Dataloader1 import GTAUAVPlaceDataModule
import os
from vpr_eval_GTA import GTAEvaluator
import math

def model_adjust(model, checkpoint):
    
    model_dict = model.state_dict()
    # 针对 aggregator.score.3 参数进行手动更新
    for key in ['aggregator.score.3.weight', 'aggregator.score.3.bias']:
        if key in checkpoint:
            pretrained_param = checkpoint[key]
            current_param = model_dict[key]
            if pretrained_param.shape != current_param.shape:
                # 假设预训练参数尺寸为 [64, ...]，而当前模型为 [65, ...]
                # 则我们将预训练参数复制到前64个通道，额外的 ghost cluster 随机初始化
                new_param = current_param.clone()  # 先获取当前模型初始化的参数
                new_param[:pretrained_param.shape[0]] = pretrained_param
                model_dict[key] = new_param
                # 你也可以在这里打印信息确认参数更新
                print(f'Parameter {key} shape mismatch, pretrained param shape {pretrained_param.shape} -> new param shape {new_param.shape}')
            else:
                model_dict[key] = pretrained_param
    return model_dict

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    # datamodule = ImageFolderDataModule(
    #     batch_size=32, #32,
    #     shuffle_all=False, # shuffle all images or keep shuffling in-city only
    #     img_per_place=8,
    #     sat_aug_per_place=5,
    #     image_size=(322,322),
    #     num_workers=8,
    #     data_path='/data/qiaoq/Project/salad_tz/datasets/LTA/train/query',
    # )
    
    # 对卫星图切Patch，自监督学习
    # datamodule = SatelliteImageDataModule(
    #     batch_size=32, #32,
    #     image_size=(322,322),
    #     patch_size=(400,400),
    #     num_places=10000,
    #     sat_aug_per_place=5,
    #     num_workers=8,
    #     data_path='/data/qiaoq/Project/salad_tz/datasets/UAV_Large_Tilt_Angle_label_finish/train/gallery',
    # )
    
    # University-1652\SUES-200\DenseUAV数据集Dataloader
    datamodule = PatchImageDataModule(
                train_satellite_path='/data/qiaoq/Project/salad_tz/datasets/DenseUAV/train/satellite',
                train_drone_path='/data/qiaoq/Project/salad_tz/datasets/DenseUAV/train/drone',
                val_set_names=['University-1652'],
                num_queries=3,
                num_augs=5,
                batch_size=32,
                image_size=(322, 322),
                num_workers=8,
    )

    # GTAUAV数据集Dataloader
    # datamodule = GTAUAVDataModule(
    #     json_path='/data/qiaoq/Project/salad_tz/datasets/GTA-UAV-LR/cross-area-drone2sate-train.json',
    #     root_dir='/data/qiaoq/Project/salad_tz/datasets/GTA-UAV-LR',
    #     batch_size=50,
    #     num_workers=8,
    #     num_augs=3,
    #     image_size=(322, 322)
    # )
    # datamodule = GTAUAVPlaceDataModule(
    #     json_path='/data/qiaoq/Project/salad_tz/datasets/GTA-UAV-LR/cross-area-drone2sate-train.json',
    #     root_dir='/data/qiaoq/Project/salad_tz/datasets/GTA-UAV-LR',
    #     batch_size=20,
    #     num_workers=8,
    #     num_augs=3,
    #     image_size=(322, 322),
    #     max_drones_per_place=8
    # )

    model = VPRModel(
        #---- Encoder
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
        },
        agg_arch='SALAD',
        agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr = 6e-5,#普通的salad是6e-5,#对于tzb是1e-6
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': (2256 // 32) * 20,  # place_num / batch_size * epochs
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir=f'/data/qiaoq/Project/salad_tz/train_result', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=20,
        check_val_every_n_epoch=100, # run validation every epoch
        # callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=1,
    )

    pretrained_weight_path = './checkpoints/dino_salad.ckpt'
    pretrained_state_dict = torch.load(pretrained_weight_path)

    model.load_state_dict(pretrained_state_dict, strict=False)
    # we call the trainer, we give it the model and the datamodule

    # model = torch.load('/data/qiaoq/Project/salad_tz/train_result/model/University-6e-5-10epoch.pth')

    trainer.fit(model=model, datamodule=datamodule)
    torch.save(model, os.path.join('./train_result/model/', 'DenseUAV-6e-5-20epoch.pth'))