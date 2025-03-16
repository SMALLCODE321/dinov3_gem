import pytorch_lightning as pl
import torch
from vpr_model import VPRModel
from dataloaders.LTA_Dataloader import ImageFolderDataModule
import os

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    datamodule = ImageFolderDataModule(
        batch_size=32, #32,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        img_per_place=8,
        sat_aug_per_place=5,
        image_size=(322,322),
        num_workers=8,
        show_data_stats=True,
        data_path='/data/qiaoq/Project/salad_tz/datasets/UAV_Large_Tilt_Angle/train/query',
        val_set_names=['UAV_Large_Tilt_Angle/val/query'], # pitts30k_val, pitts30k_test, msls_val
    )
    
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
        lr = 1e-6,#6e-5,#
        optimizer='adamw',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 1200,# 980/60 * max_epochs = 17*max_epochs
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='LTA/R1',
        filename=f'/data/qiaoq/Project/salad_tz/train_result/{model.encoder_arch}' + '_({epoch:02d})_R1[{LTA/R1:.4f}]_R10[{LTA/R10:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=5,
        save_last=True,
        mode='max'
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
        max_epochs=50,
        check_val_every_n_epoch=60, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=10, # we reload the dataset to shuffle the order
        log_every_n_steps=5,
    )

    pretrained_weight_path = './checkpoints/tzb_model.ckpt'
    pretrained_state_dict = torch.load(pretrained_weight_path)
    model.load_state_dict(pretrained_state_dict)
    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
    torch.save(model, os.path.join('./train_result/model/', 'tzb-model1e-6-50epoch.pth'))