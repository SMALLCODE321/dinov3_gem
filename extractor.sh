#!/bin/sh
DB_DIR=/data/qiaoq/Project/salad_tz/datasets/UAV_Large_Tilt_Angle_label_finish/val/gallery
EXP_DIR=/data/qiaoq/Project/salad_tz/output_extractor
MODEL_FILE='./train_result/model/model6e-5-10epoch-nosat_aug.pth'
CROP_SIZE=1200,1200 # LTA 750  UAVvisloc 980 LTA_label_finish 1200
STEP_SIZE=500,500
IM_SIZE=322,322
BS=128

CUDA_VISIBLE_DEVICES=1 python feature_extractor.py --input_dir ${DB_DIR} \
                                                     --save_dir ${EXP_DIR}/train_db_10epoch_MoE \
                                                     --ckpt_path ${MODEL_FILE} \
                                                     --image_size ${IM_SIZE} \
                                                     --crop_size ${CROP_SIZE} \
                                                     --step_size ${STEP_SIZE} \
                                                     --batch_size ${BS} \
                                                     --num_workers 8