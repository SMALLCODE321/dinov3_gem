#!/bin/sh
DB_DIR=/data/qiaoq/Project/salad_tz/datasets/UAV_Large_Tilt_Angle/val/gallery
EXP_DIR=/data/qiaoq/Project/salad_tz/output_extractor
MODEL_FILE='./train_result/model/tzb-model1e-6-50epoch-low_resolution.pth'
CROP_SIZE=750,750 # LTA 750  UAVvisloc 980
STEP_SIZE=500,500
IM_SIZE=322,322
BS=128

CUDA_VISIBLE_DEVICES=1 python feature_extractor.py --input_dir ${DB_DIR} \
                                                     --save_dir ${EXP_DIR}/train_db_LTA_50epoch_tzbbase_low_resolution \
                                                     --ckpt_path ${MODEL_FILE} \
                                                     --image_size ${IM_SIZE} \
                                                     --crop_size ${CROP_SIZE} \
                                                     --step_size ${STEP_SIZE} \
                                                     --batch_size ${BS} \
                                                     --num_workers 8