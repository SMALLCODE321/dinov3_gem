#!/bin/sh
DB_DIR=/data/qiaoq/Project/salad_tz/datasets/UAV_Large_Tilt_Angle/train/gallery
EXP_DIR=/data/qiaoq/Project/salad_tz/output_extractor
MODEL_FILE='/data/qiaoq/Project/salad_tz/checkpoints/dino_salad.ckpt'
CROP_SIZE=1080,2048 # crop databse images into crops of crop_size 
STEP_SIZE=864,1638
IM_SIZE=700,1400
BS=256

CUDA_VISIBLE_DEVICES=0,1 python feature_extractor.py --input_dir ${DB_DIR} \
                                                     --save_dir ${EXP_DIR}/train_db \
                                                     --ckpt_path ${MODEL_FILE} \
                                                     --image_size ${IM_SIZE} \
                                                     --crop_size ${CROP_SIZE} \
                                                     --step_size ${STEP_SIZE} \
                                                     --batch_size ${BS} \
                                                     --num_workers 8