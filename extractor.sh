#!/bin/sh
DB_DIR=//data/qiaoq/Project/salad_tz/datasets/1/8k
EXP_DIR=/data/qiaoq/Project/salad_tz/output_extractor
MODEL_FILE='./train_result/model/test6e-5-5epoch.pth'
CROP_SIZE=400,400 # LTA 750  UAVvisloc 980 LTA_label_finish 1200
STEP_SIZE=200,200
IM_SIZE=322,322
BS=128

CUDA_VISIBLE_DEVICES=1 python feature_extractor.py --input_dir ${DB_DIR} \
                                                     --save_dir ${EXP_DIR}/test_db_5epoch \
                                                     --ckpt_path ${MODEL_FILE} \
                                                     --image_size ${IM_SIZE} \
                                                     --crop_size ${CROP_SIZE} \
                                                     --step_size ${STEP_SIZE} \
                                                     --batch_size ${BS} \
                                                     --num_workers 8