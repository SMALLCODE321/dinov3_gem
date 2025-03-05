#!/bin/sh
DB_INFO_FILE='/data/qiaoq/Project/salad_tz/datasets/UAV_Large_Tilt_Angle/train/gallery/image_info.txt'
CROP_SIZE=1080,2048 # crop databse images into crops of crop_size 
STEP_SIZE=864,1638 
INDEX_INFO_FILE='/data/qiaoq/Project/salad_tz/output_extractor/train_db/index_info.csv'

python create_index_info.py --base_info_file ${DB_INFO_FILE}\
                            --crop_size ${CROP_SIZE}\
                            --step_size ${STEP_SIZE}\
                            --index_info_file ${INDEX_INFO_FILE}
