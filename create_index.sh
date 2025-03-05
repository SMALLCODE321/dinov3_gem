#!/bin/sh
NPY_DIR=/data/qiaoq/Project/salad_tz/output_extractor/train_db
INDEX_FILE=${NPY_DIR}'/index'
INDEX_TYPE='flat'
GROU_NUM=1 # load all npy files to memory or load by groups to save memory
python create_index.py --npy_directory ${NPY_DIR}\
                       --index_file ${INDEX_FILE}\
                       --index_type ${INDEX_TYPE}\
                       --group_num ${GROU_NUM}