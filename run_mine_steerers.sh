# DB_DIR='/autodl-fs/data/code_1/datasets/tzb_val1/images'
DB_DIR='/home/lwy/code_1/datasets/demo_db_images'
# INDEX_FILE='/autodl-fs/data/code_1/index_dir/index1/index'
INDEX_FILE='/home/lwy/code_1/output_extractor/index'
# INDEX_INFO_FILE='/autodl-fs/data/code_1/index_dir/index1/index_info.csv'
INDEX_INFO_FILE='/home/lwy/code_1/output_extractor/index_info.csv'
MODEL_FILE='/home/lwy/code_1/checkpoints/tzb_model.ckpt'
QUERY_DIR='/home/lwy/code_1/demo_query_images2'
GT_DIR='/autodl-fs/data/code_1/ground_truth/ground_truth_little.txt'
EXT='.jpg .png .tif' #'.jpg .png .TIF' 
MATCHER='flann'
IM_SIZE=700
BATCH_SIZE=1
EXTEND_SIZE=1400 
NUM_WORKERS=2
OUTPUT_RESULT='./RESULT_9_mine_steerers.txt'
 
CUDA_VISIBLE_DEVICES=0 python main_query_steerers.py --input ${QUERY_DIR} \
                                        --ext ${EXT}\
                                        --output_file ${OUTPUT_RESULT}\
                                        --db_dir ${DB_DIR}\
                                        --ckpt_path ${MODEL_FILE}\
                                        --index_path ${INDEX_FILE}\
                                        --index_info_csv ${INDEX_INFO_FILE}\
                                        --image_size ${IM_SIZE}\
                                        --batch_size ${BATCH_SIZE}\
                                        --extend_size ${EXTEND_SIZE}\
                                        --num_workers ${NUM_WORKERS}\
                                        --matcher ${MATCHER}\
                                        --gt_file ${GT_DIR}\
                                        --topk_index 5\