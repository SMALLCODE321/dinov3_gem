DB_DIR='/data/qiaoq/Project/salad_tz/datasets/test_image'
INDEX_FILE='/data/qiaoq/Project/salad_tz/output_extractor/test_db/index'
INDEX_INFO_FILE='/data/qiaoq/Project/salad_tz/output_extractor/test_db/index_info.csv'
MODEL_FILE='/data/qiaoq/Project/salad_tz/checkpoints/tzb_model.ckpt'
QUERY_DIR='/data/qiaoq/Project/salad_tz/test_images'
GT_DIR='/autodl-fs/data/code_1/ground_truth/ground_truth_little.txt'
EXT='.jpg .png .tif' #'.jpg .png .TIF'
MATCHER='flann'
IM_SIZE=294
BATCH_SIZE=1
EXTEND_SIZE=294 
NUM_WORKERS=2
OUTPUT_RESULT='./RESULT_9_mine.txt'
 
CUDA_VISIBLE_DEVICES=0 python main_query.py --input ${QUERY_DIR} \
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