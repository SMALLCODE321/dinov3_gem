CUDA_VISIBLE_DEVICES=1 python create_index.py \
  --npy_directory ./output_extractor/test_db_5epoch \
  --index_file ./output_extractor/test_db_5epoch/faiss.index \
  --index_type flat \
  --group_num 1 \
  --base_info_file ./datasets/1/8k/image_base_info.txt \
  --crop_size 400,400 \
  --step_size 200,200 \
  --index_info_file ./output_extractor/test_db_5epoch/patches_index.csv