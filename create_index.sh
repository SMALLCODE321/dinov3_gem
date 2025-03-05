python create_index.py \
  --npy_directory ./npy_files \
  --index_file faiss.index \
  --index_type flat \
  --group_num 1 \
  --base_info_file ./data/base_map_info.txt \
  --crop_size 1080,2048 \
  --step_size 864,1638 \
  --index_info_file patches_index.csv