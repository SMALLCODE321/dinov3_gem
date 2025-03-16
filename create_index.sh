CUDA_VISIBLE_DEVICES=1 python create_index.py \
  --npy_directory ./output_extractor/train_db_LTA_50epoch_tzbbase \
  --index_file ./output_extractor/train_db_LTA_50epoch_tzbbase/faiss.index \
  --index_type flat \
  --group_num 1 \
  --base_info_file ./datasets/UAV_Large_Tilt_Angle/val/gallery/image_info.txt \
  --crop_size 750,750 \
  --step_size 500,500 \
  --index_info_file ./output_extractor/train_db_LTA_50epoch_tzbbase/patches_index.csv