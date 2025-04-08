CUDA_VISIBLE_DEVICES=1 python create_index.py \
  --npy_directory ./output_extractor/train_db_10epoch_tzb_train \
  --index_file ./output_extractor/train_db_10epoch_tzb_train/faiss.index \
  --index_type flat \
  --group_num 1 \
  --base_info_file ./datasets/UAV_Large_Tilt_Angle_label_finish/val/gallery/image_base_info.txt \
  --crop_size 1200,1200 \
  --step_size 500,500 \
  --index_info_file ./output_extractor/train_db_10epoch_tzb_train/patches_index.csv