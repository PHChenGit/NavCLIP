uv run main.py \
  --name openai_clip_image_encoder_only_sa_with_DJI_NTU_4 \
  --bs 512 \
  --epochs 150 \
  --lr 3e-4 \
  --ckpt_folder output/openai_clip_image_encoder_only_sa_with_DJI_NTU_4 \
  --ds_folder ~/Documents/hsun/datasets/DJI_NTU_4 \
  --dataset_file dataset.csv \
  --sat_img ~/Documents/hsun/datasets/DJI_NTU_4/NTU.jpg