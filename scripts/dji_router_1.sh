uv run main.py \
  --name DJI_NTU_with_true_dji_images \
  --bs 512 \
  --epochs 200 \
  --lr 3e-4 \
  --ckpt_folder output/DJI_NTU_with_true_dji_images \
  --ds_folder ~/Documents/hsun/datasets/DJI_NTU/ \
  --dataset_file dataset.csv \
  --sat_img /home/rvl1421/Documents/hsun/datasets/DJI_NTU/NTU_sat.tif


uv run test.py \
  --pretrained_model_dir /home/rvl1421/Documents/hsun/NavCLIP/output/NTU_playground_layernorm \
  --output_dir output/DJI_NTU_router_1_gps_to_pixel \
  --ds_folder ~/Documents/hsun/datasets/DJI_NTU/60fps/DJI_NTU_router_1_gps_to_pixel \
  --dataset_file dataset.csv \
  --sat_img /home/rvl1421/Documents/hsun/datasets/DJI_NTU/NTU_sat.tif