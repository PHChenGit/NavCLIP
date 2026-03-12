# NavCLIP

Training and evaluation code for NavCLIP.

## Requirements

- Python `>=3.12.9`
- `uv` for dependency and environment management

Install `uv` (Linux/macOS):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Install dependencies:

```bash
uv sync
```

## Dataset Layout

Set `--ds_folder` to your dataset root. The code expects:

```text
{ds_folder}/
  train/
    {dataset_file}.csv
    gallery.csv
  val/
    {dataset_file}.csv
    gallery.csv
  test/
    {dataset_file}.csv
```

`{dataset_file}.csv` must include these columns:

- `REF_IMG`
- `QUERY_IMG`
- `LAT`
- `LON`

## Training

```bash
uv run main.py \
  --name NavCLIP \
  --bs 512 \
  --epochs 150 \
  --lr 1e-4 \
  --ds_folder {your_dataset_root} \
  --dataset_file {your_dataset_file.csv} \
  --ckpt_folder {checkpoint_output_dir}
```

## Evaluation / Inference

```bash
uv run test.py \
  --bs 1 \
  --ds_folder {your_dataset_root} \
  --dataset_file {your_dataset_file.csv} \
  --pretrained_model_dir {checkpoint_path_or_dir} \
  --sat_img {satellite_image_path} \
  --output_dir {prediction_output_dir}
```

Outputs include:

- `test_result_2.json` (metrics)
- `location_matching.jpg` (visualization)

## Dataset

You can download the UAV Taipei dataset from:

http://vision.ee.ccu.edu.tw/dataset/UAV_Taipei.zip
