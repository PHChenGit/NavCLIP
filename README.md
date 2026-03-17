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

Use the provided download script to fetch one or more datasets.
The script shows an interactive menu so you can choose which datasets to download
— useful since the full collection totals ~202 GB.

Run from the project root:

```bash
chmod +x ./scripts/download_dataset.sh
./scripts/download_dataset.sh
```

```txt
Available datasets:

   1) Daan Park 100K                                       (10.9 GB)
   2) Daan Park                                            (3.9 GB)
   3) NTU Playground 1M                                    (34.1 GB)
  ...
  11) UCLA Cross Season 100K                               (64.6 GB)

Enter numbers to download (e.g. 1 3 5), or 'a' for all:
```

The script will then show the total size, available disk space, and ask for confirmation before downloading.

**Options:**

| Flag           | Description                                        |
| -------------- | -------------------------------------------------- |
| `[OUTPUT_DIR]` | Save to a custom directory (default: `./datasets`) |
| `-y` / `--yes` | Skip menu and confirmation — download everything   |

```bash
./scripts/download_dataset.sh ~/my/path   # custom output directory
./scripts/download_dataset.sh -y          # download all, no prompts
OUTPUT_DIR=~/my/path ./scripts/download_dataset.sh -y
```

Datasets are hosted at [here](https://vision.ee.ccu.edu.tw/dataset/UAV_Taipei.php).
