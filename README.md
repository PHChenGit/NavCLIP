# Install

## Prepare environment

### UV
If you don't install uv on your PC, please install uv first.
UV is a new environment and package management system, it just like anaconda, but it's more faster. For more uv detail you can read it on uv official website. [UV introduction](https://docs.astral.sh/uv/)

Because I'm using Ubuntu 24.04 for my PC, I install the uv in this way.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

And then check it.
```
uv
```

### Python, Environment, and Packages

Install python.
```
uv python install 3.12
```

Prepare virtual environment.
```
uv venv
source .venv/bin/activate
```

Install dependency packages.
```
uv pip install .
```

# Usage

## Train

```
uv main.py --bs 512 --epochs 150 \
  --ds_folder {your_dataset_root} \
  --dataset_file {your_image_and_location.csv} \
  --output {where_you_want_to_save_checkpoints}
```

## Test
```
uv test.py --bs 1 \
  --ds_folder {your_dataset_root} \
  --dataset_file {your_image_and_location.csv} \
  --pretrained_model_dir {where_to_load_your_pretrained_checkpoints} \
  --output_dir {where_you_want_to_save_your_prediction_result}