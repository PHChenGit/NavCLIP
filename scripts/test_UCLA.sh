#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

uv run "${PROJECT_ROOT}/test.py" --name NTU_playground_image_encoder_224 \
    --pretrained_model_dir "${PROJECT_ROOT}/output/NTU_playground_image_encoder_224_StepLR" \
    --output_dir "${PROJECT_ROOT}/output/NTU_playground_image_encoder_224_StepLR" \
    --ds_folder "${PROJECT_ROOT}/datasets/NTU_playground_100k" \
    --dataset_file dataset.csv \
    --sat_img "${PROJECT_ROOT}/datasets/satellites/202410_NTU_playground.jpg"
