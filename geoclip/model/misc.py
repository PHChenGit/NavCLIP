import os
from pathlib import Path
import json

import torch
import numpy as np
import pandas as pd

file_dir = os.path.dirname(os.path.realpath(__file__))

def load_gallery_data(csv_file: str) -> torch.Tensor:
    data = pd.read_csv(csv_file)
    lat_lon = data[['LAT', 'LON']]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)

    return gps_tensor


def log_pred_result(result: dict, output_path: str, filename: str) -> None:
    full_path = Path(output_path) / filename
    with open(str(full_path), "w") as json_file:
        json.dump(result, json_file, indent=4)
    print(f">\t Prediction result has been written to {full_path}")

