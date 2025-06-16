import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from dataloader import GeoCLIPDataModule, DataLoaderTypesEnum
from models.geoclip.GeoCLIPLightning import GeoCLIPLightning
from models.geoclip.misc import log_pred_result, calculate_angle_error, calculate_location_error_metrics, gps_to_pixel_and_draw, create_gallery
from models.geoclip.satellite_img_processor import SatelliteImageProcessor


def visualize(pred_locations: List[Tuple[int, int]], true_locations:List[Tuple[int, int]], output_folder: Path, sat_img_path: str):
    assert len(pred_locations) == len(true_locations)
    COLOR_GREEN = (0, 255, 0) # Ground-Truth
    COLOR_RED = (0, 0, 255) # Predict 
    COLOR_BLUE = (255, 0, 0)
    image_path = Path(sat_img_path).expanduser()

    if not image_path.exists():
        raise ValueError(f"satellite image not found: {image_path}")

    sat_img = cv2.imread(str(image_path))
    img = sat_img.copy()
    rad = 10
    thickness = 10

    for i, pred_location in tqdm(enumerate(pred_locations), total=len(pred_locations), desc="Visualizing"):
        center_gt = (int(true_locations[i][0][0]), int(true_locations[i][0][1]))
        cv2.circle(img, center_gt, rad, COLOR_GREEN, thickness)

        center_pred = (int(pred_locations[i][0][0]), int(pred_locations[i][0][1]))
        cv2.circle(img, center_pred, rad, COLOR_RED, thickness)

        cv2.line(img, center_gt, center_pred, COLOR_BLUE, thickness=thickness)

    cv2.imwrite(str(output_folder.joinpath("location_matching.jpg")), img)

def main(args):
    seed_everything(42, workers=True)
    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    CSV_FILE = args.dataset_file
    DATASET_ROOT_PATH = args.ds_folder

    DATASET_ROOT = Path(DATASET_ROOT_PATH)
    PRED_CSV = DATASET_ROOT.joinpath('test/router_1', CSV_FILE)
    COORDINATE_GALLERY = DATASET_ROOT.joinpath('test/router_1', "gallery.csv")
    print(f">\tCoordinate gallery path: {COORDINATE_GALLERY}")
    datamodule = GeoCLIPDataModule(
        dataset_folder=str(DATASET_ROOT),
        predict_csv=str(PRED_CSV),
        dataset_type=DataLoaderTypesEnum.TestVisLoc,
        batch_size=args.bs,
        num_workers=args.num_workers,
        image_size=224,
        sat_img_path=args.sat_img
    )
    datamodule.setup('predict')

    test_dataloader = datamodule.predict_dataloader()
    if not COORDINATE_GALLERY.exists():
        create_gallery(COORDINATE_GALLERY.parent, test_dataloader)

    model = GeoCLIPLightning(gallery_path=str(COORDINATE_GALLERY), sat_img=args.sat_img, clip_model_name="facebook/metaclip-l14-400m")
    model.load_weights(args.pretrained_model_dir)
    trainer = Trainer(
        default_root_dir='output',
        accelerator=ACCELERATOR,
    )
    pred_result: List[Dict] = trainer.predict(model, datamodule=datamodule)

    # print(pred_result)
    pred_coarse_coordinate_list = []
    true_coordinate_list = []
    pred_yaw_list = []
    true_yaw_list = []
    for data in pred_result:
        if np.isnan(data["pred_yaw_angle"]):
            continue

        pred_coarse_coordinate_list.append(data["pred_coarse_coordinate"])
        true_coordinate_list.append(data["true_coordinate"])
        pred_yaw_list.append(data["pred_yaw_angle"])
        true_yaw_list.append(data["true_yaw"][0])

    # pred_dist_mae_degree, pred_dist_rmse_degree = model._common_val_test_loss(torch.Tensor(np.array(pred_coarse_coordinate_list)), torch.Tensor(np.array(true_coordinate_list)))
    # pred_dist_mae_degree = round(pred_dist_mae_degree.detach().item(), 2)
    # pred_dist_rmse_degree = round(pred_dist_rmse_degree.detach().item(), 2)

    pred_dist_dict = calculate_location_error_metrics(pred_coarse_coordinate_list, true_coordinate_list)

    pred_yaw_mae, pred_yaw_rmse = calculate_angle_error(pred_yaw_list, true_yaw_list)
    pred_yaw_mae = round(pred_yaw_mae, 2)
    pred_yaw_rmse = round(pred_yaw_rmse, 2)

    data = {
        "Pred GPS Dist MAE(Meters)": pred_dist_dict['mae_meters'],
        "Pred GPS Dist RMSE(Meters)": pred_dist_dict['rmse_meters'],
        "Pred Yaw MAE(degree)": pred_yaw_mae,
        "Pred Yaw RMSE(degree)": pred_yaw_rmse
    }
    print(data)
    log_pred_result(data, Path(args.output_dir), "test_result.json")
    # gps_to_pixel_and_draw(args.sat_img, pred_coarse_coordinate_list, str(Path(args.output_dir).joinpath("location_matching.jpg")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference GeoCLIP")
    parser.add_argument("--name", type=str, default="GeoCLIP", help="experiment name")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--queue_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pretrained_model_dir", type=str, default="/home/rvl1421/Documents/hsun/NavCLIP/output/checkpoints/", help="pretrained model folder")
    parser.add_argument("--output_dir", type=str, help="Prediction result folder")
    parser.add_argument("--ds_folder", type=str, default=r"~/Documents/hsun/datasets/NTU_playground_Cross_Season_100k", help="dataset folder path")
    parser.add_argument("--dataset_file", type=str, default=r"taipei.csv", help="dataset csv file")
    parser.add_argument("--sat_img", type=str, default=r"~/Documents/hsun/datasets/satellites/202410_NTU_playground.jpg", help="dataset folder path")
    args = parser.parse_args()
    main(args)

