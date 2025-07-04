import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import numpy.typing as npt
import pandas as pd
import cv2
from tqdm import tqdm
from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from dataloader import GeoCLIPDataModule, DataLoaderTypesEnum
from models.geoclip.GeoCLIPLightning import GeoCLIPLightning
from models.geoclip.misc import log_pred_result, calculate_angle_error, create_gallery, calculate_location_error_metrics

from models.geoclip.satellite_processor import SatelliteImageProcessor
from pyproj import  CRS

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
        true_loc = (int(true_locations[i][0]), int(true_locations[i][1]))
        cv2.circle(img, true_loc, rad, COLOR_GREEN, thickness)

        pred_loc = (int(pred_location[0]), int(pred_location[1]))
        cv2.circle(img, pred_loc, rad, COLOR_RED, thickness)

        # cv2.line(img, center_gt, center_pred, COLOR_BLUE, thickness=thickness)

    cv2.imwrite(str(output_folder.joinpath("location_matching.jpg")), img)

def visualize_error_histogram(
        errors: npt.NDArray,
        output_folder: Path,
        title: str='Distribution of Location Prediction Errors',
        x_axis_label: str='Error Distance (meters)',
        filename: str='Prediction_distance_error_histogram.png'
        ):
    """
    將位置預測誤差可視化為直方圖，並自動調整X軸刻度以獲得最佳可讀性。

    Args:
        errors (List[float]): 每個預測點的誤差距離列表（單位：公尺）。
        output_folder (Path): 儲存圖表的資料夾路徑。
    """
    if not np.all(errors):
        raise ValueError("No error data to visualize.")


    plt.style.use('seaborn-v0_8-whitegrid') # 使用更美觀的樣式
    fig, ax = plt.subplots(figsize=(12, 7))

    max_error_display = np.ceil(np.percentile(errors, 99.5))
    
    # 如果數據全為0或非常小，設定一個最小範圍
    if max_error_display < 1.0:
        max_error_display = 1.0

    # 2. 動態決定直方圖的 "bins" (分組)
    #    我們希望分的組數多一點，讓圖形更精細
    #    MaxNLocator 會在指定範圍內，找出最多 N 個漂亮的整數點作為邊界
    if max_error_display <= 15:
        bin_count = int(max_error_display) * 2 # 誤差小時，bin更密集
    elif max_error_display <= 50:
        bin_count = 40
    else:
        bin_count = 50
        
    bin_locator = mticker.MaxNLocator(nbins=bin_count, integer=True)
    bins = bin_locator.tick_values(vmin=0, vmax=max_error_display)

    ax.hist(errors, bins=bins, color='deepskyblue', edgecolor='black', alpha=0.75)

    tick_locator = mticker.MaxNLocator(nbins=10, prune='both', integer=True)
    ax.xaxis.set_major_locator(tick_locator)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel('Frequency (Count)', fontsize=12)

    ax.set_xlim(0, max_error_display)
    ax.set_ylim(bottom=0)

    if len(bins) <= 30:
        for rect in ax.patches:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    output_path = output_folder.joinpath(filename)
    fig.savefig(output_path, dpi=300)
    print(f"Error histogram has been saved to: {output_path}")
    plt.close(fig)


def main(args):
    seed_everything(42, workers=True)
    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    CSV_FILE = args.dataset_file
    DATASET_ROOT_PATH = args.ds_folder

    DATASET_ROOT = Path(DATASET_ROOT_PATH)
    PRED_CSV = DATASET_ROOT.joinpath('test', CSV_FILE)
    COORDINATE_GALLERY = DATASET_ROOT.joinpath('test/random/gallery_extend.csv')
    print(f">\tCoordinate gallery path: {COORDINATE_GALLERY}")

    OUTPUT_DIR = Path(args.output_dir)
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    datamodule = GeoCLIPDataModule(
        dataset_folder=str(DATASET_ROOT),
        predict_csv=str(PRED_CSV),
        dataset_type=DataLoaderTypesEnum.TestPose,
        batch_size=args.bs,
        num_workers=args.num_workers,
        image_size=224
    )
    datamodule.setup('predict')

    pred_dataloader = datamodule.predict_dataloader()
    if not COORDINATE_GALLERY.exists():
        create_gallery(COORDINATE_GALLERY.parent, pred_dataloader)

    model = GeoCLIPLightning(
        gallery_path=str(COORDINATE_GALLERY),
        sat_img=args.sat_img_png,
        clip_model_name="google/siglip2-base-patch16-224",
        homography_method="mapglue"
        )
    model.load_weights(args.pretrained_model_dir)
    trainer = Trainer(
        default_root_dir='output',
        accelerator=ACCELERATOR,
        devices='auto',
        precision="bf16-mixed",
    )
    pred_result: List[Dict] = trainer.predict(model, datamodule=datamodule)

    sat_processor = SatelliteImageProcessor(args.sat_img_tif)

    pred_pixel_coordinate_list = []
    true_pixel_coordinate_list = []
    pred_gps_coordinate_list = []
    true_gps_coordinate_list = []
    pred_yaw_list = []
    true_yaw_list = []
    for data in pred_result:
        if np.any(np.isnan(data["pred_yaw_angle"])):
            continue

        pred_gps_coord = sat_processor.pixel_to_gps(data["pred_pixel_coordinate"][0], data["pred_pixel_coordinate"][1])
        # print(f"pixel coordinate: {data["pred_coarse_coordinate"]}, EPGS 3857: {pred_gps_coord}")
        pred_gps_coord = sat_processor.convert_crs(sat_processor.crs, CRS("EPSG:4326"), pred_gps_coord[0], pred_gps_coord[1])

        pred_pixel_coordinate_list.append(data["pred_pixel_coordinate"])
        true_pixel_coordinate_list.append(data["true_pixel_coordinate"])
        pred_gps_coordinate_list.append(pred_gps_coord)
        true_gps_coordinate_list.append(data["true_gps_coordinate"])
        pred_yaw_list.append(data["pred_yaw_angle"])
        true_yaw_list.append(data["true_yaw"])

    pred_df = pd.DataFrame({
        "latitude": [ lat for lat, _ in pred_gps_coordinate_list],
        "longitude": [ lon for _, lon in pred_gps_coordinate_list]
    })
    pred_df.to_csv(Path(args.output_dir).joinpath("pred_coordinates.csv"))

    true_df = pd.DataFrame({
        "latitude": [ lat for lat, _ in true_gps_coordinate_list],
        "longitude": [ lon for _, lon in true_gps_coordinate_list]
    })
    true_df.to_csv(Path(args.output_dir).joinpath("true_coordinates.csv"))

    pred_dist_mae_pixel, pred_dist_rmse_pixel = model._common_val_test_loss(
        torch.Tensor(np.array(pred_pixel_coordinate_list)),
        torch.Tensor(np.array(true_pixel_coordinate_list))
        )
    pred_dist_mae_pixel = round(pred_dist_mae_pixel.detach().item(), 2)
    pred_dist_rmse_pixel = round(pred_dist_rmse_pixel.detach().item(), 2)
    dist_error_result = calculate_location_error_metrics(true_gps_coordinate_list, pred_gps_coordinate_list)

    pred_yaw_mae, pred_yaw_rmse, yaw_error_list = calculate_angle_error(pred_yaw_list, true_yaw_list)
    yaw_error_list = np.array(yaw_error_list).squeeze()
    pred_yaw_mae = round(pred_yaw_mae, 2)
    pred_yaw_rmse = round(pred_yaw_rmse, 2)

    data = {
        "Pred Dist MAE(pixels)": pred_dist_mae_pixel,
        "Pred Dist RMSE(pixels)": pred_dist_rmse_pixel,
        "Pred Dist MAE(meters)": dist_error_result['mae_meters'],
        "Pred Dist RMSE(meters)": dist_error_result['rmse_meters'],
        "Pred Yaw MAE(degree)": pred_yaw_mae,
        "Pred Yaw RMSE(degree)": pred_yaw_rmse
    }
    print(data)
    log_pred_result(data, Path(args.output_dir), "test_result_extend.json")

    visualize_error_histogram(np.array(dist_error_result['error_list']), Path(args.output_dir))
    visualize_error_histogram(
        np.array(yaw_error_list).squeeze(),
        Path(args.output_dir),
        title='Distribution of Yaw Prediction Errors',
        x_axis_label='Error Yaw (degrees)',
        filename='Prediction_yaw_error_histogram.png'
        )
    visualize(pred_pixel_coordinate_list, true_pixel_coordinate_list, OUTPUT_DIR, args.sat_img_png)

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
    parser.add_argument("--sat_img_png", type=str, default=r"~/Documents/hsun/datasets/satellites/202410_NTU_playground.jpg", help="dataset folder path")
    parser.add_argument("--sat_img_tif", type=str, default=r"~/Documents/hsun/datasets/satellites/202410_NTU_playground.jpg", help="dataset folder path")
    args = parser.parse_args()
    main(args)

