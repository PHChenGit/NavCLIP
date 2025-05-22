import os
from pathlib import Path
import json
from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image as IM
from tqdm import tqdm
import rasterio
from rasterio.transform import rowcol

import torch
import torchvision.transforms as T

file_dir = os.path.dirname(os.path.realpath(__file__))

def load_gallery_data(csv_file: str) -> torch.Tensor:
    data = pd.read_csv(csv_file)
    lat_lon = data[['LAT', 'LON']]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)

    return gps_tensor

def log_pred_result(result: dict, output_path: str, filename: str) -> None:
    full_path: Path = (Path(output_path) / filename).resolve()
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.touch(exist_ok=True)

    with open(str(full_path.resolve()), "w") as json_file:
        json.dump(result, json_file, indent=4)
    print(f">\t Prediction result has been written to {full_path}")


def create_gallery(gallery_folder, data_loader):
    gallery_path = gallery_folder.joinpath("gallery.csv")
    if not gallery_path.exists():
        all_pose = []
        bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Creating pose gallery",
        )
        for i, (_, _, ref_coordinates, _, _) in bar:
            for coord in ref_coordinates:
                lat, lon = coord.detach().cpu().numpy()
                all_pose.append([lat, lon])
        df = pd.DataFrame(all_pose, columns=["LAT", "LON"])
        df.to_csv(gallery_path, index=False)
        print(f"gallery is created successfully: {gallery_path}")
    return gallery_path

def denormalize_and_restore_image(normalized_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) -> List[IM.Image]:
    denorm = T.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    denormalized_tensor = denorm(normalized_tensor)
    
    batch_size = normalized_tensor.shape[0]
    restored_images = []
    
    for i in range(batch_size):
        denormalized_tensor = denorm(normalized_tensor[i])
        
        denormalized_tensor = denormalized_tensor.clamp(0, 1)
        pil_image = T.ToPILImage()(denormalized_tensor)
        
        restored_images.append(pil_image)
    
    return restored_images


def estimate_rotation_angle(model, ref_img: npt.NDArray, query_img: npt.NDArray) -> Tuple[npt.NDArray, float, npt.NDArray]:
    # Convert BGR to RGB
    image0 = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)

    image0 = torch.from_numpy(image0)
    image1 = torch.from_numpy(image1)
    num_keypoints = torch.tensor(2048) # Defaults to 2048

    # Run inference
    # mapglue accept image shape [H, W, C]
    points_tensor = model(image0, image1, num_keypoints)
    points0 = points_tensor[:, :2]
    points1 = points_tensor[:, 2:]

    if points0.shape[0] < 4:
        raise Exception("points0 less than 4")

    H_pred, inlier_mask = cv2.findHomography(points0.cpu().numpy(), points1.cpu().numpy(), cv2.USAC_MAGSAC, ransacReprojThreshold=3, maxIters=10000, confidence=0.9999)

    if H_pred is None or not H_pred.all():
        raise Exception("H_pred is None")

    inlier_mask = inlier_mask.ravel() > 0
    m_kpts0_valid = points0[inlier_mask]
    m_kpts1_valid = points1[inlier_mask]

    yaw_rad = np.arctan2(H_pred[1, 0], H_pred[0, 0])
    yaw_angle = np.rad2deg(yaw_rad)

    return H_pred, yaw_angle, inlier_mask

def angle_diff(a: float, b: float) -> float:
    diff = (a - b + 180) % 360 - 180
    return diff

def calculate_angle_error(true_angles: list, pred_angles: list) -> Tuple[float, float]:
    angle_diffs = [abs(angle_diff(p, g)) for p, g in zip(pred_angles, true_angles)]
    mae = np.mean(angle_diffs)
    rmse = np.sqrt(np.mean(np.square(angle_diffs)))
    return mae, rmse


def calculate_dist(true_coordinates: npt.NDArray, pred_coordinates: npt.NDArray):
    distance = np.linalg.norm(true_coordinates - pred_coordinates)
    return distance

def zoom_at(img: IM.Image, x: float, y: float, zoom: int):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), IM.Resampling.LANCZOS)

def geo_to_pixel(lat, lon, transform, width, height):
    """
    將地理座標(緯度，經度)轉換為圖像像素座標(row, col)
    
    參數:
    lat - 緯度
    lon - 經度
    transform - rasterio 轉換矩陣
    width, height - 圖像尺寸
    
    返回:
    (col, row) - 像素座標 (x, y)
    """

    print(f">\t[DEBUG] lat: {lat}, lon: {lon}")
    # 使用 rasterio 的 rowcol 函數進行座標轉換 (注意：順序是 x=經度, y=緯度)
    row, col = rowcol(transform, lon, lat)
    print(f">\t[DEBUG] row: {row}, col: {col}")
    
    # 確保座標在圖像範圍內
    col = max(0, min(width - 1, col))
    row = max(0, min(height - 1, row))
    
    return col, row
