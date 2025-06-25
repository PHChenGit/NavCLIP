import os
from pathlib import Path
import json
from typing import List, Tuple, Optional

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image as IM
from tqdm import tqdm
from geopy.distance import geodesic

import torch
import torchvision.transforms as T
import torch.nn as nn

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
        for i, (_, _, gps_coordinate, pixel_coordinate, _) in bar:
            for coord in pixel_coordinate:
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

    A = H_pred[0:2, 0:2]
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    yaw_rad = np.arctan2(c, d) if np.isclose(a, d) and np.isclose(-b, c) else np.arctan2(-b, a)
    # yaw_rad = np.arctan2(H_pred[1, 0], H_pred[0, 0])
    yaw_angle = np.rad2deg(yaw_rad)

    return H_pred, yaw_angle, inlier_mask

# def lightglue_estimate_rotation_angle(extractor, matcher, ref_img: npt.NDArray, query_img: npt.NDArray) -> Tuple[npt.NDArray, float, npt.NDArray]:
#     from models.lightglue.utils import rbd
#     feats0 = extractor.extract(ref_img)  # auto-resize the image, disable with resize=None
#     feats1 = extractor.extract(query_img)
#     matches01 = matcher({'image0': feats0, 'image1': feats1})
#     feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
#     matches = matches01['matches']
#     points0 = feats0['keypoints'][matches[..., 0]]
#     points1 = feats1['keypoints'][matches[..., 1]]

#     if points0.shape[0] < 4:
#         raise Exception("points0 less than 4")

#     H_pred, inlier_mask = cv2.findHomography(points0.cpu().numpy(), points1.cpu().numpy(), cv2.USAC_MAGSAC, ransacReprojThreshold=3, maxIters=10000, confidence=0.9999)

#     if H_pred is None or not H_pred.all():
#         raise Exception("H_pred is None")

#     inlier_mask = inlier_mask.ravel() > 0
#     m_kpts0_valid = points0[inlier_mask]
#     m_kpts1_valid = points1[inlier_mask]

#     A = H_pred[0:2, 0:2]
#     a = A[0, 0]
#     b = A[0, 1]
#     c = A[1, 0]
#     d = A[1, 1]
#     yaw_rad = np.arctan2(c, d) if np.isclose(a, d) and np.isclose(-b, c) else np.arctan2(-b, a)
#     # yaw_rad = np.arctan2(H_pred[1, 0], H_pred[0, 0])
#     yaw_angle = np.rad2deg(yaw_rad)

#     return H_pred, yaw_angle, inlier_mask

def angle_diff(a: float, b: float) -> float:
    diff = (a - b + 180) % 360 - 180
    return diff

def calculate_angle_error(true_angles: list, pred_angles: list) -> Tuple[float, float]:
    angle_diffs = [abs(angle_diff(p, g)) for p, g in zip(pred_angles, true_angles)]
    mae = np.mean(angle_diffs)
    rmse = np.sqrt(np.mean(np.square(angle_diffs)))
    return mae, rmse, angle_diffs


def calculate_dist(true_coordinates: npt.NDArray, pred_coordinates: npt.NDArray):
    distance = np.linalg.norm(true_coordinates - pred_coordinates)
    return distance

def zoom_at(img: IM.Image, x: float, y: float, zoom: int):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), IM.Resampling.LANCZOS)


def crop_image(sat_img: IM.Image, coordinate_center: Tuple[int, int], crop_size: Tuple[int, int]):
    w, h = crop_size
    half_crop = w // 2
    cx, cy = coordinate_center
    crop_top = int(cy - half_crop)
    crop_bottom = int(cy + half_crop)
    crop_left = int(cx - half_crop)
    crop_right = int(cx + half_crop)
    crop_box = (
        crop_left,
        crop_top,
        crop_right,
        crop_bottom
    )
    copy_sat_img = sat_img.copy()
    cropped_image = copy_sat_img.crop(crop_box)
    return cropped_image

def calculate_location_error_metrics(true_locations: List[Tuple[float, float]], predicted_locations: List[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
    """
    計算一組真實位置和預測位置之間的 MAE 和 RMSE。

    參數:
    true_locations (list of tuples): 真實的 (緯度, 經度) 座標列表。
                                     例如: [(lat1_true, lon1_true), (lat2_true, lon2_true), ...]
    predicted_locations (list of tuples): 預測的 (緯度, 經度) 座標列表。
                                         例如: [(lat1_pred, lon1_pred), (lat2_pred, lon2_pred), ...]

    返回:
    dict: 包含 MAE 和 RMSE 的字典 (單位: 公尺)。
          返回 None 如果輸入列表長度不匹配或為空。
    """
    if not true_locations or not predicted_locations or len(true_locations) != len(predicted_locations):
        print("錯誤: 輸入列表為空或長度不匹配。")
        return None, None

    distance_errors_meters = []
    for true_loc, pred_loc in zip(true_locations, predicted_locations):
        # true_loc 和 pred_loc 應該是 (緯度, 經度) 的元組
        distance = geodesic(true_loc, pred_loc).meters
        distance_errors_meters.append(distance)

    errors_np = np.array(distance_errors_meters)

    mae = np.mean(errors_np) # 由於距離恆為正，abs(errors_np) 等於 errors_np
    rmse = np.sqrt(np.mean(errors_np**2))

    return {
        "mae_meters": mae,
        "rmse_meters": rmse,
        'error_list': distance_errors_meters
    }
