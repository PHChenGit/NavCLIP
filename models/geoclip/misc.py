import os
from pathlib import Path
import json
from typing import List, Tuple, Optional

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image as IM
from PIL import ImageDraw
from tqdm import tqdm
import rasterio
from rasterio.plot import show

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

class EstimateHomoException(Exception):
    """Exception raised for custom error scenarios.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


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
        raise EstimateHomoException("points0 less than 4")

    H_pred, inlier_mask = cv2.findHomography(points0.cpu().numpy(), points1.cpu().numpy(), cv2.USAC_MAGSAC, ransacReprojThreshold=3, maxIters=10000, confidence=0.9999)

    if H_pred is None or not H_pred.all():
        raise EstimateHomoException("H_pred is None")

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

from geopy.distance import geodesic
import math
import numpy as np # numpy 使得計算更簡潔

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
        "rmse_meters": rmse
    }


def gps_to_pixel_and_draw(tiff_path, gps_coordinates, output_path="output_with_circles.png", circle_radius=50, circle_fill="red"):
    """
    將 GPS 座標轉換為像素座標，並在 TIFF 影像上繪製圓圈標記。

    參數:
        tiff_path (str): TIFF 衛星圖片的路徑。
        gps_coordinates (list of tuples): GPS 座標列表，每個元組格式為 (緯度, 經度)。
        output_path (str): 繪製圓圈後輸出的圖片路徑 (建議使用 PNG 或 JPG)。
        circle_radius (int): 圓圈的半徑 (像素)。
        circle_fill (str): 圓圈的填充顏色。
    """
    try:
        with rasterio.open(tiff_path) as src:
            # 獲取影像的 metadata
            # print(f"影像 CRS: {src.crs}")
            # print(f"影像 Transform: {src.transform}")
            # print(f"影像寬度: {src.width}, 影像高度: {src.height}")
            band_count = src.count
            print(f"影像 '{tiff_path}' 的波段數量為: {band_count}")

            pixel_coords_to_draw = []
            for lat, lon in gps_coordinates:
                # 使用 rasterio.DatasetReader.index() 將地理座標轉換為行列號 (像素座標)
                # 注意：rasterio 的 index 通常返回 (row, col)，對應 (y, x)
                try:
                    row, col = src.index(lon, lat) # 注意經緯度的順序
                    # 檢查轉換後的像素座標是否在影像範圍內
                    if 0 <= row < src.height and 0 <= col < src.width:
                        pixel_coords_to_draw.append((col, row)) # Pillow 使用 (x, y)
                        # print(f"GPS ({lat}, {lon}) -> Pixel ({col}, {row})")
                    else:
                        print(f"警告: GPS ({lat}, {lon}) 轉換後的像素座標 ({col}, {row}) 超出影像範圍。")
                except Exception as e:
                    print(f"錯誤：轉換 GPS ({lat}, {lon}) 失敗: {e}")
                    print("請確保 GPS 座標在 TIFF 影像的地理範圍內，且 TIFF 影像包含正確的地理參考資訊。")
                    continue

            if not pixel_coords_to_draw:
                print("沒有有效的像素座標可以繪製。")
                return
            
            

            # 讀取影像數據，這裡只讀取第一個波段用於視覺化
            # 如果你的 TIFF 是多波段彩色影像，你可能需要讀取多個波段
            # 例如：img_data = src.read([1, 2, 3])
            # 為了簡化，我們先假設是灰階或只顯示單一波段
            # 注意：直接讀取非常大的 TIFF 檔案可能會消耗大量記憶體。
            # 對於超大檔案，可能需要分塊處理或降採樣。
            # 這裡我們假設 Pillow 可以處理。

            # 轉換 rasterio 讀取的 NumPy 陣列到 Pillow Image
            # 假設是單波段影像，需要正規化到 0-255
            # 為了演示，我們直接創建一個 Pillow 影像來繪圖，
            # 更理想的做法是將 src.read() 的數據轉換為 Pillow Image
            # 這部分取決於你的 TIFF 具體格式和波段數

            # 為了在原始影像上繪圖，我們需要將影像讀取到 Pillow
            # 注意：直接讀取非常大的 TIFF 檔案 (35092, 24308) 到記憶體中
            # 使用 Pillow 可能會非常耗時且消耗大量記憶體。
            # 這裡提供一個基本流程，實際應用中可能需要優化。

            print("正在讀取影像並準備繪圖...")
            # 嘗試讀取第一個波段
            try:
                img_array = src.read(1)
                # 將 NumPy 陣列轉換為 Pillow 影像 (灰階)
                # 你可能需要根據你的影像數據類型進行調整 (例如，正規化到 0-255)
                # 這裡假設數據可以直接轉為 'L' (灰階) 模式
                # 如果數據範圍不是 0-255，需要先進行正規化
                # 例如: img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                image = IM.fromarray(img_array).convert("RGB") # 轉換為 RGB 以便繪製彩色圓圈
            except MemoryError:
                print("錯誤：讀取 TIFF 影像到記憶體時發生 MemoryError。影像過大。")
                print("考慮使用分塊處理或降採樣等方法。")
                return
            except Exception as e:
                print(f"讀取或轉換影像時發生錯誤: {e}")
                return

            draw = ImageDraw.Draw(image)

            for x, y in pixel_coords_to_draw:
                # 計算圓圈的邊界框 (x0, y0, x1, y1)
                x0 = x - circle_radius
                y0 = y - circle_radius
                x1 = x + circle_radius
                y1 = y + circle_radius
                draw.ellipse((x0, y0, x1, y1), fill=circle_fill, outline=circle_fill)

            # 儲存或顯示影像
            image.save(output_path)
            print(f"已將標記後的影像儲存到: {output_path}")
            # image.show() # 如果你想直接顯示影像

    except rasterio.errors.RasterioIOError as e:
        print(f"Rasterio 錯誤: {e}")
        print("請確保 TIFF 檔案路徑正確，且檔案未損壞並包含有效的地理參考資訊。")
    except FileNotFoundError:
        print(f"錯誤: 找不到 TIFF 檔案 '{tiff_path}'")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
