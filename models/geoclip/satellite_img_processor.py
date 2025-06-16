import os
from typing import Dict, List, Tuple, Optional, Set

import rasterio
import numpy as np
from PIL import Image

import pyproj
from pyproj import Proj, Transformer, CRS
import pandas as pd
from tqdm import tqdm


class SatelliteImageProcessor:
    def __init__(self, tif_path: str):
        """
        初始化衛星圖像處理器
        
        Args:
            tif_path: TIF檔案路徑
        """
        self.tif_path = tif_path
        self.dataset = None
        self.transform = None
        self.crs = None
        self.transformer = None

        self._load_dataset()
    
    def _load_dataset(self):
        """載入TIF檔案並提取地理資訊"""
        try:
            self.dataset = rasterio.open(self.tif_path)
            self.transform = self.dataset.transform
            self.crs = self.dataset.crs
            
            print(f"檔案資訊:")
            print(f"  尺寸: {self.dataset.width} x {self.dataset.height}")
            print(f"  波段數: {self.dataset.count}")
            print(f"  資料型態: {self.dataset.dtypes}")
            print(f"  座標系統: {self.crs}")
            print(f"  地理轉換參數: {self.transform}")
            print(f"  邊界範圍: {self.dataset.bounds}")
            
        except Exception as e:
            print(f"載入TIF檔案失敗: {e}")
            raise
    
    def gps_to_pixel(self, lat, lon):
        """
        使用rasterio將GPS座標轉換為像素座標
        
        Args:
            lat: 緯度
            lon: 經度
            
        Returns:
            tuple: (col, row) 像素座標 (x, y)
        """
        src_crs = CRS("EPSG:4326")
        dst_crs = self.dataset.crs

        # 3. 建立一個從來源 CRS 到目標 CRS 的轉換器
        #    always_xy=True 確保輸入和輸出的順序始終是 (x, y)，即 (經度, 緯度)
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        
        # 4. 執行座標轉換：將經緯度轉換為 TIF 影像對應的投影座標 (通常是公尺)
        #    注意，pyproj 的 transform 函式預期輸入順序為 (x, y)，也就是 (lon, lat)
        x_coord, y_coord = transformer.transform(lon, lat)
        
        # 5. 使用轉換後的投影座標來獲取像素位置
        #    rasterio 的 dataset.index() 方法會回傳 (row, col)
        #    這是比直接使用 `rasterio.transform.rowcol` 更推薦的做法
        row, col = self.dataset.index(x_coord, y_coord)

        return col, row
    
    def pixel_to_gps(self, col, row):
        """
        使用rasterio將像素座標轉換為GPS座標
        
        Args:
            col: 列座標 (x)
            row: 行座標 (y)
            
        Returns:
            tuple: (lon, lat) GPS座標
        """
        lon, lat = rasterio.transform.xy(self.transform, row, col)
        return lat, lon
    
    def read_image_data(self):
        """
        讀取圖像資料
        
        Returns:
            numpy.ndarray: 圖像資料陣列
        """
        # 讀取所有波段的資料
        image_data = self.dataset.read()
        
        # 如果是多波段，轉換為 (height, width, bands) 格式
        if image_data.shape[0] > 1:
            image_data = np.transpose(image_data, (1, 2, 0))
        else:
            # 單波段圖像，移除第一個維度
            image_data = image_data[0]
        
        return image_data
    
    def convert_image_to_pillow(self, image_data):
        # 轉換為Pillow圖像
        if len(image_data.shape) == 3:
            # 多波段圖像
            if image_data.shape[2] == 1:
                # 單波段但有第三個維度
                pil_img = Image.fromarray(image_data[:,:,0], mode='L')
            elif image_data.shape[2] == 3:
                # RGB圖像
                if image_data.dtype != np.uint8:
                    min_val = np.min(image_data)
                    max_val = np.max(image_data)
                    if max_val > min_val:
                        image_data = ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        image_data = image_data.astype(np.uint8)
                pil_img = Image.fromarray(image_data, mode='RGB')
            else:
                # 多於3個波段，只取前3個
                rgb_data = image_data[:,:,:3]
                if rgb_data.dtype != np.uint8:
                    min_val = np.min(rgb_data)
                    max_val = np.max(rgb_data)
                    if max_val > min_val:
                        rgb_data = ((rgb_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        rgb_data = rgb_data.astype(np.uint8)
                pil_img = Image.fromarray(rgb_data, mode='RGB')
        else:
            # 單波段圖像
            if image_data.dtype != np.uint8:
                min_val = np.min(image_data)
                max_val = np.max(image_data)
                if max_val > min_val:
                    image_data = ((image_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    image_data = image_data.astype(np.uint8)
            pil_img = Image.fromarray(image_data, mode='L')

        return pil_img