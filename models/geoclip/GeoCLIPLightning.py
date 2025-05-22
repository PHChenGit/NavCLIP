from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torchvision.transforms as T

import pytorch_lightning as pl
from transformers import CLIPModel, AutoProcessor
from PIL import Image as IM
import rasterio

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import (
    load_gallery_data,
    denormalize_and_restore_image,
    estimate_rotation_angle,
    geo_to_pixel
)

class GeoCLIPLightning(pl.LightningModule):
    def __init__(self,
                 gallery_path: str,
                 sat_img: str = "",
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 queue_size: int = 4096,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 scheduler_gamma: float = 0.5,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.CLIP = CLIPModel.from_pretrained(clip_model_name)
        # self.image_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery: torch.Tensor = load_gallery_data(gallery_path)
        self._initialize_gps_queue(queue_size)

        for param in self.CLIP.parameters():
            param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        """
        For saving prediction result and ground-truth in prediction step
        """
        self.pred_coordinate_list = []
        self.true_coordinate_list = []
        self.pred_yaw_list = []
        self.true_yaw_list = []
        """
        Only work at prediction stage.
        For estimate uav orientation and refine coordinates
        """
        self.mapglue = torch.jit.load(Path('~/Documents/hsun/NavCLIP/models/MapGlue/weights/fastmapglue_model.pt').expanduser())
        self.mapglue.eval()
        if sat_img:
            # self.sat_img = IM.open(sat_img).convert('RGB')
            self.sat_img = sat_img

    def load_weights(self, pretrained_dir):
        """Loads weights to the correct device"""
        device = self.device
        print(f"load_weights pretrained_dir: {pretrained_dir}")
        
        image_encoder_path = Path(pretrained_dir) / "image_encoder.pth"
        weights = torch.load(image_encoder_path, map_location=device, weights_only=True)
        self.image_encoder.load_state_dict(weights)

        location_encoder_path = Path(pretrained_dir) / "location_encoder.pth"
        weights = torch.load(location_encoder_path, map_location=device, weights_only=True)
        self.location_encoder.load_state_dict(weights)

        logit_scale_path = Path(pretrained_dir) / "logit_scale.pth" 
        weights = torch.load(logit_scale_path, map_location=device, weights_only=True)
        self.logit_scale.data = weights.to(device)

    def _initialize_gps_queue(self, queue_size):
        self.hparams.queue_size = queue_size
        # 使用 register_buffer 將 queue 註冊為模型狀態的一部分，但不是模型參數
        self.register_buffer("gps_queue", torch.randn(2, self.hparams.queue_size))
        self.gps_queue = F.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ 更新 GPS 佇列 """
        gps = gps.to(self.gps_queue.device)
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr.item()) # 使用 .item()

        # 確保 queue_size 可被 batch_size 整除
        if self.hparams.queue_size % gps_batch_size != 0:
             raise ValueError(f"佇列大小 {self.hparams.queue_size} 必須能被批次大小 {gps_batch_size} 整除")

        # 取代佇列中的 GPS 數據
        self.gps_queue[:, gps_ptr : gps_ptr + gps_batch_size] = gps.t() # [2, B]
        gps_ptr = (gps_ptr + gps_batch_size) % self.hparams.queue_size  # 移動指標
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()

    def encode_image(self, image):
        with torch.no_grad(): # 通常 CLIP 主幹不訓練
            image_embeddings = self.CLIP.get_image_features(pixel_values=image) # [B, 768]
        image_features = self.image_encoder(image_embeddings) # [B, 512]
        image_features = F.normalize(image_features, dim=1)
        return image_features

    def encode_location(self, location):
        # location 應為 GPS 座標張量 [B, 2]
        location_features = self.location_encoder(location) # [B, 512]
        location_features = F.normalize(location_features, dim=1)
        return location_features

    def forward(self, image, location):
        """ 計算圖像和位置之間的相似度分數 (logits) """
        image_features = self.encode_image(image)     # [N, 512]
        location_features = self.encode_location(location) # [M, 512]

        logit_scale = self.logit_scale.exp()
        # logits_per_image: [N, M]
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    def _common_val_test_loss(self, pred_coords, true_coords):
        dist_mae = F.l1_loss(pred_coords, true_coords)

        dist_mse = F.mse_loss(pred_coords, true_coords)
        dist_rmse = torch.sqrt(dist_mse)

        return dist_mae, dist_rmse

    def training_step(self, batch, batch_idx):
        def _train(imgs, gps_all, targets_img_gps):
            with torch.autocast(device_type='cuda'):
                logits_img_gps = self(imgs, gps_all)
            loss = self.criterion(logits_img_gps, targets_img_gps)

            return loss

        ref_imgs, query_imgs, coordinates, yaws = batch
        batch_size = ref_imgs.shape[0]

        gps_queue = self.get_gps_queue()
        gps_all = torch.cat([coordinates, gps_queue], dim=0)
        self.dequeue_and_enqueue(coordinates)

        # 創建標籤 (對角線為 1，代表匹配的 image-location 對)
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)

        ref_loss = _train(ref_imgs, gps_all, labels)
        query_loss = _train(query_imgs, gps_all, labels)
        loss = ref_loss + query_loss

        # 記錄損失
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ref_imgs, query_imgs, coordinates, yaws = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image = self(query_imgs, self.gps_gallery)

        probs = logits_per_image.softmax(dim=-1)
        out = torch.argmax(probs, dim=-1)
        pred_coordinates = self.gps_gallery[out]

        val_dist_mae, val_dist_rmse = self._common_val_test_loss(pred_coordinates, coordinates)

        self.log('val_dist_MAE', val_dist_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dist_RMSE', val_dist_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_dist_mae

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # with rasterio.open(self.sat_img) as src:
        #     tiff_crs = src.crs
        #     sat_width = src.width
        #     sat_height = src.height
        #     geo_transform = src.transform
        #     # print(f"TIFF Image CRS: {tiff_crs}")
        #     # print(f"TIFF Image Transform: {geo_transform}")

        #     satellite_img = src.read()
            
        #     # 如果數據是多波段的，取前三個波段作為 RGB
        #     if satellite_img.shape[0] >= 3:
        #         # 將數據從(bands, height, width)轉換為(height, width, bands)
        #         satellite_img = np.transpose(satellite_img[0:3], (1, 2, 0))
        #     elif satellite_img.shape[0] == 1:
        #         # 如果只有一個波段，複製為3通道灰度圖
        #         satellite_img = np.transpose(satellite_img, (1, 2, 0))
        #         satellite_img = np.concatenate([satellite_img] * 3, axis=2)

        sat_imgs, query_imgs, coordinates, yaws, sat_img_ori, query_img_ori = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image = self(query_imgs, self.gps_gallery) # [B, num_gallery_gps]

        probs_per_image = logits_per_image.softmax(dim=-1).cpu() # [B, num_gallery_gps]

        top_k = 1
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        pred_coordinate = self.gps_gallery[top_pred.indices[0]]

        img_h, img_w = query_imgs.shape[2], query_imgs.shape[3] # B, C, H, W = [1, 3, 224, 224])
        best_pred_coordinates = pred_coordinate[0].cpu().numpy()
        # print(f"best_pred_coordinates: {best_pred_coordinates}")
        # # best_pred_img: IM.Image = crop_image(self.sat_img, (best_pred_coordinates[0], best_pred_coordinates[1]), (img_h*2, img_w*2)) # [224, 224]

        # # lat, lon
        # print(f">\t[DEBUG] lat: {best_pred_coordinates[0]}, lon: {best_pred_coordinates[1]}")
        # pixel_x, pixel_y = geo_to_pixel(best_pred_coordinates[0], best_pred_coordinates[1], geo_transform, sat_width, sat_height)
        # pixel_x = np.round(pixel_x)
        # pixel_y = np.round(pixel_y)
        # print(f"lat: {np.round(best_pred_coordinates[0], 2)}, lon: {np.round(best_pred_coordinates[1], 2)}")
        # print(f"pred: {pixel_x}, {pixel_y}, true: (33911, 18963)")

        # exit()
        # crop_size = 800
        # half_crop = crop_size // 2
        # x_min = max(0, pixel_x - half_crop)
        # y_min = max(0, pixel_y - half_crop)
        # x_max = min(sat_width, pixel_x + half_crop)
        # y_max = min(sat_height, pixel_y + half_crop)
        # cropped_img = satellite_img[y_min:y_max, x_min:x_max]
        # best_pred_img = IM.fromarray(cropped_img.astype(np.uint8))
        # best_pred_img.show()

        # restored_query_imgs: IM.Image = denormalize_and_restore_image(query_imgs)[0] # [224, 224]
        # restored_sat_imgs = denormalize_and_restore_image(sat_imgs)[0]

        query_img_ori = T.ToPILImage()(query_img_ori[0])
        sat_img_ori = T.ToPILImage()(sat_img_ori[0])
        best_H_pred = None
        best_yaw_pred = 0.
        best_matches = None

        try:
            best_H_pred, best_yaw_pred, best_matches = estimate_rotation_angle(self.mapglue, np.array(query_img_ori), np.array(sat_img_ori))
        except Exception as e:
            print(f"pred image not found matches, {e}")
            best_matches = np.array([])
            best_H_pred = None

        # if best_H_pred is None:
        #     return {
        #         "pred_coarse_coordinate": pred_coordinate.detach().cpu().numpy(),
        #         "true_coordinate": coordinates.cpu().numpy(),
        #         "pred_yaw_angle": np.nan,
        #         "true_yaw": yaws.cpu().numpy()
        #     }

        return {
            "pred_coarse_coordinate": pred_coordinate.detach().cpu().numpy(),
            "true_coordinate": coordinates.cpu().numpy(),
            "pred_yaw_angle": best_yaw_pred,
            "true_yaw": yaws.cpu().numpy()
        }

    def configure_optimizers(self):
        params_to_optimize = [
            {'params': self.image_encoder.parameters(), "lr": 3e-4},
            {'params': self.location_encoder.parameters(), "lr": 3e-4},
        ]

        optimizer = AdamW(params_to_optimize,
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay,
                          betas=(0.9, 0.999),
                          eps=1e-08)

        # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        # scheduler = MultiStepLR(optimizer, [30, 60, 80], gamma=self.hparams.scheduler_gamma)
        scheduler = MultiStepLR(optimizer, [50, 80, 110, 130], gamma=self.hparams.scheduler_gamma)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_dist_MAE'
            }
        }
