from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import torchvision.transforms as T

import pytorch_lightning as pl
from transformers import CLIPModel, AutoProcessor
from PIL import Image as IM
import time
from geopy.distance import geodesic

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder, get_positional_encoding, get_neural_network
from .misc import (
    load_gallery_data,
    denormalize_and_restore_image,
    estimate_rotation_angle,
    EstimateHomoException,
)
from .loss import NavCLIPLoss

from models.geoclip.satellite_img_processor import SatelliteImageProcessor

class GeoCLIPLightning(pl.LightningModule):
    def __init__(self,
                 gallery_path: str,
                 sat_img: str = "",
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 queue_size: int = 4096,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 scheduler_gamma: float = 0.5,
                 epochs: int=150,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.CLIP = CLIPModel.from_pretrained(clip_model_name)
        # self.image_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.image_encoder = ImageEncoder()
        # self.posenc = get_positional_encoding(name='sphericalharmonics', harmonics_calculation='analytic', legendre_polys=16, min_radius=1, max_radius=260, frequency_num=16)
        # self.nnet = get_neural_network(name='siren', input_dim=self.posenc.embedding_dim, num_classes=512, dim_hidden=256, num_layers=2)
        self.location_encoder = LocationEncoder()

        self.gps_gallery: torch.Tensor = load_gallery_data(gallery_path)
        self._initialize_gps_queue(queue_size)

        for param in self.CLIP.parameters():
            param.requires_grad = False

        # self.criterion = nn.CrossEntropyLoss()
        self.loss_fn = NavCLIPLoss()
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

        self.sat_img_processor = SatelliteImageProcessor(sat_img)

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

        if torch.any(torch.isnan(location_features)):
            print(location_features)
            raise Exception("location features has nan")
        # print(f"image_features: {image_features.shape}, location_features shape: {location_features.shape}")

        logit_scale = self.logit_scale.exp()
        # logits_per_image: [N, M]
        logits_per_image = logit_scale * (image_features @ location_features.t())
        # print(f"logits_per_image {logits_per_image.shape}")
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location

    def _common_val_test_loss(self, pred_coords, true_coords):
        # pred_locations = []
        # for loc in pred_coords:
        #     lat, lon = self.sat_img_processor.pixel_to_gps(loc[0].detach().cpu().numpy(),loc[1].detach().cpu().numpy())
        #     pred_locations.append([lat, lon])

        # true_locations = []
        # for loc in true_coords:
        #     lat, lon = self.sat_img_processor.pixel_to_gps(loc[0].detach().cpu().numpy(), loc[1].detach().cpu().numpy())
        #     true_locations.append([lat, lon])

        distance_errors_meters = []
        for true_loc, pred_loc in zip(true_coords, pred_coords):
            distance = geodesic(true_loc, pred_loc).meters
            distance_errors_meters.append(distance)

        errors_np = np.array(distance_errors_meters)

        dist_mae = np.mean(errors_np) # 由於距離恆為正，abs(errors_np) 等於 errors_np
        dist_rmse = np.sqrt(np.mean(errors_np**2))

        return dist_mae, dist_rmse

    def training_step(self, batch, batch_idx):
        query_imgs, coordinates, yaws = batch

        gps_queue = self.get_gps_queue()
        gps_all = torch.cat([coordinates, gps_queue], dim=0)
        self.dequeue_and_enqueue(coordinates)

        with torch.autocast(device_type='cuda'):
            logits_per_image, logits_per_coord = self(query_imgs, gps_all)

        loss, image_loss, location_loss = self.loss_fn(logits_per_image, logits_per_coord)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('image loss', image_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('location loss', location_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        query_imgs, coordinates, yaws = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image, logits_per_coord = self(query_imgs, self.gps_gallery)

        probs = logits_per_image.softmax(dim=-1)
        out = torch.argmax(probs, dim=-1)
        pred_coordinates = self.gps_gallery[out]

        val_dist_mae, val_dist_rmse = self._common_val_test_loss(pred_coordinates, coordinates)

        self.log('val_dist_MAE', val_dist_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dist_RMSE', val_dist_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_dist_mae

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        coordinates: 是pixel
        """
        query_imgs, coordinates, yaws, sat_img_PIL, query_img_PIL, = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image, logits_per_coord = self(query_imgs, self.gps_gallery) # [B, num_gallery_gps]

        probs_per_image = logits_per_image.softmax(dim=-1).cpu() # [B, num_gallery_gps]

        top_k = 1
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        pred_coordinate = self.gps_gallery[top_pred.indices[0]] # 正規化後的pixel座標

        # img_h, img_w = query_imgs.shape[2], query_imgs.shape[3] # B, C, H, W = [1, 3, 224, 224])
        best_pred_coordinates = pred_coordinate[0].detach().cpu().numpy()
        restored_query_imgs: IM.Image = T.ToPILImage()(query_img_PIL[0])
        restored_sat_imgs = T.ToPILImage()(sat_img_PIL[0])

        best_H_pred = None
        best_yaw_pred = np.nan
        best_matches = None

        try:
            best_H_pred, best_yaw_pred, best_matches = estimate_rotation_angle(self.mapglue, np.array(restored_query_imgs), np.array(restored_sat_imgs))
        except EstimateHomoException as e:
            print(e)
            best_H_pred = None
            best_yaw_pred = np.nan
            best_matches = None


        if best_H_pred is None:
            return {
                "pred_coarse_coordinate": best_pred_coordinates,
                "true_coordinate": coordinates[0].detach().cpu().numpy(),
                "pred_yaw_angle": np.nan,
                "true_yaw": yaws.cpu().numpy()
            }

        return {
            "pred_coarse_coordinate": best_pred_coordinates,
            "true_coordinate": coordinates[0].detach().cpu().numpy(),
            "pred_yaw_angle": best_yaw_pred,
            "true_yaw": yaws.cpu().numpy()
        }

    def configure_optimizers(self):
        params_to_optimize = [
            {'params': self.image_encoder.parameters(), "lr": self.hparams.learning_rate},
            {'params': self.location_encoder.parameters(), "lr": self.hparams.learning_rate},
            {'params': [self.logit_scale], "lr": self.hparams.learning_rate},
        ]

        optimizer = AdamW(params_to_optimize,
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay,
                          betas=(0.9, 0.999),
                          eps=1e-08)

        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_dist_MAE'
            }
        }
