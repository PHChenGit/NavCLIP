from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from transformers import CLIPModel, AutoModel, AutoProcessor
from PIL import Image as IM

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import (
    load_gallery_data,
    denormalize_and_restore_image,
    estimate_rotation_angle,
    crop_image,
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
                 epochs: int = 500,
                 homography_method: str = 'mapglue'
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.CLIP = CLIPModel.from_pretrained(clip_model_name)
        # self.image_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.image_backbone = AutoModel.from_pretrained(clip_model_name)
        self.image_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery: torch.Tensor = load_gallery_data(gallery_path)
        self._initialize_gps_queue(queue_size)

        for param in self.image_backbone.parameters():
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
        """
        Only work at prediction stage.
        For estimate uav orientation and refine coordinates
        """
        if self.hparams.homography_method == 'mapglue':
            print('mapglue')
            self.homography_method = torch.jit.load(Path('~/Documents/hsun/NavCLIP/models/MapGlue/weights/fastmapglue_model.pt').expanduser())
            self.homography_method.eval()
        elif self.hparams.homography_method == 'superglue':
            raise RuntimeError('SuperGlue is not implemented.')
            # print('superglue')
            # config = {
            #     'superpoint': {
            #         'nms_radius': 4,
            #         'keypoint_threshold': 0.005,
            #         'max_keypoints': -1
            #     },
            #     'superglue': {
            #         'weights': 'outdoor',
            #         'sinkhorn_iterations': 20,
            #         'match_threshold': 0.2,
            #     }
            # }
            # self.homography_method = Matching(config).eval().to('cuda')
            # self.homography_method.eval()
        elif self.hparams.homography_method == 'lightglue':
            print('lightglue')
            raise RuntimeError('LightGlue is not implemented.')
            # self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
            # selfmatcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
        elif self.hparams.homography_method == 'sift':
            print('sift')
            self.homography_method = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.hparams.homography_method == 'orb':
            print('orb')
            self.homography_method = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise RuntimeError(f"Homography method: {self.hparams.homography_method} is invalid.")

        if sat_img:
            self.sat_img = IM.open(sat_img).convert('RGB')

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

    def encode_image(self, query_img, ref_img):
        # inputs = self.image_processor(images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad(): # 通常 CLIP 主幹不訓練
            # image_embeddings = self.CLIP.get_image_features(pixel_values=image) # [B, 768]
            query_image_embeddings = self.image_backbone.get_image_features(query_img) # [B, 768]
            ref_image_embeddings = self.image_backbone.get_image_features(ref_img) # [B, 768]

        image_features = self.image_encoder(query_image_embeddings, ref_image_embeddings) # [B, 512]
        image_features = F.normalize(image_features, dim=1)
        return image_features

    def encode_location(self, location):
        # location 應為 GPS 座標張量 [B, 2]
        location_features = self.location_encoder(location) # [B, 512]
        location_features = F.normalize(location_features, dim=1)
        return location_features

    def forward(self, query_img, ref_img, location):
        """ Computes similarity scores (logits) between fused images and locations """
        image_features = self.encode_image(query_img, ref_img)
        location_features = self.encode_location(location)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ location_features.t())

        return logits_per_image

    def _common_val_test_loss(self, pred_coords, true_coords):
        dist_mae = F.l1_loss(pred_coords, true_coords)

        dist_mse = F.mse_loss(pred_coords, true_coords)
        dist_rmse = torch.sqrt(dist_mse)

        return dist_mae, dist_rmse

    def training_step(self, batch, batch_idx):
        ref_imgs, query_imgs, gps_coordinate, coordinates, yaws = batch
        batch_size = ref_imgs.shape[0]

        gps_queue = self.get_gps_queue()
        gps_all = torch.cat([coordinates, gps_queue], dim=0)
        self.dequeue_and_enqueue(coordinates)

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)

        with torch.autocast(device_type='cuda'):
            # The forward pass now fuses query and reference images
            logits_img_gps = self(query_imgs, ref_imgs, gps_all)

        image_loss = self.criterion(logits_img_gps, labels)
        loss = image_loss

        # logits_gps_img = logits_img_gps.t()
        # location_loss = self.criterion(logits_gps_img, labels)

        # loss = (image_loss + location_loss) / 2

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('image_loss', image_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log('location_loss', location_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ref_imgs, query_imgs, gps_coordinate, coordinates, yaws = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            # Pass both query_imgs and ref_imgs to the forward method
            logits_per_image = self(query_imgs, ref_imgs, self.gps_gallery)

        probs = logits_per_image.softmax(dim=-1)
        out = torch.argmax(probs, dim=-1)
        pred_coordinates = self.gps_gallery[out]

        val_dist_mae, val_dist_rmse = self._common_val_test_loss(pred_coordinates, coordinates)

        self.log('val_dist_MAE', val_dist_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dist_RMSE', val_dist_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_dist_mae

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ref_imgs, query_imgs, coordinates, pixel_coordinate, yaws = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image = self(query_imgs, ref_imgs, self.gps_gallery)

        probs_per_image = logits_per_image.softmax(dim=-1).cpu() # [B, num_gallery_gps]

        top_k = 1
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        pred_coordinate = self.gps_gallery[top_pred.indices[0]]

        img_h, img_w = query_imgs.shape[2], query_imgs.shape[3] # B, C, H, W = [1, 3, 224, 224])
        best_pred_coordinates = pred_coordinate[0].detach().cpu().numpy()

        # This is the UAV image
        restored_query_img_pil: IM.Image = denormalize_and_restore_image(query_imgs)[0] # [224, 224]
        # This is the ground-truth satellite patch
        ref_img_pil: IM.Image = TF.to_pil_image(ref_imgs[0])
        # This is the satellite crop based on the coarse prediction
        pred_sat_crop_pil: IM.Image = crop_image(self.sat_img, (best_pred_coordinates[0], best_pred_coordinates[1]), (img_h*2, img_w*2)) # [224, 224]

        # Convert PIL images to NumPy arrays for CV2 processing
        uav_img_np = np.array(restored_query_img_pil)
        pred_sat_crop_np = np.array(pred_sat_crop_pil)
        ref_sat_img_np = np.array(ref_img_pil)

        best_H_pred = None
        best_yaw_pred = 0.

        # try:
        #     if self.hparams.homography_method in ['sift', 'orb']:
        #         gray_uav = cv2.cvtColor(uav_img_np, cv2.COLOR_RGB2GRAY)
        #         gray_sat = cv2.cvtColor(ref_sat_img_np, cv2.COLOR_RGB2GRAY)
                
        #         # Detect keypoints and compute descriptors
        #         kp1, des1 = self.homography_method.detectAndCompute(gray_uav, None)
        #         kp2, des2 = self.homography_method.detectAndCompute(gray_sat, None)

        #         if des1 is None or des2 is None:
        #             raise Exception("No descriptors found for one or both images.")
                
        #         # Match descriptors
        #         matches = self.matcher.knnMatch(des1, des2, k=2)
                
        #         # Apply Lowe's ratio test
        #         good_matches = []
        #         if matches and len(matches) > 0 and len(matches[0]) == 2:
        #             for m, n in matches:
        #                 if m.distance < 0.75 * n.distance:
        #                     good_matches.append(m)
                
        #         if len(good_matches) < 4: # A common threshold for robust estimation
        #             raise Exception(f"Not enough good matches found - only {len(good_matches)}.")
                
        #         # Extract location of good matches
        #         src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        #         dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                
        #         # Find homography
        #         H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #         if H is None:
        #             raise Exception("findHomography returned None.")
                
        #         best_H_pred = H
        #         # Calculate yaw from homography
        #         A = best_H_pred[0:2, 0:2]
        #         a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        #         yaw_rad = np.arctan2(c, d) if np.isclose(a, d) and np.isclose(-b, c) else np.arctan2(-b, a)
        #         best_yaw_pred = np.rad2deg(yaw_rad)
            
        #     elif self.hparams.homography_method == 'lightglue':
        #         pass
        #         # best_H_pred, best_yaw_pred, _ = lightglue_estimate_rotation_angle(self.extractor, self.matcher, uav_img_np, ref_sat_img_np)
            
        #     elif self.hparams.homography_method in ['mapglue']:
        #         best_H_pred, best_yaw_pred, _ = estimate_rotation_angle(self.homography_method, uav_img_np, pred_sat_crop_np)
        
        # except Exception as e:
        #     print(f"Homography estimation failed with method '{self.hparams.homography_method}': {e}")
        #     best_H_pred = None

        best_H_pred, best_yaw_pred, _ = estimate_rotation_angle(self.homography_method, uav_img_np, pred_sat_crop_np)

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

        # scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs, eta_min=1e-6)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_dist_MAE'
            }
        }
