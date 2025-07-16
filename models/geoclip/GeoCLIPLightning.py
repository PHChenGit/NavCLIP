from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl
from transformers import CLIPModel, AutoProcessor
from PIL import Image as IM
from models.SuperGlue.match_pair import superglue_matching
from models.LightGlue.match_pair import lightglue_matching
from models.LightGlue.lightglue.lightglue import LightGlue
from models.LightGlue.lightglue.superpoint import SuperPoint

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
                 epochs: int = 200,
                 homography_method: str = 'mapglue'
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
        Only work at prediction stage.
        For estimate uav orientation and refine coordinates
        """
        if homography_method == 'mapglue':
            self.feat_matching_model = torch.jit.load(Path('~/Documents/hsun/NavCLIP/models/MapGlue/weights/fastmapglue_model.pt').expanduser())
            self.feat_matching_model.eval()
        elif homography_method == 'lightglue':
            # Initialize feature extractor and matcher
            self.feature_extractor = SuperPoint(max_num_keypoints=2048).eval()
            self.lightglue_matcher = LightGlue(features='superpoint').eval()
        elif self.hparams.homography_method == 'sift':
            print('sift')
            self.homography_method = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.hparams.homography_method == 'orb':
            print('orb')
            self.homography_method = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
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

    def encode_image(self, image):
        with torch.no_grad(): # 通常 CLIP 主幹不訓練
            image_embeddings = self.CLIP.get_image_features(pixel_values=image) # [B, 768]
            # image_embeddings_2 = self.CLIP.get_image_features(pixel_values=img2) # [B, 768]
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
        ref_imgs, query_imgs, gps_coordinate, pixel_coordinate, yaws = batch
        batch_size = ref_imgs.shape[0]

        gps_queue = self.get_gps_queue()
        gps_all = torch.cat([pixel_coordinate, gps_queue], dim=0)
        self.dequeue_and_enqueue(pixel_coordinate)

        # 創建標籤 (對角線為 1，代表匹配的 image-location 對)
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)

        logits_ref_img_gps = self(ref_imgs, gps_all)
        ref_loss = self.criterion(logits_ref_img_gps, labels)

        logits_query_img_gps = self(query_imgs, gps_all)
        query_loss = self.criterion(logits_query_img_gps, labels)

        loss = ref_loss + query_loss

        # 記錄損失
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        ref_imgs, query_imgs, gps_coordinate, pixel_coordinate, yaws = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self(query_imgs, self.gps_gallery)
        probs = logits_per_image.softmax(dim=-1)
        out = torch.argmax(probs, dim=-1)
        pred_coordinates = self.gps_gallery[out]

        val_dist_mae, val_dist_rmse = self._common_val_test_loss(pred_coordinates, pixel_coordinate)

        self.log('val_dist_MAE', val_dist_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dist_RMSE', val_dist_rmse, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return val_dist_mae

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sat_img, query_imgs, gps_coordinate_3d, pixel_coordinate_3d, drone_filename = batch
        yaws = pixel_coordinate_3d[:, 2].detach().cpu().numpy()

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image = self(query_imgs, self.gps_gallery) # [B, num_gallery_gps]
            probs_per_image = logits_per_image.softmax(dim=-1).cpu() # [B, num_gallery_gps]

        top_k = 1
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        pred_coordinate = self.gps_gallery[top_pred.indices[0]]
        best_pred_coordinates = pred_coordinate[0].detach().cpu().numpy()

        restored_query_img_pil: IM.Image = denormalize_and_restore_image(query_imgs)[0] # [224, 224]
        restored_ref_img_pil: IM.Image = denormalize_and_restore_image(sat_img)[0] # [224, 224]
        pred_sat_crop_pil: IM.Image = crop_image(self.sat_img, (best_pred_coordinates[0], best_pred_coordinates[1]), (224, 224))
        # pred_sat_crop_pil.save(f"/home/rvl1421/Documents/hsun/datasets/DJI_NTU_5/matching_kpt_1/pred_sat_images/{drone_filename}.png")

        uav_img_np = np.array(restored_query_img_pil)
        # pred_sat_crop_tensor = self.glue_aug(pred_sat_crop_pil, return_tensor=True)
        pred_sat_crop_np = np.array(pred_sat_crop_pil)
        ref_sat_img_np = np.array(restored_ref_img_pil)
        best_yaw_pred = np.nan

        if self.hparams.homography_method == 'mapglue':
            try:
                best_H_pred, best_yaw_pred, best_matches, pt0, pt1 = estimate_rotation_angle(self.mapglue, uav_img_np, pred_sat_crop_np)

                if best_yaw_pred >= 20:
                    pt0_np = pt0.cpu().numpy()
                    pt1_np = pt1.cpu().numpy()
                    keypoints0 = [cv2.KeyPoint(p[0], p[1], 1) for p in pt0_np]
                    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in pt1_np]

                    inlier_indices = np.where(best_matches.flatten() > 0)[0]
                    matches_to_draw = [cv2.DMatch(i, i, 0) for i in inlier_indices]

                    matched_kpt_img = cv2.drawMatches(
                            uav_img_np, 
                            keypoints0, 
                            pred_sat_crop_np, 
                            keypoints1, 
                            matches_to_draw, # Pass the list of DMatch objects for inliers
                            None, 
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                    cv2.imwrite(f"/home/rvl1421/Documents/hsun/datasets/DJI_NTU_5/matching_kpt_1/{drone_filename}_{best_pred_coordinates[0]}_{best_pred_coordinates[1]}_{pixel_coordinate_3d[0, 0]}_{pixel_coordinate_3d[0, 1]}.png", matched_kpt_img)
            except Exception as e:
                print(f"pred image not found matches, {e}")
                best_matches = np.array([])
                best_H_pred = None
        elif self.hparams.homography_method == 'superglue':
            try:
                best_H_pred, best_yaw_pred = superglue_matching(uav_img_np, pred_sat_crop_np)
            except Exception as e:
                print(f"pred image not found matches, {e}")
                best_matches = np.array([])
                best_H_pred = None
        elif self.hparams.homography_method == 'lightglue':
            best_H_pred, best_yaw_pred = lightglue_matching(uav_img_np, pred_sat_crop_np, self.feature_extractor, self.lightglue_matcher, self.device)
        elif self.hparams.homography_method in ['sift', 'orb']:
            gray_uav = cv2.cvtColor(uav_img_np, cv2.COLOR_RGB2GRAY)
            gray_sat = cv2.cvtColor(ref_sat_img_np, cv2.COLOR_RGB2GRAY)
            
            # Detect keypoints and compute descriptors
            kp1, des1 = self.homography_method.detectAndCompute(gray_uav, None)
            kp2, des2 = self.homography_method.detectAndCompute(gray_sat, None)

            if des1 is None or des2 is None:
                raise Exception("No descriptors found for one or both images.")
            
            # Match descriptors
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            # print(f"matches: {matches}, len of matches: {len(matches[0])}")
            if matches and len(matches) > 0 and len(matches[0]) == 2:
                for m, n in matches:
                    if m.distance < 0.9 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 4: # A common threshold for robust estimation
                raise Exception(f"Not enough good matches found - only {len(good_matches)}.")
            
            # Extract location of good matches
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            
            # Find homography
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                raise Exception("findHomography returned None.")
            
            best_H_pred = H
            # Calculate yaw from homography
            A = best_H_pred[0:2, 0:2]
            a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
            yaw_rad = np.arctan2(c, d) if np.isclose(a, d) and np.isclose(-b, c) else np.arctan2(-b, a)
            best_yaw_pred = np.rad2deg(yaw_rad)

            if best_yaw_pred >= 20:
                matched_kpt_img = cv2.drawMatches(
                                uav_img_np, 
                                kp1, 
                                pred_sat_crop_np, 
                                kp2, 
                                good_matches, # Pass the list of DMatch objects for inliers
                                None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                            )
                cv2.imwrite(f"/home/rvl1421/Documents/hsun/NavCLIP/output/openai_clip_image_encoder_sa_and_gelu_with_DJI_NTU_5/test/route_5_sift/{drone_filename}_{best_pred_coordinates[0]}_{best_pred_coordinates[1]}_{pixel_coordinate_3d[0, 0]}_{pixel_coordinate_3d[0, 1]}.png", matched_kpt_img)

        if best_H_pred is None:
            return {
                "pred_pixel_coordinate": best_pred_coordinates,
                "true_pixel_coordinate": pixel_coordinate_3d[0, :2].detach().cpu().numpy(),
                "true_gps_coordinate": gps_coordinate_3d[0, :2].detach().cpu().numpy(),
                "pred_yaw_angle": np.nan,
                "true_yaw": yaws
            }

        return {
            "pred_pixel_coordinate": best_pred_coordinates,
            "true_pixel_coordinate": pixel_coordinate_3d[0, :2].detach().cpu().numpy(),
            "true_gps_coordinate": gps_coordinate_3d[0, :2].detach().cpu().numpy(),
            "pred_yaw_angle": best_yaw_pred,
            "true_yaw": yaws
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
