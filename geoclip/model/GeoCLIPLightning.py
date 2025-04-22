from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl
from transformers import CLIPModel, AutoProcessor

from geoclip.model.image_encoder import ImageEncoder
from geoclip.model.location_encoder import LocationEncoder
from .misc import load_gallery_data, log_pred_result

class GeoCLIPLightning(pl.LightningModule):
    def __init__(self,
                 gallery_path: str,
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 queue_size: int = 4096,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
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

    def load_weights(self, pretrained_dir):
        """Loads weights to the correct device"""
        device = self.device
        print(f">\tload_weights pretrained_dir: {pretrained_dir}")
        
        image_encoder_path = Path(pretrained_dir) / "image_encoder.pth"
        if image_encoder_path.exists():
            weights = torch.load(image_encoder_path, map_location=device, weights_only=True)
            self.image_encoder.load_state_dict(weights)

        location_encoder_path = Path(pretrained_dir) / "location_encoder.pth"
        if location_encoder_path.exists():
            weights = torch.load(location_encoder_path, map_location=device, weights_only=True)
            self.location_encoder.load_state_dict(weights)
            
        logit_scale_path = Path(pretrained_dir) / "logit_scale.pth" 
        if logit_scale_path.exists():
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
        ref_imgs, query_imgs, coordinates, yaws = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image = self(query_imgs, self.gps_gallery) # [B, num_gallery_gps]

        probs_per_image = logits_per_image.softmax(dim=-1).cpu() # [B, num_gallery_gps]

        top_k = 1
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        pred_coordinate = self.gps_gallery[top_pred.indices[0]]
        # self.pred_coordinate_list.append(pred_coordinate)
        # self.true_coordinate_list.append(coordinates.cpu().numpy())
        # pred_dist_mae, pred_dist_rmse = self._common_val_test_loss(pred_coordinate, coordinates)

        # return {"pred_dist_mae_pixel": pred_dist_mae, "pred_dist_rmse_pixel": pred_dist_rmse}
        return {
            "pred_coordinate": pred_coordinate.detach().cpu().numpy(),
            "true_coordinate": coordinates.cpu().numpy()
        }

    # def on_predict_end(self) -> None:
    #     pred_dist_mae, pred_dist_rmse = self._common_val_test_loss(torch.Tensor(np.array(self.pred_coordinate_list)), torch.Tensor(np.array(self.true_coordinate_list)))
    #     data = {
    #         "Pred Dist MAE(pixel)": pred_dist_mae,
    #         "Pred Dist RMSE(pixel)": pred_dist_rmse,
    #     }
    #     log_pred_result(data, "/home/rvl1421/Documents/hsun/NavCLIP/output/", "pred_NTUT_playground.json")
    #
    #     print(f"pred dist mae: {pred_dist_mae}")
    #     print(f"pred dist RMSE: {pred_dist_rmse}")
    #     return super().on_predict_end()

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
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        return [optimizer], [scheduler]
