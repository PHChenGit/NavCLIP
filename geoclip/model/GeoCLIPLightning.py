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
from .misc import load_gallery_data

class GeoCLIPLightning(pl.LightningModule):
    def __init__(self,
                 gallery_path: str,
                 clip_model_name: str = "openai/clip-vit-large-patch14",
                 from_pretrained: bool = True,
                 queue_size: int = 4096,
                 weights: dict = None,
                 learning_rate: float = 1e-4, # 添加學習率超參數
                 weight_decay: float = 0.01,  # 添加權重衰減超參數
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.CLIP = CLIPModel.from_pretrained(clip_model_name)
        # self.image_processor = AutoProcessor.from_pretrained(clip_model_name)
        self.image_encoder = ImageEncoder()

        weight_folder = Path(__file__).resolve().parent / "weights"

        # default_image_encoder_weight_path = weight_folder / "image_encoder_mlp_weights.pth"
        # default_location_encoder_weight_path = weight_folder / "location_encoder_weights.pth"
        # default_logit_scale_weight_path = weight_folder / "logit_scale_weights.pth"

        if weights is not None:
            self.image_encoder_weight_path = weight_folder / weights.get('image_encoder')
            self.location_encoder_weight_path = weight_folder / weights.get('location_encoder')
            self.logit_scale_weight_path = weight_folder / weights.get('logit_scale')
            self.location_encoder = LocationEncoder(from_pretrained=True, pretrained_path=self.location_encoder_weight_path)
        else:
            self.location_encoder = LocationEncoder(from_pretrained=False)

        self.gps_gallery: torch.Tensor = load_gallery_data(gallery_path)
        self._initialize_gps_queue(queue_size)
        self.should_load_weights = from_pretrained

        for param in self.CLIP.parameters():
            param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()

    def setup(self, stage=None):
        """Called by Lightning after model is moved to device"""
        if self.should_load_weights:
            print(f">\t正在載入預訓練權重...")
            if hasattr(self, 'image_encoder_weight_path'):
                print(f">\t圖像編碼器權重: {self.image_encoder_weight_path}")
            if hasattr(self, 'location_encoder_weight_path'):    
                print(f">\t位置編碼器權重: {self.location_encoder_weight_path}")
            if hasattr(self, 'logit_scale_weight_path'):
                print(f">\tLogit Scale 權重: {self.logit_scale_weight_path}")
            self._load_weights()

    def _load_weights(self):
        """Loads weights to the correct device"""
        device = self.device
        
        if hasattr(self, 'image_encoder_weight_path') and Path(self.image_encoder_weight_path).exists():
            weights = torch.load(self.image_encoder_weight_path, map_location=device, weights_only=True)
            self.image_encoder.load_state_dict(weights)
            
        if hasattr(self, 'location_encoder_weight_path') and Path(self.location_encoder_weight_path).exists():
            weights = torch.load(self.location_encoder_weight_path, map_location=device, weights_only=True)
            self.location_encoder.load_state_dict(weights)
            
        if hasattr(self, 'logit_scale_weight_path') and Path(self.logit_scale_weight_path).exists():
            weights = torch.load(self.logit_scale_weight_path, map_location=device, weights_only=True)
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

        top_pred = torch.topk(probs, 1, dim=1)
        top_pred_coordinates = self.gps_gallery[top_pred.indices[0]]
        val_dist_mae, val_dist_rmse = self._common_val_test_loss(top_pred_coordinates, coordinates)

        self.log('Valdiation Dist MAE(pixel)', val_dist_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_dist_rmse', val_dist_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # 可以額外計算和記錄其他指標，例如 Top-k accuracy
        # topk_acc = calculate_topk_accuracy(logits_per_image, labels, k=1) # 假設有此函數
        # self.log('val_acc_top1', topk_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return val_dist_rmse

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ref_imgs, query_imgs, coordinates, yaws = batch

        if self.gps_gallery.device != self.device:
            self.gps_gallery = self.gps_gallery.to(self.device)

        with torch.autocast(device_type='cuda'):
            logits_per_image = self(query_imgs, self.gps_gallery) # [B, num_gallery_gps]

        probs_per_image = logits_per_image.softmax(dim=-1).cpu() # [B, num_gallery_gps]

        top_k = 1
        top_pred_prob, top_pred_indices = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps_list = []
        for i in range(top_pred_indices.shape[0]):
            indices_for_image_i = top_pred_indices[i] # [k]
            gps_for_image_i = self.gps_gallery[indices_for_image_i] # [k, 2]
            top_pred_gps_list.append(gps_for_image_i)

        # 如果需要，可以將列表堆疊成張量
        top_pred_gps = torch.stack(top_pred_gps_list) # [B, k, 2]

        # 返回預測結果，可以是一個字典
        return {"top_k_gps": top_pred_gps, "top_k_probs": top_pred_prob}


    def configure_optimizers(self):
        # params_to_optimize = [
        #     {'params': self.image_encoder.parameters(), "lr": 3e-4},
        #     {'params': self.location_encoder.parameters(), "lr": 3e-4},
        #     {'params': [self.logit_scale]}
        # ]
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