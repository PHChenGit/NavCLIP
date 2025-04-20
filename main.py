from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet18 # 範例中使用 ResNet18

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataloader import GeoCLIPDataModule
from geoclip.model.GeoCLIPLightning import GeoCLIPLightning

# --- 2. PyTorch Lightning 模型 (LightningModule) ---
class CoordinatePredictor(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # --- 模型架構範例 ---
        # 使用預訓練的 ResNet18 作為特徵提取器
        # 移除最後的全連接層
        self.ref_feature_extractor = resnet18(pretrained=True)
        self.ref_feature_extractor.fc = nn.Identity() # 移除分類層

        self.query_feature_extractor = resnet18(pretrained=True)
        self.query_feature_extractor.fc = nn.Identity() # 移除分類層

        # 假設 ResNet18 移除 fc 層後輸出 512 維特徵
        feature_dim = 512
        # 將兩個圖像的特徵拼接起來
        combined_feature_dim = feature_dim * 2

        # 添加一些全連接層來預測座標 (LAT, LON)
        self.regressor = nn.Sequential(
            nn.Linear(combined_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 輸出 2 個值：LAT, LON
        )
        # --- 模型架構範例結束 ---

        self.criterion = nn.MSELoss() # 使用均方誤差作為損失函數 (適用於回歸)

    def forward(self, ref_img, query_img):
        ref_features = self.ref_feature_extractor(ref_img)
        query_features = self.query_feature_extractor(query_img)

        # 拼接特徵
        combined_features = torch.cat((ref_features, query_features), dim=1)

        # 預測座標
        predicted_coords = self.regressor(combined_features)
        return predicted_coords

    def _common_step(self, batch, batch_idx):
        ref_img, query_img, coords = batch
        predicted_coords = self(ref_img, query_img)
        loss = self.criterion(predicted_coords, coords)
        return loss, predicted_coords, coords

    def training_step(self, batch, batch_idx):
        # 處理可能的 None batch (來自 Dataset 的錯誤處理)
        if batch is None or any(item is None for item in batch):
             print(f"警告：跳過一個損壞的 batch (訓練)")
             return None # 或者返回一個 0 loss 的 tensor

        loss, _, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None or any(item is None for item in batch):
             print(f"警告：跳過一個損壞的 batch (驗證)")
             return None

        loss, predicted_coords, coords = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # 可以計算其他指標，例如 MAE (Mean Absolute Error)
        val_mae = torch.abs(predicted_coords - coords).mean()
        self.log('val_mae', val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # validation_epoch_end (可選): 如果你想在每個 fold 的驗證結束時計算整體指標
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_mae = torch.stack([x['val_mae'] for x in outputs]).mean() # 假設你在 step 記錄了 mae
    #     self.log('val_loss_epoch', avg_loss)
    #     self.log('val_mae_epoch', avg_mae)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # 可以添加學習率調度器 (scheduler)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return optimizer


# --- 4. 主訓練迴圈 ---
if __name__ == '__main__':
    # --- 設定參數 ---
    DATASET_ROOT_PATH = r'/home/hsun/Documents/hsun/datasets/NTU_playground_100k'
    CSV_FILE = 'taipei.csv'
    EPOCHS = 6
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 8
    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    DEVICES = 1 if torch.cuda.is_available() else None

    DATASET_ROOT = Path(DATASET_ROOT_PATH)
    TRAIN_CSV = DATASET_ROOT.joinpath('train', CSV_FILE)
    VAL_CSV = DATASET_ROOT.joinpath('val', CSV_FILE)
    PRED_CSV = DATASET_ROOT.joinpath('test', CSV_FILE)
    COORDINATE_GALLERY = DATASET_ROOT.joinpath('train', 'gallery.csv')

    # 實例化 DataModule
    datamodule = GeoCLIPDataModule(
        TRAIN_CSV,
        VAL_CSV,
        PRED_CSV,
        dataset_folder=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    datamodule.setup()

    checkpoint_callback = ModelCheckpoint(
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_dist_rmse', # 監控驗證損失
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_dist_rmse',
        patience=5, # 連續 5 個 Epoch 沒有改善就停止
        verbose=True,
        mode='min'
    )
    model = GeoCLIPLightning(from_pretrained=False, gallery_path=COORDINATE_GALLERY, learning_rate=LEARNING_RATE)

    trainer = Trainer(
        default_root_dir='ouput',
        max_epochs=EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger(save_dir="logs"),
        log_every_n_steps=50,
        # deterministic=True, # 如果需要可重現性，但可能影響效能
    )

    trainer.fit(model, datamodule=datamodule)

    print("訓練完成！")