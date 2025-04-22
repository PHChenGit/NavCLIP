import argparse
from pathlib import Path

import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataloader import GeoCLIPDataModule
from geoclip.model.GeoCLIPLightning import GeoCLIPLightning
from mymodelckpt import MyModelCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train NavCLIP")
    parser.add_argument("--name", type=str, default="NavCLIP", help="experiment name")
    parser.add_argument("--bs", type=int, default=512, help="batch size")
    parser.add_argument("--epochs", type=int, default=500, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--queue_size", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default='output', help="checkpoint folder")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ds_folder", type=str, default=r"~/Documents/hsun/datasets/NTU_playground_Cross_Season_100k", help="dataset folder path")
    parser.add_argument("--dataset_file", type=str, default=r"taipei.csv", help="dataset csv file")
    args = parser.parse_args()

    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    DEVICES = 1 if torch.cuda.is_available() else None
    CSV_FILE = args.dataset_file
    DATASET_ROOT_PATH = args.ds_folder

    DATASET_ROOT = Path(DATASET_ROOT_PATH)
    TRAIN_CSV = DATASET_ROOT.joinpath('train', CSV_FILE)
    VAL_CSV = DATASET_ROOT.joinpath('val', CSV_FILE)
    PRED_CSV = DATASET_ROOT.joinpath('test', CSV_FILE)
    COORDINATE_GALLERY = DATASET_ROOT.joinpath('train', 'gallery.csv')

    datamodule = GeoCLIPDataModule(
        str(TRAIN_CSV),
        str(VAL_CSV),
        str(PRED_CSV),
        dataset_folder=str(DATASET_ROOT),
        batch_size=args.bs,
        num_workers=args.num_workers,
    )

    datamodule.setup()

    checkpoint_callback = MyModelCheckpoint(
        dirpath=f"{args.output_dir}/checkpoints",
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_dist_MAE', # 監控驗證損失
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_dist_MAE',
        patience=20, # 連續 n 個 Epoch 沒有改善就停止
        verbose=True,
        mode='min'
    )
    model = GeoCLIPLightning(gallery_path=str(COORDINATE_GALLERY), learning_rate=args.lr)

    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="logs", name=args.name)

    trainer = Trainer(
        default_root_dir='output',
        max_epochs=args.epochs,
        accelerator=ACCELERATOR,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)

    print("訓練完成！")
