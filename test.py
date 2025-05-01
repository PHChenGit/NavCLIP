import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything

from dataloader import GeoCLIPDataModule
from models.geoclip.GeoCLIPLightning import GeoCLIPLightning
from models.geoclip.misc import log_pred_result


def main(args):
    seed_everything(42, workers=True)
    ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
    CSV_FILE = args.dataset_file
    DATASET_ROOT_PATH = args.ds_folder

    DATASET_ROOT = Path(DATASET_ROOT_PATH)
    TRAIN_CSV = DATASET_ROOT.joinpath('train', CSV_FILE)
    VAL_CSV = DATASET_ROOT.joinpath('val', CSV_FILE)
    PRED_CSV = DATASET_ROOT.joinpath('test', CSV_FILE)
    COORDINATE_GALLERY = DATASET_ROOT.joinpath('test', args.dataset_file)
    print(f">\tCoordinate gallery path: {COORDINATE_GALLERY}")
    datamodule = GeoCLIPDataModule(
        str(TRAIN_CSV),
        str(VAL_CSV),
        str(PRED_CSV),
        dataset_folder=str(DATASET_ROOT),
        batch_size=args.bs,
        num_workers=args.num_workers,
    )
    datamodule.setup()

    model = GeoCLIPLightning(gallery_path=str(COORDINATE_GALLERY))

    if Path(args.pretrained_model_dir).exists():
        model.load_weights(args.pretrained_model_dir)
    else:
        raise FileNotFoundError(f"Pretrained model dir not found: {args.pretrained_model_dir}")
    trainer = Trainer(
        default_root_dir='output',
        accelerator=ACCELERATOR,
    )
    pred_result: List[Dict] = trainer.predict(model, datamodule=datamodule)
    # print(pred_result)
    pred_coordinate_list = []
    true_coordinate_list = []
    for data in pred_result:
        pred_coordinate_list.append(data["pred_coordinate"])
        true_coordinate_list.append(data["true_coordinate"])
    pred_dist_mae_pixel, pred_dist_rmse_pixel = model._common_val_test_loss(torch.Tensor(np.array(pred_coordinate_list)), torch.Tensor(np.array(true_coordinate_list)))
    pred_dist_mae_pixel = round(pred_dist_mae_pixel.detach().item(), 2)
    pred_dist_rmse_pixel = round(pred_dist_rmse_pixel.detach().item(), 2)
    data = {
        "Pred Dist MAE(pixel)": pred_dist_mae_pixel,
        "Pred Dist RMSE(pixel)": pred_dist_rmse_pixel,
    }
    print(data)
    exit()
    log_pred_result(data, "/home/rvl1421/Documents/hsun/NavCLIP/output/", "pred_NTUT_playground.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference GeoCLIP")
    parser.add_argument("--name", type=str, default="GeoCLIP", help="experiment name")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--queue_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pretrained_model_dir", type=str, default="/home/rvl1421/Documents/hsun/NavCLIP/output/checkpoints/", help="pretrained model folder")
    parser.add_argument("--output_dir", type=str, help="Prediction result folder")
    parser.add_argument("--ds_folder", type=str, default=r"~/Documents/hsun/datasets/NTU_playground_Cross_Season_100k", help="dataset folder path")
    parser.add_argument("--dataset_file", type=str, default=r"taipei.csv", help="dataset csv file")
    parser.add_argument("--sat_img", type=str, default=r"~/Documents/hsun/datasets/satellites/202410_NTU_playground.jpg", help="dataset folder path")
    args = parser.parse_args()
    main(args)

