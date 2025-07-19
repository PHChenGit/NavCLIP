import os
from os.path import exists
from pathlib import Path
import random
from enum import Enum

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image as IM
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import pytorch_lightning as pl

def drone_img_train_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop((size, size)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.3), scale=(0.5, 0.75)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet defaults
    ])

def satellite_img_train_transform(size=224):
    return transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.CenterCrop((size, size)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet defaults
    ])

def drone_img_val_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop((size, size)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.3), scale=(0.5, 0.75)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet defaults
    ])

def satellite_img_val_transform(size=224):
    return transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.CenterCrop((size, size)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet defaults
    ])

def drone_img_test_transform(size=224):
     # Usually test/predict also need normalization if the model expects it
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop((size, size)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.3), scale=(0.5, 0.75)),
        # transforms.PILToTensor(),
        # transforms.ConvertImageDtype(torch.float),
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
    ])

def satellite_img_test_transform(size=224):
     # Usually test/predict also need normalization if the model expects it
    return transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.CenterCrop((size, size)),
        # transforms.PILToTensor(),
        # transforms.ConvertImageDtype(torch.float),
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
    ])


class RandomDiscreteRotation:
    """
    Applies a random rotation from a discrete list of angles.
    Rotate image in counter-wise.
    """
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return angle, F.rotate(img, angle=angle)

class BaseDataLoader(Dataset):
    """
    Base DataLoader for image-gps datasets. Reads CSV for image paths and coordinates.

    Expected CSV columns: 'REF_IMG', 'QUERY_IMG', 'LAT', 'LON'.
    Subclasses should implement __getitem__.
    """
    def __init__(self, dataset_file: str, dataset_folder: str, sat_transform=None, drone_transform=None, size=224, is_cross_season: bool=False, is_testing: bool=False):
        if not exists(dataset_file):
             raise FileNotFoundError(f"Dataset CSV file not found: {dataset_file}")
        if not os.path.isdir(dataset_folder):
             raise NotADirectoryError(f"Dataset image folder not found: {dataset_folder}")

        self.dataset_folder = dataset_folder
        self.sat_transform = sat_transform
        self.drone_transform = drone_transform
        self.img_size = size
        self.is_cross_season = is_cross_season
        self.is_testing = is_testing

        if self.is_cross_season:
            self.ref_imgs, self.query_imgs, self.coordinates = self._load_dataset(dataset_file)
        else:
            self.ref_imgs, self.coordinates = self._load_dataset(dataset_file)

        self.rotate_angles = list(range(-180, 180, 15))
        self.rotation_transform = RandomDiscreteRotation(self.rotate_angles)

    def _load_dataset(self, dataset_file, required_cols: dict={'REF_IMG', 'QUERY_IMG', 'LAT', 'LON'}):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f">\tError reading {dataset_file}: {e}")

        if not required_cols.issubset(dataset_info.columns):
            raise ValueError(f"CSV file {dataset_file} must contain columns: {required_cols}")

        if self.is_testing:
            return self._load_testing_dataset(dataset_file, dataset_info)

        if self.is_cross_season is True:
            return self._load_cross_season_dataset(dataset_file, dataset_info)
        else:
            return self._load_single_season_dataset(dataset_file, dataset_info)

    def _load_single_season_dataset(self, dataset_file: str, dataset_info):
        ref_imgs = []
        coordinates = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            ref_img_path = os.path.join(self.dataset_folder, str(row['REF_IMG']))
            if exists(ref_img_path):
                ref_imgs.append(ref_img_path)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))

        if not ref_imgs:
             raise RuntimeError(f"No valid image pairs found based on CSV {dataset_file} and folder {self.dataset_folder}")

        return ref_imgs, coordinates

    def _load_cross_season_dataset(self, dataset_file: str, dataset_info):
        ref_imgs = []
        query_imgs = []
        coordinates = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            ref_img_path = os.path.join(self.dataset_folder, str(row['REF_IMG']))
            query_img_path = os.path.join(self.dataset_folder, str(row['QUERY_IMG']))
            if exists(ref_img_path) and exists(query_img_path):
                ref_imgs.append(ref_img_path)
                query_imgs.append(query_img_path)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))

        if not ref_imgs:
             raise RuntimeError(f"No valid image pairs found based on CSV {dataset_file} and folder {self.dataset_folder}")

        return ref_imgs, query_imgs, coordinates

    def _load_testing_dataset(self, dataset_file: str, dataset_info):
        query_imgs = []
        coordinates = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            query_img_path = os.path.join(self.dataset_folder, str(row['QUERY_IMG']))
            if exists(query_img_path):
                query_imgs.append(query_img_path)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))

        if not query_imgs:
             raise RuntimeError(f"No valid image pairs found based on CSV {dataset_file} and folder {self.dataset_folder}")

        return query_imgs, coordinates

    def __len__(self):
        return len(self.ref_imgs)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class PoseDataLoader(BaseDataLoader):
    """
    Loads ref/query image pair, applies random rotation to query, returns images, coords, angle.
    """
    def __init__(self, dataset_file, dataset_folder, sat_transform=None, drone_transform=None, size=224):
        super().__init__(dataset_file, dataset_folder, sat_transform, drone_transform, size)

    def __getitem__(self, idx):
        ref_img_path = self.ref_imgs[idx]
        coordinate = self.coordinates[idx]

        ref_img = IM.open(ref_img_path).convert('RGB')
        rotate_angle, query_img = self.rotation_transform(ref_img.copy())

        if self.sat_transform and self.drone_transform:
            ref_img = self.sat_transform(ref_img)
            query_img = self.drone_transform(query_img)
        else:
            ref_img = F.to_tensor(ref_img)
            query_img = F.to_tensor(query_img)

        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(rotate_angle)

        return ref_img, query_img, coordinate, yaw

class CrossSeasonPoseDataLoader(BaseDataLoader):
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        super().__init__(dataset_file, dataset_folder, transform, size=size, is_cross_season=True)

    def __getitem__(self, idx):
        ref_img_path = self.ref_imgs[idx]
        coordinate = self.coordinates[idx]

        ref_img = IM.open(ref_img_path).convert('RGB')
        query_img = IM.open(ref_img_path).convert('RGB')
        rotate_angle, query_img = self.rotation_transform(query_img)

        ref_img = transforms.Resize(self.img_size)(ref_img)
        # query_img = transforms.Resize(self.img_size)(query_img)
        cx, cy = query_img.size
        left = (cx - self.img_size) // 2
        top = (cy - self.img_size) // 2
        right = left + self.img_size
        bottom = top + self.img_size

        query_img = query_img.crop((left, top, right, bottom))

        if self.transform:
            ref_img = self.transform(ref_img)
            query_img = self.transform(query_img)
        else:
            ref_img = F.to_tensor(ref_img)
            query_img = F.to_tensor(query_img)

        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(rotate_angle)

        return ref_img, query_img, coordinate, yaw

class TestGEDataLoader(Dataset):
    """
    Loads ref/query image pair, applies random rotation to query, returns images, coords, angle.
    """
    def __init__(self, dataset_file, dataset_folder, sat_transform, drone_transform, size=224):
        self.dataset_folder = dataset_folder
        self.sat_transform = sat_transform
        self.drone_transform = drone_transform
        self.img_size = size

        self.ref_imgs, self.query_imgs, self.gps_coordinates, self.pixel_coordinates = self._load_dataset(dataset_file)

        self.rotate_angles = list(range(-180, 180, 15))
        self.rotation_transform = RandomDiscreteRotation(self.rotate_angles)

    def _load_dataset(self, dataset_file, required_cols: dict={'REF_IMG', 'QUERY_IMG', 'LAT', 'LON'}):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f">\tError reading {dataset_file}: {e}")

        if not required_cols.issubset(dataset_info.columns):
            raise ValueError(f"CSV file {dataset_file} must contain columns: {required_cols}")

        ref_imgs = []
        query_imgs = []
        gps_coordinates = []
        pixel_coordinates = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            ref_img_path = os.path.join(self.dataset_folder, str(row['REF_IMG']))
            query_img_path = os.path.join(self.dataset_folder, str(row['QUERY_IMG']))

            if not exists(ref_img_path):
                raise FileNotFoundError(ref_img_path)
            
            if not exists(query_img_path):
                raise FileNotFoundError(query_img_path)

            ref_imgs.append(ref_img_path)
            query_imgs.append(query_img_path)

            latitude = float(row['LAT'])
            longitude = float(row['LON'])
            gps_coordinates.append((latitude, longitude))

            x = float(row['PIXEL_X'])
            y = float(row['PIXEL_Y'])
            pixel_coordinates.append((x, y))

        return ref_imgs, query_imgs, gps_coordinates, pixel_coordinates

    def __len__(self):
        return len(self.ref_imgs)

    def __getitem__(self, idx):
        sat_img_path = self.ref_imgs[idx]
        query_img_path = Path(self.query_imgs[idx])
        gps_coordinates = self.gps_coordinates[idx]
        pixel_coordinates = self.pixel_coordinates[idx]

        sat_img = IM.open(sat_img_path).convert('RGB')
        query_img = IM.open(query_img_path).convert('RGB')
        
        yaws, query_img = self.rotation_transform(query_img)

        if self.sat_transform and self.drone_transform:
            sat_img = self.sat_transform(sat_img)
            query_img = self.drone_transform(query_img)

        else:
            sat_img = F.to_tensor(sat_img)
            query_img = F.to_tensor(query_img)

        gps_coordinate = torch.tensor(gps_coordinates, dtype=torch.float)
        pixel_coordinate = torch.tensor(pixel_coordinates, dtype=torch.float)
        yaws = torch.tensor(yaws)

        pattern = r"()"
        drone_img_filename = re.sub(pattern, "", query_img_path.stem)

        return sat_img, query_img, gps_coordinate, pixel_coordinate, yaws, drone_img_filename


class DJIPoseDataLoader(Dataset):
    """
    Loads ref/query image pair, applies random rotation to query, returns images, coords, angle.
    """
    def __init__(self, dataset_file, dataset_folder, sat_transform, drone_transform, size=224):
        self.dataset_folder = dataset_folder
        self.sat_transform = sat_transform
        self.drone_transform = drone_transform
        self.img_size = size

        self.rotate_angles = list(range(-180, 180, 15))
        self.rotation_transform = RandomDiscreteRotation(self.rotate_angles)

        self.ref_imgs, self.query_imgs, self.coordinates, self.pixel_coordinates = self._load_dataset(dataset_file)

    def _load_dataset(self, dataset_file, required_cols: dict={'REF_IMG', 'QUERY_IMG', 'LAT', 'LON'}):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f">\tError reading {dataset_file}: {e}")

        if not required_cols.issubset(dataset_info.columns):
            raise ValueError(f"CSV file {dataset_file} must contain columns: {required_cols}")

        ref_imgs = []
        query_imgs = []
        gps_coordinates = []
        pixel_coordinates = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            ref_img_path = os.path.join(self.dataset_folder, str(row['REF_IMG']))
            query_img_path = os.path.join(self.dataset_folder, str(row['QUERY_IMG']))

            if not exists(ref_img_path):
                raise FileNotFoundError(ref_img_path)
            
            if not exists(query_img_path):
                raise FileNotFoundError(query_img_path)

            ref_imgs.append(ref_img_path)
            query_imgs.append(query_img_path)
            latitude = float(row['LAT'])
            longitude = float(row['LON'])
            gps_coordinates.append((latitude, longitude))

            x = float(row['PIXEL_X'])
            y = float(row['PIXEL_Y'])
            pixel_coordinates.append((x, y))

        
        return np.array(ref_imgs), np.array(query_imgs), np.array(gps_coordinates), np.array(pixel_coordinates)
    
    def __len__(self):
        return len(self.ref_imgs)
    
    def __getitem__(self, idx):
        ref_img_path = self.ref_imgs[idx]
        query_img_path = self.query_imgs[idx]
        gps_coordinate = self.coordinates[idx]
        pixel_coordinate = self.pixel_coordinates[idx]

        ref_img = IM.open(ref_img_path).convert('RGB')
        query_img = IM.open(query_img_path).convert('RGB')

        yaw, query_img = self.rotation_transform(query_img.copy())

        if self.sat_transform and self.drone_transform:
            ref_img = self.sat_transform(ref_img)
            query_img = self.drone_transform(query_img)
        else:
            ref_img = F.to_tensor(ref_img)
            query_img = F.to_tensor(query_img)

        gps_coordinate = torch.tensor(gps_coordinate, dtype=torch.float32)
        pixel_coordinate = torch.tensor(pixel_coordinate, dtype=torch.float32)
        yaw = torch.tensor(yaw, dtype=torch.float32)

        return ref_img, query_img, gps_coordinate, pixel_coordinate, yaw

class TestDJIeDataLoader(Dataset):
    """
    Loads ref/query image pair, applies random rotation to query, returns images, coords, angle.
    """
    def __init__(self, dataset_file, dataset_folder, sat_transform, drone_transform, size=224):
        self.dataset_folder = dataset_folder
        self.sat_transform = sat_transform
        self.drone_transform = drone_transform
        self.img_size = size

        self.ref_imgs, self.query_imgs, self.gps_coordinates, self.pixel_coordinates = self._load_dataset(dataset_file)

    def _load_dataset(self, dataset_file, required_cols: dict={'REF_IMG', 'QUERY_IMG', 'LAT', 'LON', 'YAW'}):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f">\tError reading {dataset_file}: {e}")

        if not required_cols.issubset(dataset_info.columns):
            raise ValueError(f"CSV file {dataset_file} must contain columns: {required_cols}")

        ref_imgs = []
        query_imgs = []
        gps_coordinates = []
        pixel_coordinates = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            ref_img_path = os.path.join(self.dataset_folder, str(row['REF_IMG']))
            query_img_path = os.path.join(self.dataset_folder, str(row['QUERY_IMG']))

            if not exists(ref_img_path):
                raise FileNotFoundError(ref_img_path)
            
            if not exists(query_img_path):
                raise FileNotFoundError(query_img_path)

            ref_imgs.append(ref_img_path)
            query_imgs.append(query_img_path)

            yaw = float(row['YAW'])

            latitude = float(row['LAT'])
            longitude = float(row['LON'])
            gps_coordinates.append((latitude, longitude, yaw))

            x = float(row['PIXEL_X'])
            y = float(row['PIXEL_Y'])
            pixel_coordinates.append((x, y, yaw))

        return ref_imgs, query_imgs, gps_coordinates, pixel_coordinates, 

    def __len__(self):
        return len(self.ref_imgs)

    def __getitem__(self, idx):
        sat_img_path = self.ref_imgs[idx]
        query_img_path = Path(self.query_imgs[idx])
        gps_coordinates = self.gps_coordinates[idx]
        pixel_coordinates = self.pixel_coordinates[idx]

        sat_img = IM.open(sat_img_path).convert('RGB')
        query_img = IM.open(query_img_path).convert('RGB')
        query_img_np = np.array(query_img)

        if self.sat_transform and self.drone_transform:
            sat_img = self.sat_transform(sat_img)
            query_img = self.drone_transform(query_img)
        else:
            sat_img = F.to_tensor(sat_img)
            query_img = F.to_tensor(query_img)

        gps_coordinate_3d = torch.tensor(gps_coordinates, dtype=torch.float)
        pixel_coordinate_3d = torch.tensor(pixel_coordinates, dtype=torch.float)
        pattern = r"()"
        drone_img_filename = re.sub(pattern, "", query_img_path.stem)

        return sat_img, query_img, gps_coordinate_3d, pixel_coordinate_3d, drone_img_filename

class DataLoaderTypesEnum(Enum):
    Pose = 'Pose'
    CrossSeasonPose = 'CrossSeasonPose'
    WarpingOmniGlue = 'WarpingOmniGlue'
    WarpingLightGlue = 'WarpingLightGlue'
    WarpingSIFT = 'WarpingSIFT'
    TestGE = 'TestGEDataLoader'
    TestDJI = 'TestDJIeDataLoader'
    DJIPose = 'DJIPose'

class GeoCLIPDataModule(pl.LightningDataModule):
    def __init__(self,
                dataset_folder: str,
                train_csv: str = None,
                val_csv: str = None,
                predict_csv: str = None,
                test_csv: str = None,
                dataset_type: DataLoaderTypesEnum = DataLoaderTypesEnum.Pose,
                image_size: int = 224,
                batch_size: int = 32,
                num_workers: int = 4,
                is_cross_season: bool = False):
        """
        PyTorch Lightning DataModule for GeoCLIP datasets.

        Args:
            train_csv: Path to the training CSV file.
            val_csv: Path to the validation CSV file.
            test_csv: Path to the test CSV file.
            predict_csv: Path to the prediction CSV file (if different from test).
            dataset_folder: Path to the folder containing all images.
            dataset_type: Type of dataset to use for training/validation.
                          Options: 'Pose', 'PoseV2', 'WarpingOmniGlue',
                                   'WarpingLightGlue', 'WarpingSIFT'.
            image_size: Target size for image resizing in transforms.
            batch_size: The batch size for the dataloaders.
            num_workers: The number of workers for the dataloaders.
        """
        super().__init__()
        if not isinstance(dataset_type, DataLoaderTypesEnum):
            raise ValueError(
                f"Invalid dataset_type '{dataset_type}'. Must be one of {DataLoaderTypesEnum._member_names_}"
            )

        self.save_hyperparameters()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.satellite_train_transform = satellite_img_train_transform(size=self.hparams.image_size)
        self.drone_train_transform = drone_img_train_transform(size=self.hparams.image_size)
        self.satellite_val_transform = satellite_img_val_transform(size=self.hparams.image_size)
        self.drone_val_transform = drone_img_val_transform(size=self.hparams.image_size)
        self.satellite_test_transform = satellite_img_test_transform(size=self.hparams.image_size)
        self.drone_test_transform = drone_img_test_transform(size=self.hparams.image_size)

    def setup(self, stage: str = None):
        """Create the datasets based on the stage."""

        if self.hparams.dataset_type is DataLoaderTypesEnum.Pose:
            TrainValPredDatasetClass = PoseDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.CrossSeasonPose:
            TrainValPredDatasetClass = CrossSeasonPoseDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.TestGE:
            PredDatasetClass = TestGEDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.TestDJI:
            PredDatasetClass = TestDJIeDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.DJIPose:
            TrainValPredDatasetClass = DJIPoseDataLoader
        else:
            raise ValueError(f"Unknown dataset_type: {self.hparams.dataset_type}")

        if stage == 'fit':
            print(f"Setting up 'fit' stage with dataset type: {self.hparams.dataset_type}")
            if self.hparams.dataset_type is DataLoaderTypesEnum.DJIPose:
                self.train_dataset = TrainValPredDatasetClass(
                    dataset_file=self.hparams.train_csv,
                    dataset_folder=self.hparams.dataset_folder,
                    sat_transform=self.satellite_train_transform,
                    drone_transform=self.drone_train_transform,
                    size=self.hparams.image_size
                )
                # Use the same dataset class type for validation, but with validation transform
                self.val_dataset = TrainValPredDatasetClass(
                    dataset_file=self.hparams.val_csv,
                    dataset_folder=self.hparams.dataset_folder,
                    sat_transform=self.satellite_val_transform,
                    drone_transform=self.drone_val_transform,
                    size=self.hparams.image_size
                )
            else:
                self.train_dataset = TrainValPredDatasetClass(
                    dataset_file=self.hparams.train_csv,
                    dataset_folder=self.hparams.dataset_folder,
                    sat_transform=self.satellite_val_transform,
                    drone_transform=self.drone_val_transform,
                    size=self.hparams.image_size
                )
                # Use the same dataset class type for validation, but with validation transform
                self.val_dataset = TrainValPredDatasetClass(
                    dataset_file=self.hparams.val_csv,
                    dataset_folder=self.hparams.dataset_folder,
                    sat_transform=self.satellite_val_transform,
                    drone_transform=self.drone_val_transform,
                    size=self.hparams.image_size
                )
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage == 'validate':
            self.val_dataset = TrainValPredDatasetClass(
                dataset_file=self.hparams.val_csv,
                dataset_folder=self.hparams.dataset_folder,
                transform=self.val_transform,
                size=self.hparams.image_size
            )
            print(f"Validation dataset size: {len(self.val_dataset)}")
            
        # if stage == 'test' or stage is None:
        #     print(f"Setting up 'test' stage.")
        #     self.test_dataset = TestDatasetClass(
        #         dataset_file=self.hparams.test_csv,
        #         dataset_folder=self.hparams.dataset_folder,
        #         transform=self.test_transform
        #     print(f"Test dataset size: {len(self.test_dataset)}")

        if stage == 'predict' or stage is None:
            # Use predict_csv if provided, otherwise fall back to test_csv
            predict_csv_path = self.hparams.predict_csv if self.hparams.predict_csv else self.hparams.test_csv
            print(f">\tSetting up 'predict' stage using CSV: {predict_csv_path}")
            self.predict_dataset = PredDatasetClass(
                dataset_file=predict_csv_path,
                dataset_folder=self.hparams.dataset_folder,
                sat_transform=self.satellite_test_transform,
                drone_transform=self.drone_test_transform,
                size=self.hparams.image_size
            )
            print(f">\tPredict dataset size: {len(self.predict_dataset)}")


    def train_dataloader(self):
        if not self.train_dataset:
            raise RuntimeError("Train dataset not initialized. Run setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False, # Avoid worker restart overhead
            drop_last=True # Often beneficial for training stability
        )

    def val_dataloader(self):
        if not self.val_dataset:
            raise RuntimeError("Validation dataset not initialized. Run setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            drop_last=True
        )

    def predict_dataloader(self):
        if not self.predict_dataset:
            print("Predict dataset not available.")
            # Option 1: Raise error
            raise RuntimeError("Predict dataset not initialized. Run setup('predict') or provide predict_csv.")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            drop_last=True
        )

