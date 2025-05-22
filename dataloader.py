import os
from os.path import exists
import random
from enum import Enum

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image as IM

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F

import pytorch_lightning as pl

def img_train_transform(size=224):
    return transforms.Compose([
        # transforms.Resize(size),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet defaults
    ])

def img_val_transform(size=224):
    return transforms.Compose([
        # transforms.Resize(size),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet defaults
    ])

def img_test_transform(size=224):
     # Usually test/predict also need normalization if the model expects it
    return transforms.Compose([
        # transforms.Resize(size), # Ensure consistent input size
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # CLIP Defaults
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet defaults
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
    def __init__(self, dataset_file: str, dataset_folder: str, transform=None, size=224, is_cross_season: bool=False, is_testing: bool=False):
        if not exists(dataset_file):
             raise FileNotFoundError(f"Dataset CSV file not found: {dataset_file}")
        if not os.path.isdir(dataset_folder):
             raise NotADirectoryError(f"Dataset image folder not found: {dataset_folder}")

        self.dataset_folder = dataset_folder
        self.transform = transform
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
            query_img_path = os.path.join(self.dataset_folder, str(row['REF_IMG']))
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
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        super().__init__(dataset_file, dataset_folder, transform, size)

    def __getitem__(self, idx):
        ref_img_path = self.ref_imgs[idx]
        coordinate = self.coordinates[idx]

        ref_img = IM.open(ref_img_path).convert('RGB')
        rotate_angle, query_img = self.rotation_transform(ref_img.copy())

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

class PoseDataLoader_Warping(BaseDataLoader):
    """
    Base class for datasets applying Yaw, Roll, Pitch warping using OpenCV.
    Subclasses must implement __getitem__.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        super().__init__(dataset_file, dataset_folder, transform)

    def _apply_warp(self, img, M, border_mode=cv2.BORDER_CONSTANT, border_value=(0,0,0)):
        """Applies affine warp using OpenCV."""
        rows, cols = img.shape[:2]
        # Ensure M is float32 for cv2.warpAffine
        M = np.float32(M)
        # Use BORDER_CONSTANT or other modes as needed
        warped_img = cv2.warpAffine(img, M, (cols, rows), borderMode=border_mode, borderValue=border_value)
        # Convert 2x3 affine matrix M to 3x3 homography matrix H
        H = np.vstack([M, [0, 0, 1]])
        return warped_img, H

    def apply_yaw(self, img, yaw_angle):
        """Applies yaw rotation (around vertical axis). Positive = clockwise."""
        # Normalize yaw_angle to [-180, 180] for rotation function if needed
        # M = cv2.getRotationMatrix2D expects angle in degrees, counter-clockwise positive.
        # If yaw_angle is clockwise positive, use -yaw_angle.
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle=-yaw_angle, scale=1)
        rotated_img, H_yaw = self._apply_warp(img, M)
        return rotated_img, yaw_angle, H_yaw # Return original yaw angle

    def apply_roll(self, img, roll_angle):
        """Applies roll (rotation around longitudinal axis -> shear vertically)."""
        rows, cols = img.shape[:2]
        shear_factor = np.tan(np.deg2rad(roll_angle)) # Shear factor based on angle
        # Roll affects vertical shear (shifts y based on x)
        M = np.array([[1, 0, 0],
                      [shear_factor, 1, 0]], dtype=np.float32) # Corrected matrix for roll simulation via shear
        rolled_img, H_roll = self._apply_warp(img, M)
        return rolled_img, roll_angle, H_roll

    def apply_pitch(self, img, pitch_angle):
        """Applies pitch (rotation around lateral axis -> shear horizontally)."""
        rows, cols = img.shape[:2]
        shear_factor = np.tan(np.deg2rad(pitch_angle)) # Shear factor based on angle
        # Pitch affects horizontal shear (shifts x based on y)
        M = np.array([[1, shear_factor, 0], # Corrected matrix for pitch simulation via shear
                      [0, 1, 0]], dtype=np.float32)
        pitched_img, H_pitch = self._apply_warp(img, M)
        return pitched_img, pitch_angle, H_pitch


    def generate_query_img(self, ref_img_np):
        """Applies random yaw, roll, pitch warps to a numpy image."""
        # yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        yaw_angles = list(range(0, 360, 45)) # Simpler range
        roll_angles = [-30, -15, 0, 15, 30]
        pitch_angles = [-30, -15, 0, 15, 30]

        # Randomly select angles
        yaw_angle = np.random.choice(yaw_angles)
        roll_angle = np.random.choice(roll_angles)
        pitch_angle = np.random.choice(pitch_angles)

        # Apply warps sequentially (order might matter)
        img_yaw, yaw, H_yaw = self.apply_yaw(ref_img_np, yaw_angle)
        img_roll, roll, H_roll = self.apply_roll(img_yaw, roll_angle)
        query_img_np, pitch, H_pitch = self.apply_pitch(img_roll, pitch_angle)

        # Calculate combined homography (order: Yaw -> Roll -> Pitch)
        # H = H_pitch @ H_roll @ H_yaw # Matrix multiplication order
        # Note: Original code used np.dot which is equivalent for 2D arrays here

        # Return warped image and individual angles
        # We don't return H here as it wasn't returned by the subclasses' __getitem__
        return query_img_np, yaw, roll, pitch

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__")


class PoseDataLoader_Warping_OmniGlue(PoseDataLoader_Warping):
    """ Warping dataloader variant possibly for OmniGlue. """
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        super().__init__(dataset_file, dataset_folder, transform)
        self.img_size = (size, size)
        # Note: This specific transform seems unused in __getitem__ based on original code
        self.rot_transforms = transforms.Compose([
            # transforms.Resize(size), # Resizing done explicitly below
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

    def __getitem__(self, index):
        ref_img_path = self.ref_imgs[index]
        # query_img_path = self.query_imgs[index] # Original query path not used to generate warp
        coordinate = self.coordinates[index]

        try:
            # Load reference image as PIL for main transform
            ref_img_pil = IM.open(ref_img_path).convert('RGB')
            # Load reference image as numpy for warping
            ref_img_np = np.array(ref_img_pil)
        except Exception as e:
            print(f"Error opening image file: {ref_img_path}. Skipping index {index}. Error: {e}")
            return self.__getitem__((index + 1) % len(self))

        # Generate warped query image (numpy) from reference (numpy)
        query_img_np, yaw, roll, pitch = self.generate_query_img(ref_img_np.copy())

        # Convert original ref PIL and warped query numpy back to PIL for transforms
        # query_img_pil = IM.fromarray(query_img_np) # Not needed for main transform pipeline

        # Apply main transform (incl. resize, normalization) to the *original* ref PIL image
        if self.transform:
            ref_img_transformed = self.transform(ref_img_pil)
            # Apply the *same* transform to a PIL version of the *original* ref_img
            # This seems odd - why transform the original image twice?
            # Assuming the goal was to transform the original ref and the *warped* query
            # Let's transform the warped query instead of the original query_img_path content
            query_img_transformed = self.transform(IM.fromarray(query_img_np)) # Transform the WARPED query
        else:
            # Manual transform if needed
            ref_img_transformed = F.to_tensor(transforms.Resize(self.img_size[0])(ref_img_pil))
            query_img_transformed = F.to_tensor(transforms.Resize(self.img_size[0])(IM.fromarray(query_img_np)))


        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(yaw, dtype=torch.float)
        roll = torch.tensor(roll, dtype=torch.float)
        pitch = torch.tensor(pitch, dtype=torch.float)

        # Return transformed original ref, transformed warped query,
        # and the numpy versions *before* resize/norm (as per original return signature)
        return ref_img_transformed, query_img_transformed, ref_img_np, query_img_np, coordinate, roll, yaw, pitch


class PoseDataLoader_Warping_LightGlue(PoseDataLoader_Warping):
    """ Warping dataloader variant possibly for LightGlue (uses grayscale). """
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        super().__init__(dataset_file, dataset_folder, transform)
        self.img_size = (size, size)
        # Grayscale transform for feature matching part
        self.rot_transforms = transforms.Compose([
            transforms.Resize(size), # Resize grayscale images too
            transforms.ToTensor(), # Converts grayscale PIL/numpy to [1, H, W] tensor
        ])

    def __getitem__(self, index):
        ref_img_path = self.ref_imgs[index]
        # query_img_path = self.query_imgs[index] # Not used
        coordinate = self.coordinates[index]

        try:
            # Load RGB images for main transforms
            ref_img_rgb_pil = IM.open(ref_img_path).convert('RGB')
            # query_img_rgb_pil = IM.open(query_img_path).convert('RGB') # Not used

            # Load Grayscale images for warping and feature matching transforms
            ref_img_gray_pil = IM.open(ref_img_path).convert('L')
        except Exception as e:
            print(f"Error opening image file: {ref_img_path}. Skipping index {index}. Error: {e}")
            return self.__getitem__((index + 1) % len(self))

        # Generate warped query (numpy grayscale) from reference (numpy grayscale)
        query_img_gray_np, yaw, roll, pitch = self.generate_query_img(np.array(ref_img_gray_pil))
        # Convert warped numpy back to PIL grayscale
        query_img_gray_pil = IM.fromarray(query_img_gray_np)

        # Apply main transform (RGB, incl. resize, norm) to original RGB ref/query
        if self.transform:
            # Transform the original RGB reference image
            ref_img_transformed = self.transform(ref_img_rgb_pil)
            # Transform the *original* RGB query image (from file path, seems intended based on original returns)
            # Let's assume we need the warped query transformed instead for consistency with OmniGlue variant
            query_img_transformed = self.transform(IM.fromarray(cv2.cvtColor(query_img_gray_np, cv2.COLOR_GRAY2RGB))) # Transform warped query
        else:
             # Manual transform if needed
             ref_img_transformed = F.to_tensor(transforms.Resize(self.img_size[0])(ref_img_rgb_pil))
             # Transform warped query manually
             query_rgb_pil = IM.fromarray(cv2.cvtColor(query_img_gray_np, cv2.COLOR_GRAY2RGB))
             query_img_transformed = F.to_tensor(transforms.Resize(self.img_size[0])(query_rgb_pil))


        # Apply grayscale transform (resize, ToTensor) to grayscale PIL images
        ref_img_rot = self.rot_transforms(ref_img_gray_pil)     # [1, H, W] Tensor
        query_img_rot = self.rot_transforms(query_img_gray_pil) # [1, H, W] Tensor


        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(yaw, dtype=torch.float)
        roll = torch.tensor(roll, dtype=torch.float)
        pitch = torch.tensor(pitch, dtype=torch.float)

        # Return transformed RGB ref/query, transformed grayscale ref/query, coords, angles
        return ref_img_transformed, query_img_transformed, ref_img_rot, query_img_rot, coordinate, roll, yaw, pitch


class PoseDataLoader_Warping_SIFT(PoseDataLoader_Warping):
    """ Warping dataloader variant possibly for SIFT (uses OpenCV directly). """
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        # perturbation parameter seems unused?
        super().__init__(dataset_file, dataset_folder, transform)
        self.img_size = (size, size)
        # This transform seems unused based on __getitem__ logic
        # self.rot_transforms = transforms.Compose([ ... ])

    def __getitem__(self, index):
        ref_img_path = self.ref_imgs[index]
        query_img_path = self.query_imgs[index] # Used for localization part
        coordinate = self.coordinates[index]

        try:
            # --- Localization Module Part (RGB, Transformed) ---
            ref_img_pil = IM.open(ref_img_path).convert('RGB')
            query_img_pil = IM.open(query_img_path).convert('RGB') # Load original query

            # Apply main transform (incl. resize, norm)
            if self.transform:
                ref_img_transformed = self.transform(ref_img_pil)
                query_img_transformed = self.transform(query_img_pil)
            else:
                ref_img_transformed = F.to_tensor(transforms.Resize(self.img_size[0])(ref_img_pil))
                query_img_transformed = F.to_tensor(transforms.Resize(self.img_size[0])(query_img_pil))

            # --- Rotation Module Part (Grayscale Numpy) ---
            ref_img_rot_np = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
            if ref_img_rot_np is None:
                 raise IOError(f"cv2 failed to read image: {ref_img_path}")

            # Generate warped query (numpy grayscale) from reference (numpy grayscale)
            query_img_rot_np, yaw, roll, pitch = self.generate_query_img(ref_img_rot_np.copy())

        except Exception as e:
            print(f"Error processing image file at index {index}: {ref_img_path} or {query_img_path}. Error: {e}")
            return self.__getitem__((index + 1) % len(self))


        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(yaw, dtype=torch.float)
        roll = torch.tensor(roll, dtype=torch.float)
        pitch = torch.tensor(pitch, dtype=torch.float)

        # Return transformed RGB ref/query, UNTRANSFORMED grayscale numpy ref/query, coords, angles
        return ref_img_transformed, query_img_transformed, ref_img_rot_np, query_img_rot_np, coordinate, roll, yaw, pitch

class TestPoseDataLoader(BaseDataLoader):
    """
    Loads ref/query image pair, applies random rotation to query, returns images, coords, angle.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        super().__init__(dataset_file, dataset_folder, transform, size, is_testing=True)

    def __getitem__(self, idx):
        query_img_path = self.ref_imgs[idx]
        coordinate = self.coordinates[idx]

        query_img = IM.open(query_img_path).convert('RGB')
        rotate_angle, query_img = self.rotation_transform(query_img.copy())

        cx, cy = query_img.size
        left = (cx - self.img_size) // 2
        top = (cy - self.img_size) // 2
        right = left + self.img_size
        bottom = top + self.img_size

        query_img = query_img.crop((left, top, right, bottom))

        if self.transform:
            query_img = self.transform(query_img)
        else:
            query_img = F.to_tensor(query_img)

        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(rotate_angle)

        return query_img, coordinate, yaw


class VisLocDataLoader(Dataset):
    """
    Loads ref/query image pair, applies random rotation to query, returns images, coords, angle.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        if not exists(dataset_file):
             raise FileNotFoundError(f"Dataset CSV file not found: {dataset_file}")
        if not os.path.isdir(dataset_folder):
             raise NotADirectoryError(f"Dataset image folder not found: {dataset_folder}")

        self.dataset_folder = dataset_folder
        self.transform = transform
        self.img_size = size

        self.rotate_angles = list(range(-180, 180, 15))
        self.rotation_transform = RandomDiscreteRotation(self.rotate_angles)

        self.sat_imgs, self.drone_imgs, self.coordinates = self._load_dataset(dataset_file, required_cols={'SAT_IMG', 'DRONE_IMG', 'LAT', 'LON'})

    def _load_dataset(self, dataset_file, required_cols: dict={'REF_IMG', 'QUERY_IMG', 'LAT', 'LON'}):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f">\tError reading {dataset_file}: {e}")

        if not required_cols.issubset(dataset_info.columns):
            raise ValueError(f"CSV file {dataset_file} must contain columns: {required_cols}")

        sat_imgs = []
        drone_imgs = []
        coordinates = []
        # drone_heading_angles = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            ref_img_path = os.path.join(self.dataset_folder, str(row['SAT_IMG']))
            query_img_path = os.path.join(self.dataset_folder, str(row['DRONE_IMG']))

            if exists(ref_img_path) and exists(query_img_path):
                sat_imgs.append(ref_img_path)
                drone_imgs.append(query_img_path)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))
                # yaw = float(row['YAW'])
                # drone_heading_angles.append(yaw)

        if not sat_imgs:
             raise RuntimeError(f"No valid image pairs found based on CSV {dataset_file} and folder {self.dataset_folder}")
        
        assert len(sat_imgs) == len(drone_imgs)
        assert len(sat_imgs) == len(coordinates)
        # assert len(sat_imgs) == len(drone_heading_angles)

        return sat_imgs, drone_imgs, coordinates

    def __len__(self):
        return len(self.sat_imgs)

    def __getitem__(self, idx):
        sat_img_path = self.sat_imgs[idx]
        query_img_path = self.drone_imgs[idx]
        coordinate = self.coordinates[idx]
        # yaw = self.yaws[idx]

        sat_img = IM.open(sat_img_path).convert('RGB')
        sat_img_ori = sat_img.copy()
        query_img = IM.open(query_img_path).convert('RGB')
        query_img_ori = query_img.copy()

        yaw, query_img = self.rotation_transform(query_img)

        sat_img = transforms.Resize(self.img_size)(sat_img)
        cx, cy = query_img.size
        left = (cx - self.img_size) // 2
        top = (cy - self.img_size) // 2
        right = left + self.img_size
        bottom = top + self.img_size

        query_img = query_img.crop((left, top, right, bottom))

        if self.transform:
            sat_img = self.transform(sat_img)
            query_img = self.transform(query_img)
        else:
            sat_img = F.to_tensor(sat_img)
            query_img = F.to_tensor(query_img)

        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(yaw)

        return sat_img, query_img, coordinate, yaw
    
class TestVisLocDataLoader(Dataset):
    def __init__(self, dataset_file, dataset_folder, transform=None, size=224):
        if not exists(dataset_file):
             raise FileNotFoundError(f"Dataset CSV file not found: {dataset_file}")
        if not os.path.isdir(dataset_folder):
             raise NotADirectoryError(f"Dataset image folder not found: {dataset_folder}")

        self.dataset_folder = dataset_folder
        self.transform = transform
        self.img_size = size

        self.sat_imgs, self.drone_imgs, self.coordinates, self.yaws = self._load_dataset(dataset_file, required_cols={'SAT_IMG', 'DRONE_IMG', 'LAT', 'LON', 'YAW'})

    def _load_dataset(self, dataset_file, required_cols: dict={'REF_IMG', 'QUERY_IMG', 'LAT', 'LON'}):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f">\tError reading {dataset_file}: {e}")

        if not required_cols.issubset(dataset_info.columns):
            raise ValueError(f"CSV file {dataset_file} must contain columns: {required_cols}")

        sat_imgs = []
        drone_imgs = []
        coordinates = []
        drone_heading_angles = []

        print(f">\tLoading dataset info from: {dataset_file}")
        for _, row in tqdm(dataset_info.iterrows(), total=len(dataset_info), desc="Checking image paths"):
            ref_img_path = os.path.join(self.dataset_folder, str(row['SAT_IMG']))
            query_img_path = os.path.join(self.dataset_folder, str(row['DRONE_IMG']))

            if exists(ref_img_path) and exists(query_img_path):
                sat_imgs.append(ref_img_path)
                drone_imgs.append(query_img_path)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))
                yaw = float(row['YAW'])
                drone_heading_angles.append(yaw)

        if not sat_imgs:
             raise RuntimeError(f"No valid image pairs found based on CSV {dataset_file} and folder {self.dataset_folder}")
        
        assert len(sat_imgs) == len(drone_imgs)
        assert len(sat_imgs) == len(coordinates)
        assert len(sat_imgs) == len(drone_heading_angles)

        return sat_imgs, drone_imgs, coordinates, drone_heading_angles
    
    def __len__(self):
        return len(self.sat_imgs)

    def __getitem__(self, idx):
        sat_img_path = self.sat_imgs[idx]
        query_img_path = self.drone_imgs[idx]
        coordinate = self.coordinates[idx]
        yaw = self.yaws[idx]

        sat_img = IM.open(sat_img_path).convert('RGB')
        sat_img_ori = sat_img.copy()
        query_img = IM.open(query_img_path).convert('RGB')
        query_img_ori = query_img.copy()

        query_img = transforms.Resize(self.img_size)(query_img)

        if self.transform:
            sat_img = self.transform(sat_img)
            query_img = self.transform(query_img)
        else:
            sat_img = F.to_tensor(sat_img)
            query_img = F.to_tensor(query_img)

        coordinate = torch.tensor(coordinate, dtype=torch.float)
        yaw = torch.tensor(yaw)

        return sat_img, query_img, coordinate, yaw, F.to_tensor(sat_img_ori),  F.to_tensor(query_img_ori)

class DataLoaderTypesEnum(Enum):
    Pose = 'Pose'
    CrossSeasonPose = 'CrossSeasonPose'
    WarpingOmniGlue = 'WarpingOmniGlue'
    WarpingLightGlue = 'WarpingLightGlue'
    WarpingSIFT = 'WarpingSIFT'
    TestPose = 'TestPose'

    #####
    # Public Datasets
    #####
    VisLoc = 'VisLoc'
    TestVisLoc = 'TestVisLoc'

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

        self.train_transform = img_train_transform(size=self.hparams.image_size)
        self.val_transform = img_val_transform(size=self.hparams.image_size)
        self.test_transform = img_test_transform(size=self.hparams.image_size)

    def setup(self, stage: str = None):
        """Create the datasets based on the stage."""

        if self.hparams.dataset_type is DataLoaderTypesEnum.Pose:
            TrainValPredDatasetClass = PoseDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.CrossSeasonPose:
            TrainValPredDatasetClass = CrossSeasonPoseDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.WarpingOmniGlue:
            TrainValPredDatasetClass = lambda csv, folder, transform: PoseDataLoader_Warping_OmniGlue(csv, folder, transform, size=self.hparams.image_size)
        elif self.hparams.dataset_type is DataLoaderTypesEnum.WarpingLightGlue:
            TrainValPredDatasetClass = lambda csv, folder, transform: PoseDataLoader_Warping_LightGlue(csv, folder, transform, size=self.hparams.image_size)
        elif self.hparams.dataset_type is DataLoaderTypesEnum.WarpingSIFT:
            TrainValPredDatasetClass = lambda csv, folder, transform: PoseDataLoader_Warping_SIFT(csv, folder, transform, size=self.hparams.image_size)
        elif self.hparams.dataset_type is DataLoaderTypesEnum.TestPose:
            PredDatasetClass = TestPoseDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.VisLoc:
            TrainValPredDatasetClass = VisLocDataLoader
        elif self.hparams.dataset_type is DataLoaderTypesEnum.TestVisLoc:
            PredDatasetClass = TestVisLocDataLoader
        else:
            raise ValueError(f"Unknown dataset_type: {self.hparams.dataset_type}")

        if stage == 'fit':
            print(f"Setting up 'fit' stage with dataset type: {self.hparams.dataset_type}")

            if self.hparams.dataset_type is DataLoaderTypesEnum.VisLoc:
                total_ds = TrainValPredDatasetClass(
                    dataset_file=self.hparams.train_csv,
                    dataset_folder=self.hparams.dataset_folder,
                    transform=self.train_transform,
                    size=self.hparams.image_size
                )
                print(f"len(total_ds): {len(total_ds)}")
                self.train_dataset, self.val_dataset = random_split(total_ds, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
            else:
                self.train_dataset = TrainValPredDatasetClass(
                    dataset_file=self.hparams.train_csv,
                    dataset_folder=self.hparams.dataset_folder,
                    transform=self.train_transform,
                    size=self.hparams.image_size
                )
                # Use the same dataset class type for validation, but with validation transform
                self.val_dataset = TrainValPredDatasetClass(
                    dataset_file=self.hparams.val_csv,
                    dataset_folder=self.hparams.dataset_folder,
                    transform=self.val_transform,
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
                transform=self.test_transform, # Use test transform for prediction
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

