import cv2
import numpy as np

import torch

from .matching import Matching
from .utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


def superglue_matching(img1, img2, device='cuda'):
    def preprocess_image(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None, None].to(device)
        return img

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 2048
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }

    matching = Matching(config).eval().to(device)

    with torch.no_grad():
        img1_tensor = preprocess_image(img1)
        img2_tensor = preprocess_image(img2)
        
        pred = matching({'image0': img1_tensor, 'image1': img2_tensor})
        
        # 提取匹配結果
        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    H_pred, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=0.25, maxIters=10000, confidence=0.9999)

    if H_pred is None or not H_pred.all():
        raise Exception("H_pred is None")

    A = H_pred[0:2, 0:2]
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    yaw_rad = np.arctan2(c, d) if np.isclose(a, d) and np.isclose(-b, c) else np.arctan2(-b, a)
    # yaw_rad = np.arctan2(H_pred[1, 0], H_pred[0, 0])
    yaw_angle = np.rad2deg(yaw_rad)

    return H_pred, yaw_angle