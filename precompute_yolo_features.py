#!/usr/bin/env python3
"""
Pre-compute YOLO detection features for NuScenes VQA.

Similar to bev_image_radar.ipynb but uses YOLOv8 instead of ResNet BEV.
Run this ONCE before training. Output is saved as .npy files keyed by sample_token.

Usage:
    CUDA_VISIBLE_DEVICES=1 python precompute_yolo_features.py

Output features per object (13 dims):
    [x1, y1, x2, y2, area, aspect_ratio, confidence, center_x, center_y, class_id,
     radar_velocity, radar_rcs, radar_match]

Output shape: (80, 13) per sample — saved to yolo_features/{sample_token}.npy
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "/media/nas_mount/anwar2/experiment/dataset/nuscenes"
VERSION = "v1.0-trainval"
OUT_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/yolo_features"

YOLO_MODEL = "yolov8m.pt"          # YOLOv8 medium (COCO pretrained)
CONF_THRESHOLD = 0.25               # Lower threshold to get more detections
NUM_OBJECTS = 80                     # Max objects per sample (pad/trim)
FEATURE_DIM = 13                    # Feature vector size per object
FUSE_RADAR = True                   # Fuse radar data with YOLO detections

# ============================================================
# YOLO Feature Extraction
# ============================================================

def parse_yolo_detections(results, img_w, img_h, max_objs=NUM_OBJECTS, conf_thresh=CONF_THRESHOLD):
    """
    Parse YOLO results into a (max_objs, 10) feature matrix.
    Features: [x1_n, y1_n, x2_n, y2_n, area_n, aspect_ratio, conf, cx_n, cy_n, class_id]
    All spatial coords normalized by image dimensions.
    """
    features = np.zeros((max_objs, 10), dtype=np.float32)

    if results is None or len(results) == 0:
        return features

    result = results[0]  # single image
    if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
        return features

    boxes = result.boxes
    bbox = boxes.xyxy.cpu().numpy()      # (N, 4) [x1, y1, x2, y2]
    conf = boxes.conf.cpu().numpy()      # (N,)
    cls = boxes.cls.cpu().numpy()        # (N,)

    # Filter by confidence
    mask = conf >= conf_thresh
    bbox = bbox[mask]
    conf = conf[mask]
    cls = cls[mask]

    if len(bbox) == 0:
        return features

    # Sort by confidence (highest first)
    sort_idx = np.argsort(-conf)
    bbox = bbox[sort_idx]
    conf = conf[sort_idx]
    cls = cls[sort_idx]

    n = min(len(bbox), max_objs)
    bbox = bbox[:n]
    conf = conf[:n]
    cls = cls[:n]

    # Normalized coordinates
    x1_n = bbox[:, 0] / img_w
    y1_n = bbox[:, 1] / img_h
    x2_n = bbox[:, 2] / img_w
    y2_n = bbox[:, 3] / img_h

    # Derived features
    area_n = (x2_n - x1_n) * (y2_n - y1_n)
    aspect_ratio = (bbox[:, 2] - bbox[:, 0]) / (bbox[:, 3] - bbox[:, 1] + 1e-6)
    cx_n = (x1_n + x2_n) / 2.0
    cy_n = (y1_n + y2_n) / 2.0

    features[:n, 0] = x1_n
    features[:n, 1] = y1_n
    features[:n, 2] = x2_n
    features[:n, 3] = y2_n
    features[:n, 4] = area_n
    features[:n, 5] = aspect_ratio
    features[:n, 6] = conf
    features[:n, 7] = cx_n
    features[:n, 8] = cy_n
    features[:n, 9] = cls

    return features


# ============================================================
# Radar Fusion
# ============================================================

def fuse_radar_with_detections(det_features, radar_pc, cam_intrinsic, img_w, img_h):
    """
    Match radar points to YOLO detections by projecting radar points to image plane.
    Adds 3 radar dims: [velocity, rcs, radar_match_flag]

    Returns: (80, 13) array
    """
    n_objs = NUM_OBJECTS
    radar_feats = np.zeros((n_objs, 3), dtype=np.float32)

    if radar_pc is None or radar_pc.points.shape[1] == 0:
        return np.concatenate([det_features, radar_feats], axis=-1)

    # Radar points in camera frame
    pts_3d = radar_pc.points[:3, :]  # (3, N) — x, y, z in camera coords
    vx = radar_pc.points[8, :]       # velocity x
    vy = radar_pc.points[9, :]       # velocity y
    velocity = np.sqrt(vx**2 + vy**2)
    rcs = radar_pc.points[5, :]      # radar cross section

    # Only keep points in front of camera (z > 0)
    front_mask = pts_3d[2, :] > 0
    if not front_mask.any():
        return np.concatenate([det_features, radar_feats], axis=-1)

    pts_3d = pts_3d[:, front_mask]
    velocity = velocity[front_mask]
    rcs = rcs[front_mask]

    # Project to 2D image coordinates: p = K * [x/z, y/z, 1]
    cam_K = np.array(cam_intrinsic)
    pts_2d = cam_K @ (pts_3d / pts_3d[2:3, :])  # (3, N)
    px = pts_2d[0, :]  # pixel x
    py = pts_2d[1, :]  # pixel y

    # Filter points within image bounds
    img_mask = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    px = px[img_mask]
    py = py[img_mask]
    velocity = velocity[img_mask]
    rcs = rcs[img_mask]

    if len(px) == 0:
        return np.concatenate([det_features, radar_feats], axis=-1)

    # Match radar points to YOLO detections
    for r_idx in range(len(px)):
        rx, ry = px[r_idx], py[r_idx]
        rx_n, ry_n = rx / img_w, ry / img_h  # normalized

        best_det = -1
        best_dist = float('inf')

        for d_idx in range(n_objs):
            if det_features[d_idx, 6] == 0:  # no detection here
                continue

            # Check if radar point is inside bbox
            x1, y1, x2, y2 = det_features[d_idx, 0:4]
            if x1 <= rx_n <= x2 and y1 <= ry_n <= y2:
                # Point is inside box → compute distance to center
                cx, cy = det_features[d_idx, 7], det_features[d_idx, 8]
                dist = np.sqrt((rx_n - cx)**2 + (ry_n - cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_det = d_idx

        if best_det >= 0:
            # Accumulate radar info (average if multiple radar points per detection)
            if radar_feats[best_det, 2] == 0:  # first match
                radar_feats[best_det, 0] = velocity[r_idx]
                radar_feats[best_det, 1] = rcs[r_idx]
                radar_feats[best_det, 2] = 1.0
            else:  # average with existing
                n_matches = radar_feats[best_det, 2]
                radar_feats[best_det, 0] = (radar_feats[best_det, 0] * n_matches + velocity[r_idx]) / (n_matches + 1)
                radar_feats[best_det, 1] = (radar_feats[best_det, 1] * n_matches + rcs[r_idx]) / (n_matches + 1)
                radar_feats[best_det, 2] += 1.0

    # Normalize radar match count (cap at 1.0 for the flag, keep count info in value)
    radar_feats[:, 2] = np.clip(radar_feats[:, 2], 0, 1)

    return np.concatenate([det_features, radar_feats], axis=-1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-compute YOLO features for NuScenes VQA")
    parser.add_argument("--data-root", default=DATA_ROOT, help="NuScenes data root")
    parser.add_argument("--version", default=VERSION, help="NuScenes version")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for features")
    parser.add_argument("--yolo-model", default=YOLO_MODEL, help="YOLO model name")
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD, help="Detection confidence threshold")
    parser.add_argument("--no-radar", action="store_true", help="Disable radar fusion")
    parser.add_argument("--batch-size", type=int, default=16, help="YOLO inference batch size")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load NuScenes ---
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import RadarPointCloud
    from pyquaternion import Quaternion

    print(f"Loading NuScenes {args.version} from {args.data_root}...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    print(f"Loaded {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    # --- Load YOLO ---
    from ultralytics import YOLO
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo.to(device)
    print(f"YOLO loaded on {device}")

    # --- Check existing features (for resume) ---
    existing = set()
    if os.path.exists(args.out_dir):
        existing = {f.replace('.npy', '') for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    print(f"Found {len(existing)} existing features, will skip those")

    # --- Process all samples ---
    fuse_radar = FUSE_RADAR and not args.no_radar
    skipped = 0
    processed = 0
    errors = 0

    for scene in tqdm(nusc.scene, desc="Scenes"):
        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = nusc.get("sample", sample_token)

            # Skip if already computed
            if sample_token in existing:
                skipped += 1
                sample_token = sample["next"]
                continue

            try:
                # ---------- CAMERA ----------
                cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                img_path = os.path.join(args.data_root, cam_sd["filename"])
                image = Image.open(img_path).convert("RGB")
                img_w, img_h = image.size

                # ---------- YOLO DETECTION ----------
                results = yolo(image, verbose=False, conf=args.conf_thresh)
                det_features = parse_yolo_detections(results, img_w, img_h)

                # ---------- RADAR FUSION ----------
                if fuse_radar:
                    # Get radar data
                    radar_sd = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
                    radar_pc = RadarPointCloud.from_file(
                        os.path.join(args.data_root, radar_sd["filename"])
                    )

                    # Transform radar to camera frame
                    radar_cs = nusc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])
                    radar_pose = nusc.get("ego_pose", radar_sd["ego_pose_token"])
                    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
                    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

                    # Radar → Ego
                    radar_pc.rotate(Quaternion(radar_cs["rotation"]).rotation_matrix)
                    radar_pc.translate(np.array(radar_cs["translation"]))

                    # Ego → Global
                    radar_pc.rotate(Quaternion(radar_pose["rotation"]).rotation_matrix)
                    radar_pc.translate(np.array(radar_pose["translation"]))

                    # Global → Ego (camera timestamp)
                    radar_pc.translate(-np.array(cam_pose["translation"]))
                    radar_pc.rotate(Quaternion(cam_pose["rotation"]).rotation_matrix.T)

                    # Ego → Camera
                    radar_pc.translate(-np.array(cam_cs["translation"]))
                    radar_pc.rotate(Quaternion(cam_cs["rotation"]).rotation_matrix.T)

                    cam_intrinsic = cam_cs["camera_intrinsic"]
                    obj_feat = fuse_radar_with_detections(det_features, radar_pc, cam_intrinsic, img_w, img_h)
                else:
                    obj_feat = det_features  # (80, 10) without radar

                # ---------- SAVE ----------
                np.save(
                    os.path.join(args.out_dir, f"{sample_token}.npy"),
                    obj_feat.astype(np.float32)
                )
                processed += 1

            except Exception as e:
                print(f"\n  Error on {sample_token}: {e}")
                errors += 1

            sample_token = sample["next"]

    print(f"\n✅ YOLO feature extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped (existing): {skipped}")
    print(f"   Errors: {errors}")
    print(f"   Output: {args.out_dir}")
    print(f"   Feature shape: ({NUM_OBJECTS}, {FEATURE_DIM if fuse_radar else 10})")


if __name__ == "__main__":
    main()
