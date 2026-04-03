#!/usr/bin/env python3
"""
Pre-compute RICH YOLO features for NuScenes VQA (Phase 2).

Extracts ROI-pooled backbone features from YOLOv8 for each detection,
giving 256-dim visual appearance features PER OBJECT instead of just
13-dim bounding box metadata.

Output features per object (269 dims):
    [x1, y1, x2, y2, area, aspect_ratio, confidence, cx, cy, class_id,
     radar_velocity, radar_rcs, radar_match,
     backbone_feat_0 ... backbone_feat_255]

Output shape: (80, 269) per sample

Usage:
    # Run on GPU 1 while training runs on GPU 0:
    CUDA_VISIBLE_DEVICES=2 python precompute_rich_yolo_features.py

    # Or CPU-only (slower but won't compete with training):
    CUDA_VISIBLE_DEVICES="" python precompute_rich_yolo_features.py --device cpu
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "/media/nas_mount/anwar2/experiment/dataset/nuscenes"
VERSION = "v1.0-trainval"

# Where existing 13-dim YOLO features are stored
EXISTING_YOLO_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/yolo_features"

# Output directory for rich features
OUT_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/yolo_features_rich"

YOLO_MODEL = "yolov8m.pt"
CONF_THRESHOLD = 0.25
NUM_OBJECTS = 80
BACKBONE_FEAT_DIM = 256      # Feature dimension from ROI pooling
TOTAL_FEAT_DIM = 13 + BACKBONE_FEAT_DIM  # 269

ROI_OUTPUT_SIZE = 7           # ROI-align output spatial size (7x7)


# ============================================================
# Backbone Feature Extractor
# ============================================================

class YOLOFeatureExtractor:
    """
    Wraps YOLOv8 to extract both detections AND backbone features.
    Uses a forward hook on the SPPF layer (backbone output) to capture
    the feature map, then ROI-aligns each detection box.
    """

    def __init__(self, model_name, device='cuda'):
        from ultralytics import YOLO

        self.device = device
        self.yolo = YOLO(model_name)
        self.yolo.to(device)

        # Storage for hooked features
        self._backbone_feats = None

        # Register hook on the SPPF layer (layer 9 in YOLOv8)
        # This is the last layer of the backbone before the neck
        backbone_layer = self.yolo.model.model[9]
        backbone_layer.register_forward_hook(self._hook_fn)

        # Get the actual feature dimension by running a dummy forward pass
        from PIL import Image
        dummy_img = Image.new("RGB", (640, 640))
        with torch.no_grad():
            self.yolo(dummy_img, verbose=False)
        self._feat_channels = self._backbone_feats.shape[1]
        print(f"  Backbone feature channels: {self._feat_channels}")

    def _hook_fn(self, module, input, output):
        """Captures backbone feature map during forward pass."""
        self._backbone_feats = output

    def extract(self, image_path, conf_thresh=CONF_THRESHOLD, max_objs=NUM_OBJECTS):
        """
        Run YOLO detection and extract backbone features for each detection.

        Returns:
            features: (max_objs, 13 + feat_dim) array
        """
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # Run inference (this triggers the hook)
        results = self.yolo(image, verbose=False, conf=conf_thresh)

        # Get the backbone feature map captured by hook
        feat_map = self._backbone_feats  # (1, C, H_feat, W_feat)

        # Parse detections
        features = np.zeros((max_objs, TOTAL_FEAT_DIM), dtype=np.float32)

        if results is None or len(results) == 0:
            return features

        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            return features

        boxes = result.boxes
        bbox = boxes.xyxy.cpu().numpy()      # (N, 4) pixel coords
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        # Filter and sort by confidence
        mask = conf >= conf_thresh
        bbox, conf, cls = bbox[mask], conf[mask], cls[mask]
        if len(bbox) == 0:
            return features

        sort_idx = np.argsort(-conf)
        bbox, conf, cls = bbox[sort_idx], conf[sort_idx], cls[sort_idx]
        n = min(len(bbox), max_objs)

        # ---- Compute standard 13-dim features (same as original) ----
        x1_n = bbox[:n, 0] / img_w
        y1_n = bbox[:n, 1] / img_h
        x2_n = bbox[:n, 2] / img_w
        y2_n = bbox[:n, 3] / img_h
        area_n = (x2_n - x1_n) * (y2_n - y1_n)
        aspect_ratio = (bbox[:n, 2] - bbox[:n, 0]) / (bbox[:n, 3] - bbox[:n, 1] + 1e-6)
        cx_n = (x1_n + x2_n) / 2.0
        cy_n = (y1_n + y2_n) / 2.0

        features[:n, 0] = x1_n
        features[:n, 1] = y1_n
        features[:n, 2] = x2_n
        features[:n, 3] = y2_n
        features[:n, 4] = area_n
        features[:n, 5] = aspect_ratio
        features[:n, 6] = conf[:n]
        features[:n, 7] = cx_n
        features[:n, 8] = cy_n
        features[:n, 9] = cls[:n]
        # dims 10-12 (radar) will be filled later by radar fusion

        # ---- ROI-pool backbone features for each detection ----
        if feat_map is not None:
            _, C, H_feat, W_feat = feat_map.shape

            for i in range(n):
                # Convert pixel coords to feature map coords
                fx1 = bbox[i, 0] / img_w * W_feat
                fy1 = bbox[i, 1] / img_h * H_feat
                fx2 = bbox[i, 2] / img_w * W_feat
                fy2 = bbox[i, 3] / img_h * H_feat

                # Clamp to valid range
                fx1 = max(0, min(int(fx1), W_feat - 1))
                fy1 = max(0, min(int(fy1), H_feat - 1))
                fx2 = max(fx1 + 1, min(int(fx2) + 1, W_feat))
                fy2 = max(fy1 + 1, min(int(fy2) + 1, H_feat))

                # Extract region and adaptive pool to fixed size
                roi = feat_map[0, :, fy1:fy2, fx1:fx2]  # (C, h, w)
                roi = roi.unsqueeze(0)  # (1, C, h, w)

                # Adaptive average pool to get fixed-size feature
                pooled = F.adaptive_avg_pool2d(roi, (1, 1))  # (1, C, 1, 1)
                roi_feat = pooled.squeeze().cpu().numpy()  # (C,)

                # Truncate/pad to target dim
                feat_dim = min(len(roi_feat), BACKBONE_FEAT_DIM)
                features[i, 13:13 + feat_dim] = roi_feat[:feat_dim]

        return features


# ============================================================
# Radar Fusion (same as original)
# ============================================================

def fuse_radar_with_detections(det_features, radar_pc, cam_intrinsic, img_w, img_h):
    """
    Match radar points to detections and fill dims 10-12.
    Works with both 13-dim and 269-dim feature arrays.
    """
    if radar_pc is None or radar_pc.points.shape[1] == 0:
        return det_features

    pts_3d = radar_pc.points[:3, :]
    vx = radar_pc.points[8, :]
    vy = radar_pc.points[9, :]
    velocity = np.sqrt(vx**2 + vy**2)
    rcs = radar_pc.points[5, :]

    front_mask = pts_3d[2, :] > 0
    if not front_mask.any():
        return det_features

    pts_3d = pts_3d[:, front_mask]
    velocity = velocity[front_mask]
    rcs = rcs[front_mask]

    cam_K = np.array(cam_intrinsic)
    pts_2d = cam_K @ (pts_3d / pts_3d[2:3, :])
    px, py = pts_2d[0, :], pts_2d[1, :]

    img_mask = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    px, py = px[img_mask], py[img_mask]
    velocity, rcs = velocity[img_mask], rcs[img_mask]

    if len(px) == 0:
        return det_features

    for r_idx in range(len(px)):
        rx_n, ry_n = px[r_idx] / img_w, py[r_idx] / img_h
        best_det, best_dist = -1, float('inf')

        for d_idx in range(NUM_OBJECTS):
            if det_features[d_idx, 6] == 0:
                continue
            x1, y1, x2, y2 = det_features[d_idx, 0:4]
            if x1 <= rx_n <= x2 and y1 <= ry_n <= y2:
                cx, cy = det_features[d_idx, 7], det_features[d_idx, 8]
                dist = np.sqrt((rx_n - cx)**2 + (ry_n - cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_det = d_idx

        if best_det >= 0:
            if det_features[best_det, 12] == 0:  # first match
                det_features[best_det, 10] = velocity[r_idx]
                det_features[best_det, 11] = rcs[r_idx]
                det_features[best_det, 12] = 1.0
            else:
                n_matches = det_features[best_det, 12]
                det_features[best_det, 10] = (det_features[best_det, 10] * n_matches + velocity[r_idx]) / (n_matches + 1)
                det_features[best_det, 11] = (det_features[best_det, 11] * n_matches + rcs[r_idx]) / (n_matches + 1)
                det_features[best_det, 12] += 1.0

    det_features[:, 12] = np.clip(det_features[:, 12], 0, 1)
    return det_features


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Extract rich YOLO backbone features for NuScenes VQA")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--yolo-model", default=YOLO_MODEL)
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--no-radar", action="store_true")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load NuScenes ---
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import RadarPointCloud
    from pyquaternion import Quaternion

    print(f"Loading NuScenes {args.version} from {args.data_root}...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    print(f"Loaded {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    # --- Load YOLO with feature extraction ---
    print(f"Loading YOLO feature extractor: {args.yolo_model}")
    extractor = YOLOFeatureExtractor(args.yolo_model, device=args.device)
    print(f"Ready! Output feature dim: {TOTAL_FEAT_DIM}")

    # --- Check existing (for resume) ---
    existing = set()
    if os.path.exists(args.out_dir):
        existing = {f.replace('.npy', '') for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    print(f"Found {len(existing)} existing features, will skip those")

    # --- Process ---
    fuse_radar = not args.no_radar
    skipped, processed, errors = 0, 0, 0

    for scene in tqdm(nusc.scene, desc="Scenes"):
        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = nusc.get("sample", sample_token)

            if sample_token in existing:
                skipped += 1
                sample_token = sample["next"]
                continue

            try:
                # --- Camera ---
                cam_sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                img_path = os.path.join(args.data_root, cam_sd["filename"])
                image = Image.open(img_path).convert("RGB")
                img_w, img_h = image.size

                # --- YOLO + backbone features ---
                with torch.no_grad():
                    obj_feat = extractor.extract(img_path, conf_thresh=args.conf_thresh)

                # --- Radar fusion ---
                if fuse_radar:
                    radar_sd = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
                    radar_pc = RadarPointCloud.from_file(
                        os.path.join(args.data_root, radar_sd["filename"])
                    )

                    radar_cs = nusc.get("calibrated_sensor", radar_sd["calibrated_sensor_token"])
                    radar_pose = nusc.get("ego_pose", radar_sd["ego_pose_token"])
                    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
                    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

                    radar_pc.rotate(Quaternion(radar_cs["rotation"]).rotation_matrix)
                    radar_pc.translate(np.array(radar_cs["translation"]))
                    radar_pc.rotate(Quaternion(radar_pose["rotation"]).rotation_matrix)
                    radar_pc.translate(np.array(radar_pose["translation"]))
                    radar_pc.translate(-np.array(cam_pose["translation"]))
                    radar_pc.rotate(Quaternion(cam_pose["rotation"]).rotation_matrix.T)
                    radar_pc.translate(-np.array(cam_cs["translation"]))
                    radar_pc.rotate(Quaternion(cam_cs["rotation"]).rotation_matrix.T)

                    cam_intrinsic = cam_cs["camera_intrinsic"]
                    obj_feat = fuse_radar_with_detections(obj_feat, radar_pc, cam_intrinsic, img_w, img_h)

                # --- Save ---
                np.save(
                    os.path.join(args.out_dir, f"{sample_token}.npy"),
                    obj_feat.astype(np.float32)
                )
                processed += 1

            except Exception as e:
                print(f"\n  Error on {sample_token}: {e}")
                errors += 1

            sample_token = sample["next"]

    print(f"\n✅ Rich YOLO feature extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped (existing): {skipped}")
    print(f"   Errors: {errors}")
    print(f"   Output: {args.out_dir}")
    print(f"   Feature shape: ({NUM_OBJECTS}, {TOTAL_FEAT_DIM})")


if __name__ == "__main__":
    main()
