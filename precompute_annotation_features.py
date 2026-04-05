#!/usr/bin/env python3
"""
Pre-compute ANNOTATION-BASED features for NuScenes VQA.

Instead of raw BEV/YOLO backbone features, extract structured per-object
features from NuScenes ground-truth annotations:
  - Object category (23 classes → embedded by model)
  - Object attribute (9 attributes → embedded by model)
  - 3D position in ego frame (x, y, z)
  - 3D size (width, length, height)
  - Velocity (vx, vy)
  - Distance and angle from ego
  - Heading direction
  - Visibility (num lidar/radar points)

Output shape: (MAX_OBJECTS, 16) per sample

Usage:
    python precompute_annotation_features.py
    python precompute_annotation_features.py --data-root /path/to/nuscenes
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "/media/nas_mount/anwar2/experiment/dataset/nuscenes"
VERSION = "v1.0-trainval"
OUT_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/annotation_features"
MAX_OBJECTS = 100  # max objects per scene

# Feature layout: 16 dims per object
# [0]  category_id      (0-22, categorical)
# [1]  attribute_id     (0-8, categorical; 0 = no attribute)
# [2]  x               (ego frame, normalized: /50)
# [3]  y               (ego frame, normalized: /50)
# [4]  z               (ego frame, normalized: /5)
# [5]  width           (normalized: /10)
# [6]  length          (normalized: /10)
# [7]  height          (normalized: /5)
# [8]  vx              (ego frame, normalized: /20)
# [9]  vy              (ego frame, normalized: /20)
# [10] heading_sin     (sin of yaw in ego frame)
# [11] heading_cos     (cos of yaw in ego frame)
# [12] distance        (dist from ego, normalized: /50)
# [13] angle_sin       (sin of angle from ego forward)
# [14] angle_cos       (cos of angle from ego forward)
# [15] visibility      (log(1 + num_lidar_pts) / 5)

FEAT_DIM = 16


# NuScenes category → integer mapping (23 categories)
CATEGORY_MAP = {
    'human.pedestrian.adult': 0,
    'human.pedestrian.child': 1,
    'human.pedestrian.wheelchair': 2,
    'human.pedestrian.stroller': 3,
    'human.pedestrian.personal_mobility': 4,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.construction_worker': 6,
    'animal': 7,
    'vehicle.car': 8,
    'vehicle.motorcycle': 9,
    'vehicle.bicycle': 10,
    'vehicle.bus.bendy': 11,
    'vehicle.bus.rigid': 12,
    'vehicle.truck': 13,
    'vehicle.construction': 14,
    'vehicle.emergency.ambulance': 15,
    'vehicle.emergency.police': 16,
    'vehicle.trailer': 17,
    'movable_object.barrier': 18,
    'movable_object.trafficcone': 19,
    'movable_object.pushable_pullable': 20,
    'movable_object.debris': 21,
    'static_object.bicycle_rack': 22,
}

# NuScenes attribute → integer mapping (9 attributes, 0 = none)
ATTRIBUTE_MAP = {
    '': 0,
    'vehicle.moving': 1,
    'vehicle.parked': 2,
    'vehicle.stopped': 3,
    'cycle.with_rider': 4,
    'cycle.without_rider': 5,
    'pedestrian.moving': 6,
    'pedestrian.standing': 7,
    'pedestrian.sitting_lying_down': 8,
}


def extract_sample_features(nusc, sample_token, Quaternion):
    """
    Extract structured annotation features for one sample.
    Returns (MAX_OBJECTS, FEAT_DIM) array.
    """
    features = np.zeros((MAX_OBJECTS, FEAT_DIM), dtype=np.float32)

    sample = nusc.get('sample', sample_token)

    # Get ego pose (using LIDAR_TOP as reference)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    # Get all annotations for this sample
    ann_tokens = sample['anns']
    n_objects = min(len(ann_tokens), MAX_OBJECTS)

    for i in range(n_objects):
        ann = nusc.get('sample_annotation', ann_tokens[i])

        # ---- Category ----
        cat_name = ann['category_name']
        cat_id = CATEGORY_MAP.get(cat_name, 0)
        features[i, 0] = cat_id

        # ---- Attribute ----
        attr_id = 0
        if len(ann['attribute_tokens']) > 0:
            attr = nusc.get('attribute', ann['attribute_tokens'][0])
            attr_id = ATTRIBUTE_MAP.get(attr['name'], 0)
        features[i, 1] = attr_id

        # ---- Position (global → ego frame) ----
        pos_global = np.array(ann['translation'])
        pos_ego = ego_rotation.inverse.rotate(pos_global - ego_translation)

        features[i, 2] = pos_ego[0] / 50.0   # x (forward)
        features[i, 3] = pos_ego[1] / 50.0   # y (left)
        features[i, 4] = pos_ego[2] / 5.0    # z (up)

        # ---- Size ----
        size = ann['size']  # [width, length, height]
        features[i, 5] = size[0] / 10.0   # width
        features[i, 6] = size[1] / 10.0   # length
        features[i, 7] = size[2] / 5.0    # height

        # ---- Velocity ----
        try:
            vel = nusc.box_velocity(ann['token'])
            if np.any(np.isnan(vel)):
                vel = np.array([0.0, 0.0, 0.0])
            # Transform to ego frame
            vel_ego = ego_rotation.inverse.rotate(vel)
            features[i, 8] = vel_ego[0] / 20.0   # vx
            features[i, 9] = vel_ego[1] / 20.0   # vy
        except:
            features[i, 8] = 0.0
            features[i, 9] = 0.0

        # ---- Heading ----
        rot_global = Quaternion(ann['rotation'])
        rot_ego = ego_rotation.inverse * rot_global
        yaw = rot_ego.yaw_pitch_roll[0]
        features[i, 10] = np.sin(yaw)
        features[i, 11] = np.cos(yaw)

        # ---- Distance and angle from ego ----
        dist = np.sqrt(pos_ego[0]**2 + pos_ego[1]**2)
        angle = np.arctan2(pos_ego[1], pos_ego[0])

        features[i, 12] = dist / 50.0
        features[i, 13] = np.sin(angle)
        features[i, 14] = np.cos(angle)

        # ---- Visibility (lidar points) ----
        features[i, 15] = np.log1p(ann['num_lidar_pts']) / 5.0

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    print(f"Loading NuScenes {args.version} from {args.data_root}...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    print(f"Loaded {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    # Check existing (for resume)
    existing = set()
    if os.path.exists(args.out_dir):
        existing = {f.replace('.npy', '') for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    print(f"Found {len(existing)} existing features, will skip those")

    processed, skipped, errors = 0, 0, 0

    for scene in tqdm(nusc.scene, desc="Scenes"):
        sample_token = scene["first_sample_token"]

        while sample_token:
            sample = nusc.get("sample", sample_token)

            if sample_token in existing:
                skipped += 1
                sample_token = sample["next"]
                continue

            try:
                feat = extract_sample_features(nusc, sample_token, Quaternion)
                np.save(
                    os.path.join(args.out_dir, f"{sample_token}.npy"),
                    feat.astype(np.float32)
                )
                processed += 1
            except Exception as e:
                print(f"\n  Error on {sample_token}: {e}")
                errors += 1

            sample_token = sample["next"]

    print(f"\n✅ Annotation feature extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped (existing): {skipped}")
    print(f"   Errors: {errors}")
    print(f"   Output: {args.out_dir}")
    print(f"   Feature shape: ({MAX_OBJECTS}, {FEAT_DIM})")


if __name__ == "__main__":
    main()
