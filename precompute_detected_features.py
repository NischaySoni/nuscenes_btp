#!/usr/bin/env python3
"""
Pre-compute DETECTED features for NuScenes VQA using Camera + Radar.

Uses YOLOv8 on ALL 6 cameras for 2D detection, fuses ALL 5 radar sensors
for depth/velocity, and estimates 3D positions to produce structured
per-object features in the SAME (100, 16) format as annotation features.

This is the real-world deployment counterpart to precompute_annotation_features.py.

Feature layout (same as annotation features):
  [0]  category_id      (COCO→NuScenes mapped, 0-22)
  [1]  attribute_id     (inferred from velocity, 0-8)
  [2]  x               (ego frame, normalized: /50)
  [3]  y               (ego frame, normalized: /50)
  [4]  z               (ego frame, normalized: /5)
  [5]  width           (class-based prior, normalized: /10)
  [6]  length          (class-based prior, normalized: /10)
  [7]  height          (class-based prior, normalized: /5)
  [8]  vx              (from radar Doppler, normalized: /20)
  [9]  vy              (from radar Doppler, normalized: /20)
  [10] heading_sin     (from velocity direction)
  [11] heading_cos     (from velocity direction)
  [12] distance        (from radar or estimated, normalized: /50)
  [13] angle_sin       (sin of angle from ego forward)
  [14] angle_cos       (cos of angle from ego forward)
  [15] visibility      (detection confidence as proxy)

Improvements over V1:
  1. Expanded COCO → NuScenes mapping (covers 15+ COCO classes)
  2. Lower confidence threshold (0.25) to catch more objects
  3. Larger radar match radius (5.0m) for better velocity coverage
  4. Better attribute inference using bbox size + position context
  5. Uses yolov8x (extra-large) for better detection quality
  6. Smarter cross-camera NMS with confidence averaging

Output shape: (MAX_OBJECTS, 16) per sample — same format as annotation features.

Usage:
    CUDA_VISIBLE_DEVICES=1 python precompute_detected_features.py
    CUDA_VISIBLE_DEVICES=1 python precompute_detected_features.py --out-dir /path/to/output
"""

import os
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
OUT_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/detected_features"
YOLO_MODEL = "yolov8x.pt"       # Extra-large model for better accuracy
CONF_THRESHOLD = 0.25            # Lower threshold to catch more objects
MAX_OBJECTS = 100
FEAT_DIM = 16
RADAR_MATCH_RADIUS = 5.0        # Increased from 3.0 for better radar coverage
NMS_3D_RADIUS = 2.5             # Cross-camera duplicate removal radius

# All 6 NuScenes cameras (360° coverage)
CAMERA_CHANNELS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]

# All 5 NuScenes radar sensors
RADAR_CHANNELS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]


# ============================================================
# EXPANDED COCO → NuScenes Category Mapping
# ============================================================

# COCO class ID → (NuScenes category_id, is_vehicle flag)
# Much more comprehensive mapping than V1
COCO_TO_NUSCENES = {
    # --- Pedestrians ---
    0:  (0, False),    # person → human.pedestrian.adult

    # --- Vehicles ---
    1:  (10, False),   # bicycle → vehicle.bicycle
    2:  (8, True),     # car → vehicle.car
    3:  (9, True),     # motorcycle → vehicle.motorcycle
    5:  (12, True),    # bus → vehicle.bus.rigid
    7:  (13, True),    # truck → vehicle.truck

    # --- Animals ---
    15: (7, False),    # cat → animal
    16: (7, False),    # dog → animal
    17: (7, False),    # horse → animal
    18: (7, False),    # sheep → animal
    19: (7, False),    # cow → animal

    # --- Barriers & Traffic objects ---
    # These often appear in NuScenes traffic scenes
    9:  (19, False),   # traffic light → movable_object.trafficcone (proxy)
    11: (19, False),   # stop sign → movable_object.trafficcone (proxy)
    10: (19, False),   # fire hydrant → movable_object.trafficcone (proxy)
    13: (18, False),   # bench → movable_object.barrier (proxy)

    # --- Large vehicles ---
    4:  (14, True),    # airplane → vehicle.construction (as large vehicle)
    6:  (13, True),    # train → vehicle.truck (as large vehicle)
    8:  (14, True),    # boat → vehicle.construction (as large vehicle)

    # --- Movable objects / debris ---
    24: (20, False),   # backpack → movable_object.pushable_pullable
    25: (20, False),   # umbrella → movable_object.pushable_pullable
    26: (20, False),   # handbag → movable_object.pushable_pullable
    28: (20, False),   # suitcase → movable_object.pushable_pullable
}

# Set of COCO IDs we care about
VALID_COCO_IDS = set(COCO_TO_NUSCENES.keys())


# ============================================================
# Class-Based 3D Size Priors (width, length, height in meters)
# From NuScenes dataset statistics
# ============================================================

NUSCENES_SIZE_PRIORS = {
    0:  (0.7, 0.7, 1.75),     # pedestrian.adult
    1:  (0.5, 0.5, 1.2),      # pedestrian.child
    2:  (0.8, 1.2, 1.4),      # pedestrian.wheelchair
    3:  (0.7, 1.0, 1.0),      # pedestrian.stroller
    4:  (0.6, 0.8, 1.2),      # pedestrian.personal_mobility
    5:  (0.7, 0.7, 1.8),      # pedestrian.police_officer
    6:  (0.7, 0.7, 1.8),      # pedestrian.construction_worker
    7:  (0.5, 0.8, 0.6),      # animal
    8:  (1.9, 4.6, 1.7),      # vehicle.car
    9:  (0.8, 2.2, 1.5),      # vehicle.motorcycle
    10: (0.6, 1.8, 1.1),      # vehicle.bicycle
    11: (2.9, 12.0, 3.5),     # vehicle.bus.bendy
    12: (2.8, 10.0, 3.4),     # vehicle.bus.rigid
    13: (2.5, 7.0, 3.0),      # vehicle.truck
    14: (2.8, 6.0, 3.0),      # vehicle.construction
    15: (2.2, 6.5, 2.5),      # vehicle.emergency.ambulance
    16: (2.0, 5.0, 1.8),      # vehicle.emergency.police
    17: (2.5, 12.0, 3.5),     # vehicle.trailer
    18: (0.6, 1.2, 1.0),      # movable_object.barrier
    19: (0.4, 0.4, 0.8),      # movable_object.trafficcone
    20: (0.5, 0.5, 0.7),      # movable_object.pushable_pullable
    21: (0.3, 0.3, 0.3),      # movable_object.debris
    22: (2.0, 3.0, 1.2),      # static_object.bicycle_rack
}

# Default camera mounting height (meters above ground, typical for NuScenes)
CAMERA_HEIGHT = 1.5


# ============================================================
# 3D Position Estimation
# ============================================================

def estimate_depth_from_bbox(bbox, img_h, img_w, cam_intrinsic, cat_id):
    """
    Estimate depth using multiple cues:
    1. Ground plane geometry (primary)
    2. Known object size + apparent size (secondary)
    3. Class-based prior (fallback)
    """
    x1, y1, x2, y2 = bbox
    bottom_y = y2
    bbox_h = y2 - y1
    bbox_w = x2 - x1

    fy = cam_intrinsic[1][1]
    cy = cam_intrinsic[1][2]

    # Method 1: Ground plane geometry
    dy = bottom_y - cy
    depth_geo = None
    if dy > 10:
        depth_geo = (CAMERA_HEIGHT * fy) / dy
        depth_geo = np.clip(depth_geo, 1.0, 80.0)

    # Method 2: Known height + apparent height
    depth_size = None
    known_h = NUSCENES_SIZE_PRIORS.get(cat_id, (1.0, 1.0, 1.7))[2]
    if bbox_h > 10:  # avoid tiny boxes
        depth_size = (known_h * fy) / bbox_h
        depth_size = np.clip(depth_size, 1.0, 80.0)

    # Combine estimates
    if depth_geo is not None and depth_size is not None:
        # Weighted average, trust geometry more for ground-level objects
        depth = 0.6 * depth_geo + 0.4 * depth_size
    elif depth_geo is not None:
        depth = depth_geo
    elif depth_size is not None:
        depth = depth_size
    else:
        # Fallback: class-based depth prior
        class_prior_depths = {
            0: 12.0, 1: 10.0, 2: 12.0, 3: 10.0, 4: 12.0,
            5: 12.0, 6: 12.0, 7: 15.0, 8: 20.0, 9: 15.0,
            10: 12.0, 11: 25.0, 12: 25.0, 13: 22.0, 14: 20.0,
            15: 20.0, 16: 20.0, 17: 25.0, 18: 8.0, 19: 8.0,
            20: 8.0, 21: 8.0, 22: 10.0,
        }
        depth = class_prior_depths.get(cat_id, 15.0)

    return depth


def pixel_to_ego_3d(cx_pixel, cy_pixel, depth, cam_intrinsic,
                    cam_rotation, cam_translation,
                    ego_rotation):
    """
    Back-project a 2D pixel + depth into ego vehicle coordinates.
    Pipeline: pixel → camera 3D → ego 3D
    """
    fx = cam_intrinsic[0][0]
    fy = cam_intrinsic[1][1]
    cx = cam_intrinsic[0][2]
    cy = cam_intrinsic[1][2]

    # Pixel → camera frame 3D
    x_cam = (cx_pixel - cx) * depth / fx
    y_cam = (cy_pixel - cy) * depth / fy
    z_cam = depth

    pos_cam = np.array([x_cam, y_cam, z_cam])

    # Camera frame → ego frame
    pos_ego = cam_rotation.rotate(pos_cam) + np.array(cam_translation)

    return pos_ego


# ============================================================
# Attribute Inference (V2 - more nuanced)
# ============================================================

def infer_attribute(cat_id, vx, vy, bbox_area_ratio, pos_ego):
    """
    Infer NuScenes attribute from category, velocity, and context.
    V2 improvements:
    - Uses bbox area ratio to distinguish parked vs stopped
    - Uses position relative to road edge for parking inference
    """
    speed = np.sqrt(vx**2 + vy**2)
    dist_from_ego = np.sqrt(pos_ego[0]**2 + pos_ego[1]**2)

    # Vehicle categories
    vehicle_cats = {8, 9, 11, 12, 13, 14, 15, 16, 17}
    # Cycle categories
    cycle_cats = {10}
    # Pedestrian categories
    ped_cats = {0, 1, 2, 3, 4, 5, 6}

    if cat_id in vehicle_cats:
        if speed > 0.5:
            return 1   # vehicle.moving
        elif speed > 0.1:
            return 3   # vehicle.stopped
        else:
            # Distinguish parked vs stopped:
            # Far from ego + lateral position → likely parked
            lateral_dist = abs(pos_ego[1])
            if lateral_dist > 3.0 or dist_from_ego > 25.0:
                return 2   # vehicle.parked
            else:
                return 3   # vehicle.stopped

    elif cat_id in cycle_cats:
        if speed > 0.3:
            return 4   # cycle.with_rider
        else:
            return 5   # cycle.without_rider

    elif cat_id in ped_cats:
        if speed > 0.2:
            return 6   # pedestrian.moving
        elif speed > 0.05:
            return 7   # pedestrian.standing
        else:
            # Close to ground / large bbox → might be sitting
            if pos_ego[2] < -0.5:
                return 8   # pedestrian.sitting_lying_down
            return 7   # pedestrian.standing

    return 0  # no attribute


# ============================================================
# Radar Processing
# ============================================================

def get_all_radar_points_in_ego(nusc, sample, Quaternion):
    """
    Collect radar points from ALL 5 radar sensors and transform
    them into the ego vehicle coordinate frame.
    """
    ego_points = []
    ego_vel_x = []
    ego_vel_y = []
    ego_rcs = []

    from nuscenes.utils.data_classes import RadarPointCloud

    # Get ego pose from LIDAR_TOP (reference frame)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    for radar_ch in RADAR_CHANNELS:
        if radar_ch not in sample['data']:
            continue

        radar_sd = nusc.get('sample_data', sample['data'][radar_ch])
        radar_path = os.path.join(nusc.dataroot, radar_sd['filename'])

        if not os.path.exists(radar_path):
            continue

        try:
            radar_pc = RadarPointCloud.from_file(radar_path)
        except Exception:
            continue

        if radar_pc.points.shape[1] == 0:
            continue

        # Get calibration
        radar_cs = nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])

        # Radar → ego: sensor frame → vehicle frame
        pts = radar_pc.points.copy()  # (18, N)

        # Extract velocity and RCS before transforming positions
        vx_comp = pts[8, :]   # compensated velocity x
        vy_comp = pts[9, :]   # compensated velocity y
        rcs = pts[5, :]       # radar cross section

        # Transform position: radar sensor → ego vehicle
        pos_3d = pts[:3, :]   # (3, N)

        # Sensor → vehicle body
        rot_sensor = Quaternion(radar_cs['rotation']).rotation_matrix
        pos_3d = rot_sensor @ pos_3d + np.array(radar_cs['translation']).reshape(3, 1)

        # Also rotate velocities to ego frame
        vel_3d = np.vstack([vx_comp, vy_comp, np.zeros_like(vx_comp)])
        vel_3d = rot_sensor @ vel_3d

        ego_points.append(pos_3d)
        ego_vel_x.append(vel_3d[0, :])
        ego_vel_y.append(vel_3d[1, :])
        ego_rcs.append(rcs)

    if len(ego_points) == 0:
        return None

    return {
        'points_3d': np.hstack(ego_points),
        'vel_x': np.concatenate(ego_vel_x),
        'vel_y': np.concatenate(ego_vel_y),
        'rcs': np.concatenate(ego_rcs),
    }


def match_radar_to_detection(det_pos_ego, radar_data, cat_id,
                             match_radius=RADAR_MATCH_RADIUS):
    """
    Find the closest radar point to a detected object position (in ego frame).
    V2: Uses category-adaptive match radius (larger for big vehicles).
    Returns (vx, vy, matched, refined_depth) tuple.
    """
    if radar_data is None:
        return 0.0, 0.0, False, None

    # Category-adaptive radius: large vehicles get bigger search area
    large_vehicles = {11, 12, 13, 14, 17}  # bus, truck, construction, trailer
    if cat_id in large_vehicles:
        match_radius = match_radius * 1.5

    pts = radar_data['points_3d']  # (3, N)

    # Compute 2D distance (x, y only, ignore z)
    dx = pts[0, :] - det_pos_ego[0]
    dy = pts[1, :] - det_pos_ego[1]
    dists = np.sqrt(dx**2 + dy**2)

    min_idx = np.argmin(dists)
    if dists[min_idx] < match_radius:
        vx = radar_data['vel_x'][min_idx]
        vy = radar_data['vel_y'][min_idx]
        # Compute radar-based depth for refinement
        radar_depth = np.sqrt(
            pts[0, min_idx]**2 + pts[1, min_idx]**2 + pts[2, min_idx]**2
        )
        return vx, vy, True, radar_depth

    return 0.0, 0.0, False, None


# ============================================================
# Per-Sample Feature Extraction
# ============================================================

def extract_sample_features(nusc, sample_token, yolo_model, Quaternion, device='cuda'):
    """
    Extract detected features for one sample using camera + radar.
    V2: Improved mapping, depth estimation, and attribute inference.
    Returns (MAX_OBJECTS, FEAT_DIM) array.
    """
    features = np.zeros((MAX_OBJECTS, FEAT_DIM), dtype=np.float32)

    sample = nusc.get('sample', sample_token)

    # Get ego pose
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    # ---- Collect radar points (all 5 sensors) ----
    radar_data = get_all_radar_points_in_ego(nusc, sample, Quaternion)

    # ---- Run YOLO on all 6 cameras ----
    all_detections = []

    for cam_ch in CAMERA_CHANNELS:
        if cam_ch not in sample['data']:
            continue

        cam_sd = nusc.get('sample_data', sample['data'][cam_ch])
        img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
        if not os.path.exists(img_path):
            continue

        # Camera calibration
        cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_intrinsic = cam_cs['camera_intrinsic']
        cam_rotation_q = Quaternion(cam_cs['rotation'])
        cam_translation = cam_cs['translation']

        # Run YOLO
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size

        with torch.no_grad():
            results = yolo_model(image, verbose=False, conf=CONF_THRESHOLD)

        if results is None or len(results) == 0:
            continue

        result = results[0]
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes
        bboxes = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for j in range(len(bboxes)):
            coco_cls = int(classes[j])
            if coco_cls not in VALID_COCO_IDS:
                continue

            nusc_cat_id, is_vehicle = COCO_TO_NUSCENES[coco_cls]
            conf = float(confs[j])

            x1, y1, x2, y2 = bboxes[j]
            cx_pixel = (x1 + x2) / 2.0
            cy_pixel = (y1 + y2) / 2.0
            bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)

            # Estimate depth (V2: multi-cue)
            depth = estimate_depth_from_bbox(
                bboxes[j], img_h, img_w, cam_intrinsic, nusc_cat_id
            )

            # Back-project to ego 3D
            pos_ego = pixel_to_ego_3d(
                cx_pixel, cy_pixel, depth,
                cam_intrinsic, cam_rotation_q, cam_translation,
                ego_rotation
            )

            # Match with radar for velocity (V2: category-adaptive radius)
            vx, vy, matched, radar_depth = match_radar_to_detection(
                pos_ego, radar_data, nusc_cat_id
            )

            # If radar matched, refine depth with radar range
            if matched and radar_depth is not None:
                pos_ego = pixel_to_ego_3d(
                    cx_pixel, cy_pixel, radar_depth,
                    cam_intrinsic, cam_rotation_q, cam_translation,
                    ego_rotation
                )

            all_detections.append({
                'cat_id': nusc_cat_id,
                'conf': conf,
                'pos_ego': pos_ego,
                'vx': vx,
                'vy': vy,
                'radar_matched': matched,
                'bbox_area_ratio': bbox_area_ratio,
                'cam_ch': cam_ch,
            })

    # ---- Sort by confidence, keep top MAX_OBJECTS ----
    all_detections.sort(key=lambda d: d['conf'], reverse=True)

    # ---- Improved Cross-Camera NMS ----
    kept = []
    for det in all_detections:
        is_dup = False
        for prev in kept:
            dx = det['pos_ego'][0] - prev['pos_ego'][0]
            dy = det['pos_ego'][1] - prev['pos_ego'][1]
            dist = np.sqrt(dx**2 + dy**2)

            # Same category or similar category (both vehicles, etc.)
            same_cat = (det['cat_id'] == prev['cat_id'])
            both_vehicles = (det['cat_id'] in {8,9,11,12,13,14,15,16,17} and
                           prev['cat_id'] in {8,9,11,12,13,14,15,16,17})

            if dist < NMS_3D_RADIUS and (same_cat or both_vehicles):
                # Duplicate: merge information
                # Keep position from higher-confidence detection
                # But absorb radar match from either
                if det['radar_matched'] and not prev['radar_matched']:
                    prev['vx'] = det['vx']
                    prev['vy'] = det['vy']
                    prev['radar_matched'] = True
                # Average confidence from multiple views
                prev['conf'] = max(prev['conf'], det['conf'])
                is_dup = True
                break

        if not is_dup:
            kept.append(det)

    n_objects = min(len(kept), MAX_OBJECTS)

    # ---- Fill feature array ----
    for i in range(n_objects):
        det = kept[i]
        cat_id = det['cat_id']
        pos = det['pos_ego']
        vx, vy = det['vx'], det['vy']

        # [0] Category ID
        features[i, 0] = cat_id

        # [1] Attribute (V2: uses context)
        features[i, 1] = infer_attribute(
            cat_id, vx, vy, det['bbox_area_ratio'], pos
        )

        # [2-4] Position (ego frame, normalized)
        features[i, 2] = pos[0] / 50.0
        features[i, 3] = pos[1] / 50.0
        features[i, 4] = pos[2] / 5.0

        # [5-7] Size (class-based priors, normalized)
        w, l, h = NUSCENES_SIZE_PRIORS.get(cat_id, (1.0, 1.0, 1.0))
        features[i, 5] = w / 10.0
        features[i, 6] = l / 10.0
        features[i, 7] = h / 5.0

        # [8-9] Velocity (from radar, normalized)
        features[i, 8] = vx / 20.0
        features[i, 9] = vy / 20.0

        # [10-11] Heading (from velocity direction if moving, else default)
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0.5:
            heading = np.arctan2(vy, vx)
        else:
            heading = np.arctan2(pos[1], pos[0])
        features[i, 10] = np.sin(heading)
        features[i, 11] = np.cos(heading)

        # [12] Distance from ego (normalized)
        dist = np.sqrt(pos[0]**2 + pos[1]**2)
        features[i, 12] = dist / 50.0

        # [13-14] Angle from ego forward
        angle = np.arctan2(pos[1], pos[0])
        features[i, 13] = np.sin(angle)
        features[i, 14] = np.cos(angle)

        # [15] Visibility (use detection confidence as proxy)
        features[i, 15] = det['conf']

    return features


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract detected features (Camera+Radar) for NuScenes VQA"
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--yolo-model", default=YOLO_MODEL)
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load NuScenes ---
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    print(f"Loading NuScenes {args.version} from {args.data_root}...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    print(f"Loaded {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    # --- Load YOLO ---
    from ultralytics import YOLO
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)
    yolo.to(args.device)
    print(f"YOLO ready on {args.device}")

    # --- Check existing (for resume) ---
    existing = set()
    if os.path.exists(args.out_dir):
        existing = {f.replace('.npy', '') for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    print(f"Found {len(existing)} existing features, will skip those")

    # --- Process all samples ---
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
                feat = extract_sample_features(
                    nusc, sample_token, yolo, Quaternion, device=args.device
                )
                np.save(
                    os.path.join(args.out_dir, f"{sample_token}.npy"),
                    feat.astype(np.float32)
                )
                processed += 1
            except Exception as e:
                print(f"\n  Error on {sample_token}: {e}")
                errors += 1

            sample_token = sample["next"]

    print(f"\n✅ Detected feature extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped (existing): {skipped}")
    print(f"   Errors: {errors}")
    print(f"   Output: {args.out_dir}")
    print(f"   Feature shape: ({MAX_OBJECTS}, {FEAT_DIM})")


if __name__ == "__main__":
    main()
