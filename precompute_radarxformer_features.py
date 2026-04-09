#!/usr/bin/env python3
"""
Pre-compute RadarXFormer-Inspired features for NuScenes VQA.

Adapts key principles from RadarXFormer (2603.14822v1):
  1. CLIP visual feature extraction from detection image crops
     (analogous to multi-scale image feature maps)
  2. Attention-weighted multi-radar aggregation
     (analogous to cross-dimension deformable attention)
  3. CLIP-assisted category refinement
     (leverages rich image semantics for better classification)
  4. Multi-view triangulation for depth
     (combines information across camera views)
  5. Radar-primary depth when available
     (radar provides direct range measurement)

Output shape: (MAX_OBJECTS, 32) per sample
  Dims  0-15: Enhanced structured features (same layout as annotation features)
  Dims 16-31: CLIP visual features (PCA-compressed from 512-d)

Usage:
    # Step 1: Fit PCA on a subset (fast, ~5 min)
    CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features.py --mode fit-pca

    # Step 2: Extract all features (GPU, ~2-3 hours)
    CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features.py --mode extract

    # Or do both in one go:
    CUDA_VISIBLE_DEVICES=0 python precompute_radarxformer_features.py --mode all
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "/media/nas_mount/anwar2/experiment/dataset/nuscenes"
VERSION = "v1.0-trainval"
OUT_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/radarxf_features_v2"
YOLO_MODEL = "yolov8m.pt"
CONF_THRESHOLD = 0.15          # Lower threshold to catch small objects (cones, barriers)
MAX_OBJECTS = 100
STRUCT_DIM = 16                # Structured feature dimensions
CLIP_RAW_DIM = 512             # CLIP ViT-B/32 output dimension
CLIP_PCA_DIM = 32              # PCA-compressed CLIP dimension (doubled for richer features)
FEAT_DIM = STRUCT_DIM + CLIP_PCA_DIM  # 48 total

# Radar aggregation parameters (inspired by RadarXFormer's deformable attention)
RADAR_SEARCH_RADIUS = 5.0      # meters — wider than detected_features (3.0)
RADAR_MIN_RCS = -20.0          # minimum RCS to consider
RADAR_TOPK = 5                 # max radar points to aggregate per detection

# NuScenes cameras and radars
CAMERA_CHANNELS = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]
RADAR_CHANNELS = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
]

# PCA model path
PCA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "clip_pca_model_v2.pkl")


# ============================================================
# COCO → NuScenes Category Mapping
# ============================================================

COCO_TO_NUSCENES = {
    # === Core vehicle/pedestrian categories (high confidence) ===
    0:  (0, False),    # person → pedestrian.adult
    1:  (10, False),   # bicycle → vehicle.bicycle
    2:  (8, True),     # car → vehicle.car
    3:  (9, True),     # motorcycle → vehicle.motorcycle
    5:  (12, True),    # bus → vehicle.bus.rigid
    7:  (13, True),    # truck → vehicle.truck
    # === Animals ===
    15: (7, False),    # cat → animal
    16: (7, False),    # dog → animal
    17: (7, False),    # horse → animal
    # === Construction & emergency vehicles ===
    8:  (14, True),    # boat → vehicle.construction (approximate)
    # === Road objects (critical for exist/count questions) ===
    9:  (19, False),   # traffic light → trafficcone (road furniture)
    10: (18, False),   # fire hydrant → barrier (road obstacle)
    11: (19, False),   # stop sign → trafficcone (road sign)
    13: (20, False),   # bench → pushable_pullable (street furniture)
    # === Movable objects ===
    24: (20, False),   # backpack → pushable_pullable
    25: (20, False),   # umbrella → pushable_pullable
    28: (20, False),   # suitcase → pushable_pullable
    56: (20, False),   # chair → pushable_pullable
    62: (21, False),   # tv → debris (approximate)
}
VALID_COCO_IDS = set(COCO_TO_NUSCENES.keys())

# NuScenes category names for CLIP zero-shot verification
NUSCENES_CATEGORY_PROMPTS = {
    0: "a pedestrian or person walking",
    1: "a child pedestrian",
    7: "an animal on the road",
    8: "a car or sedan",
    9: "a motorcycle",
    10: "a bicycle",
    12: "a bus",
    13: "a truck",
    14: "a construction vehicle",
    18: "a road barrier or guardrail",
    19: "a traffic cone or road sign",
    20: "a movable object on the road",
    21: "debris or scattered objects on the road",
}

# Size priors (width, length, height in meters)
NUSCENES_SIZE_PRIORS = {
    0:  (0.7, 0.7, 1.75),   1:  (0.5, 0.5, 1.2),
    2:  (0.8, 1.2, 1.4),    3:  (0.7, 1.0, 1.0),
    4:  (0.6, 0.8, 1.2),    5:  (0.7, 0.7, 1.8),
    6:  (0.7, 0.7, 1.8),    7:  (0.5, 0.8, 0.6),
    8:  (1.9, 4.6, 1.7),    9:  (0.8, 2.2, 1.5),
    10: (0.6, 1.8, 1.1),    11: (2.9, 12.0, 3.5),
    12: (2.8, 10.0, 3.4),   13: (2.5, 7.0, 3.0),
    14: (2.8, 6.0, 3.0),    15: (2.2, 6.5, 2.5),
    16: (2.0, 5.0, 1.8),    17: (2.5, 12.0, 3.5),
    18: (0.6, 1.2, 1.0),    19: (0.4, 0.4, 0.8),
    20: (0.5, 0.5, 0.7),    21: (0.3, 0.3, 0.3),
    22: (2.0, 3.0, 1.2),
}

CAMERA_HEIGHT = 1.5  # meters above ground


# ============================================================
# CLIP Feature Extractor
# ============================================================

class CLIPFeatureExtractor:
    """Extracts CLIP visual features from image crops."""

    def __init__(self, device='cuda'):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

        # Pre-compute text embeddings for category verification
        self.category_text_features = {}
        with torch.no_grad():
            for cat_id, prompt in NUSCENES_CATEGORY_PROMPTS.items():
                tokens = clip.tokenize([prompt]).to(device)
                feat = self.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                self.category_text_features[cat_id] = feat

        print(f"  [CLIP] Loaded ViT-B/32, {len(self.category_text_features)} category prompts")

    def extract_crop_features(self, image_pil, bboxes):
        """
        Extract CLIP features for a batch of image crops.

        Args:
            image_pil: PIL Image
            bboxes: list of (x1,y1,x2,y2) tuples

        Returns:
            features: (N, 512) numpy array
        """
        if len(bboxes) == 0:
            return np.array([]).reshape(0, CLIP_RAW_DIM)

        crops = []
        img_w, img_h = image_pil.size

        for (x1, y1, x2, y2) in bboxes:
            # Clamp and ensure minimum size
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))

            if x2 - x1 < 5 or y2 - y1 < 5:
                # Too small, use center 32x32
                cx, cy = (x1+x2)//2, (y1+y2)//2
                x1 = max(0, cx-16)
                y1 = max(0, cy-16)
                x2 = min(img_w, cx+16)
                y2 = min(img_h, cy+16)

            crop = image_pil.crop((x1, y1, x2, y2))
            crop = self.preprocess(crop)
            crops.append(crop)

        # Batch inference
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features.float()  # ensure float32

        return features.cpu().numpy()

    def verify_category(self, clip_feature, yolo_cat_id):
        """
        CLIP-assisted category verification.
        Returns the category ID with highest CLIP similarity.

        Inspired by RadarXFormer's principle of using complementary
        modalities to improve accuracy.
        """
        if yolo_cat_id not in self.category_text_features:
            return yolo_cat_id

        # Determine target dtype to match CLIP's precision (often float16 / Half)
        target_dtype = next(iter(self.category_text_features.values())).dtype

        clip_feat = torch.from_numpy(clip_feature).to(self.device, dtype=target_dtype)
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

        # Compare against all relevant categories
        best_cat = yolo_cat_id
        best_sim = -1.0

        for cat_id, text_feat in self.category_text_features.items():
            sim = (clip_feat @ text_feat.T).item()
            if sim > best_sim:
                best_sim = sim
                best_cat = cat_id

        # Only override YOLO if CLIP is very confident AND disagrees
        yolo_sim = -1.0
        if yolo_cat_id in self.category_text_features:
            yolo_sim = (clip_feat @ self.category_text_features[yolo_cat_id].T).item()

        # Override only if best_cat != yolo_cat AND similarity difference > 0.05
        if best_cat != yolo_cat_id and (best_sim - yolo_sim) > 0.05:
            return best_cat

        return yolo_cat_id


# ============================================================
# Attention-Weighted Radar Aggregation
# ============================================================

def attention_radar_aggregation(det_pos_ego, radar_data,
                                 search_radius=RADAR_SEARCH_RADIUS,
                                 topk=RADAR_TOPK):
    """
    Attention-weighted aggregation of radar points near a detection.

    Inspired by RadarXFormer's multi-scale deformable cross-attention:
    instead of hard nearest-neighbor matching, we softly aggregate
    information from multiple radar returns using learned-quality weights.

    Weights based on:
      - Inverse distance (closer = higher weight)
      - RCS strength (stronger reflection = more likely the object)
      - Height consistency (radar z close to expected object z)

    Returns: (vx, vy, matched, agg_depth, confidence)
    """
    if radar_data is None:
        return 0.0, 0.0, False, 0.0, 0.0

    pts = radar_data['points_3d']  # (3, N)
    if pts.shape[1] == 0:
        return 0.0, 0.0, False, 0.0, 0.0

    # Compute 3D distances
    dx = pts[0, :] - det_pos_ego[0]
    dy = pts[1, :] - det_pos_ego[1]
    dz = pts[2, :] - det_pos_ego[2] if len(det_pos_ego) > 2 else np.zeros_like(dx)
    dists_2d = np.sqrt(dx**2 + dy**2)

    # Filter by search radius
    mask = dists_2d < search_radius
    if not mask.any():
        return 0.0, 0.0, False, 0.0, 0.0

    # Get candidate points
    indices = np.where(mask)[0]
    candidate_dists = dists_2d[indices]
    candidate_vx = radar_data['vel_x'][indices]
    candidate_vy = radar_data['vel_y'][indices]
    candidate_rcs = radar_data['rcs'][indices]
    candidate_pts = pts[:, indices]

    # Keep top-K closest
    if len(indices) > topk:
        topk_idx = np.argsort(candidate_dists)[:topk]
        candidate_dists = candidate_dists[topk_idx]
        candidate_vx = candidate_vx[topk_idx]
        candidate_vy = candidate_vy[topk_idx]
        candidate_rcs = candidate_rcs[topk_idx]
        candidate_pts = candidate_pts[:, topk_idx]

    # Compute attention weights (soft, differentiable-style)
    # Weight 1: Inverse distance (Gaussian kernel)
    sigma = search_radius / 3.0
    dist_weights = np.exp(-candidate_dists**2 / (2 * sigma**2))

    # Weight 2: RCS strength (normalized sigmoid-like)
    rcs_normalized = (candidate_rcs - RADAR_MIN_RCS) / (40.0 - RADAR_MIN_RCS)  # normalize to [0,1]
    rcs_weights = np.clip(rcs_normalized, 0.0, 1.0)

    # Combined attention weights
    attention = dist_weights * (0.6 + 0.4 * rcs_weights)
    attention_sum = attention.sum()

    if attention_sum < 1e-8:
        return 0.0, 0.0, False, 0.0, 0.0

    # Normalize attention
    attention = attention / attention_sum

    # Weighted aggregation of velocity
    agg_vx = (attention * candidate_vx).sum()
    agg_vy = (attention * candidate_vy).sum()

    # Weighted aggregation of depth (range from ego)
    candidate_ranges = np.sqrt(candidate_pts[0]**2 + candidate_pts[1]**2)
    agg_depth = (attention * candidate_ranges).sum()

    # Confidence = how concentrated the attention is (higher = more certain)
    confidence = np.max(attention)

    return float(agg_vx), float(agg_vy), True, float(agg_depth), float(confidence)


# ============================================================
# Multi-View Triangulation
# ============================================================

def triangulate_multi_view(detections_per_camera, match_threshold=0.15):
    """
    When the same object is detected in multiple camera views,
    triangulate its true 3D position.

    Inspired by RadarXFormer's principle of combining information
    across different views for more accurate 3D estimation.

    Returns: list of merged detections with refined positions.
    """
    if len(detections_per_camera) <= 1:
        # Flatten single-camera detections
        merged = []
        for cam_dets in detections_per_camera:
            merged.extend(cam_dets)
        return merged

    # Collect all detections from all cameras
    all_dets = []
    for cam_idx, cam_dets in enumerate(detections_per_camera):
        for det in cam_dets:
            det['_cam_idx'] = cam_idx
            all_dets.append(det)

    # Sort by confidence (highest first)
    all_dets.sort(key=lambda d: d['conf'], reverse=True)

    # Greedy merge: for each detection, check if a previous detection
    # of the same category is nearby in 3D
    merged = []
    for det in all_dets:
        found_match = False
        for prev in merged:
            if det['cat_id'] != prev['cat_id']:
                continue

            dx = det['pos_ego'][0] - prev['pos_ego'][0]
            dy = det['pos_ego'][1] - prev['pos_ego'][1]
            dist = np.sqrt(dx**2 + dy**2)

            # Dynamic threshold based on distance from ego
            ego_dist = np.sqrt(prev['pos_ego'][0]**2 + prev['pos_ego'][1]**2)
            thresh = max(2.0, ego_dist * match_threshold)

            if dist < thresh:
                # Found a match — triangulate by weighted average
                w_prev = prev['conf']
                w_curr = det['conf']
                w_sum = w_prev + w_curr

                prev['pos_ego'] = (
                    prev['pos_ego'] * w_prev + det['pos_ego'] * w_curr
                ) / w_sum

                # Keep the better confidence
                prev['conf'] = max(prev['conf'], det['conf'])

                # Prefer radar-matched detection
                if det.get('radar_matched') and not prev.get('radar_matched'):
                    prev['vx'] = det['vx']
                    prev['vy'] = det['vy']
                    prev['radar_matched'] = True

                # Accumulate CLIP features (will average later)
                if 'clip_features_list' not in prev:
                    prev['clip_features_list'] = [prev.get('clip_feature')]
                prev['clip_features_list'].append(det.get('clip_feature'))

                prev['_num_views'] = prev.get('_num_views', 1) + 1
                found_match = True
                break

        if not found_match:
            det['_num_views'] = 1
            merged.append(det)

    # Average accumulated CLIP features for multi-view detections
    for det in merged:
        if 'clip_features_list' in det:
            valid_feats = [f for f in det['clip_features_list'] if f is not None]
            if valid_feats:
                det['clip_feature'] = np.mean(valid_feats, axis=0)
            del det['clip_features_list']

    return merged


# ============================================================
# 3D Position Estimation
# ============================================================

def estimate_depth_from_bbox(bbox_bottom_y, img_h, cam_intrinsic, cat_id):
    """Estimate depth using ground plane assumption."""
    fy = cam_intrinsic[1][1]
    cy = cam_intrinsic[1][2]
    dy = bbox_bottom_y - cy

    if dy > 10:
        depth = (CAMERA_HEIGHT * fy) / dy
        depth = np.clip(depth, 1.0, 80.0)
        return depth

    class_prior_depths = {
        0: 12.0, 1: 10.0, 2: 12.0, 3: 10.0, 4: 12.0,
        5: 12.0, 6: 12.0, 7: 15.0, 8: 20.0, 9: 15.0,
        10: 12.0, 11: 25.0, 12: 25.0, 13: 22.0, 14: 20.0,
        15: 20.0, 16: 20.0, 17: 25.0, 18: 8.0, 19: 8.0,
        20: 8.0, 21: 8.0, 22: 10.0,
    }
    return class_prior_depths.get(cat_id, 15.0)


def pixel_to_ego_3d(cx_pixel, cy_pixel, depth, cam_intrinsic,
                    cam_rotation, cam_translation, ego_rotation=None):
    """Back-project pixel + depth into ego vehicle coordinates."""
    fx = cam_intrinsic[0][0]
    fy = cam_intrinsic[1][1]
    cx = cam_intrinsic[0][2]
    cy = cam_intrinsic[1][2]

    x_cam = (cx_pixel - cx) * depth / fx
    y_cam = (cy_pixel - cy) * depth / fy
    z_cam = depth

    pos_cam = np.array([x_cam, y_cam, z_cam])
    pos_ego = cam_rotation.rotate(pos_cam) + np.array(cam_translation)

    return pos_ego


# ============================================================
# Radar Processing
# ============================================================

def get_all_radar_points_in_ego(nusc, sample, Quaternion):
    """Collect radar points from ALL 5 sensors into ego frame."""
    from nuscenes.utils.data_classes import RadarPointCloud

    ego_points = []
    ego_vel_x = []
    ego_vel_y = []
    ego_rcs = []

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

        radar_cs = nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])

        pts = radar_pc.points.copy()
        vx_comp = pts[8, :]
        vy_comp = pts[9, :]
        rcs = pts[5, :]
        pos_3d = pts[:3, :]

        rot_sensor = Quaternion(radar_cs['rotation']).rotation_matrix
        pos_3d = rot_sensor @ pos_3d + np.array(radar_cs['translation']).reshape(3, 1)

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


# ============================================================
# Attribute Inference
# ============================================================

def infer_attribute(cat_id, vx, vy):
    """Infer NuScenes attribute from category and velocity."""
    speed = np.sqrt(vx**2 + vy**2)
    vehicle_cats = {8, 9, 11, 12, 13, 14, 15, 16, 17}
    cycle_cats = {10}
    ped_cats = {0, 1, 2, 3, 4, 5, 6}

    if cat_id in vehicle_cats:
        if speed > 0.5: return 1
        elif speed > 0.1: return 3
        else: return 2
    elif cat_id in cycle_cats:
        return 4 if speed > 0.3 else 5
    elif cat_id in ped_cats:
        return 6 if speed > 0.3 else 7
    return 0


# ============================================================
# Per-Sample Feature Extraction
# ============================================================

def extract_sample_features(nusc, sample_token, yolo_model, clip_extractor,
                            pca_model, Quaternion, device='cuda'):
    """
    Extract RadarXFormer-inspired features for one sample.
    Returns (MAX_OBJECTS, FEAT_DIM) array — 32 dims per object.
    """
    features = np.zeros((MAX_OBJECTS, FEAT_DIM), dtype=np.float32)

    sample = nusc.get('sample', sample_token)

    # Get ego pose
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    # ---- Collect radar points (all 5 sensors) ----
    radar_data = get_all_radar_points_in_ego(nusc, sample, Quaternion)

    # ---- Run YOLO + CLIP on all 6 cameras ----
    detections_per_camera = []

    for cam_ch in CAMERA_CHANNELS:
        if cam_ch not in sample['data']:
            detections_per_camera.append([])
            continue

        cam_sd = nusc.get('sample_data', sample['data'][cam_ch])
        img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
        if not os.path.exists(img_path):
            detections_per_camera.append([])
            continue

        cam_cs = nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_intrinsic = cam_cs['camera_intrinsic']
        cam_rotation_q = Quaternion(cam_cs['rotation'])
        cam_translation = cam_cs['translation']

        # Run YOLO
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size

        with torch.no_grad():
            results = yolo_model(image, verbose=False, conf=CONF_THRESHOLD)

        if results is None or len(results) == 0 or not hasattr(results[0], 'boxes') \
                or results[0].boxes is None or len(results[0].boxes) == 0:
            detections_per_camera.append([])
            continue

        boxes = results[0].boxes
        bboxes = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # Filter to valid NuScenes classes
        valid_mask = np.array([int(c) in VALID_COCO_IDS for c in classes])
        if not valid_mask.any():
            detections_per_camera.append([])
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_bboxes = bboxes[valid_indices]
        valid_confs = confs[valid_indices]
        valid_classes = classes[valid_indices]

        # Extract CLIP features for all valid detections in this camera
        clip_bboxes = [(b[0], b[1], b[2], b[3]) for b in valid_bboxes]
        clip_features = clip_extractor.extract_crop_features(image, clip_bboxes)

        cam_detections = []
        for j in range(len(valid_indices)):
            coco_cls = int(valid_classes[j])
            nusc_cat_id, is_vehicle = COCO_TO_NUSCENES[coco_cls]
            conf = float(valid_confs[j])

            x1, y1, x2, y2 = valid_bboxes[j]
            cx_pixel = (x1 + x2) / 2.0
            cy_pixel = (y1 + y2) / 2.0
            bottom_y = y2

            # ---- CLIP-assisted category refinement ----
            clip_feat = clip_features[j] if j < len(clip_features) else None
            if clip_feat is not None:
                refined_cat = clip_extractor.verify_category(clip_feat, nusc_cat_id)
            else:
                refined_cat = nusc_cat_id

            # ---- Estimate depth ----
            depth = estimate_depth_from_bbox(bottom_y, img_h, cam_intrinsic, refined_cat)

            # ---- Back-project to ego 3D ----
            pos_ego = pixel_to_ego_3d(
                cx_pixel, cy_pixel, depth,
                cam_intrinsic, cam_rotation_q, cam_translation, ego_rotation
            )

            # ---- Attention-weighted radar aggregation ----
            vx, vy, matched, agg_depth, radar_conf = attention_radar_aggregation(
                pos_ego, radar_data
            )

            # If radar matched, refine depth with radar range (radar-primary depth)
            if matched and agg_depth > 0:
                # Re-project with radar-refined depth
                pos_ego = pixel_to_ego_3d(
                    cx_pixel, cy_pixel, agg_depth,
                    cam_intrinsic, cam_rotation_q, cam_translation, ego_rotation
                )

            cam_detections.append({
                'cat_id': refined_cat,
                'conf': conf,
                'pos_ego': pos_ego,
                'vx': vx,
                'vy': vy,
                'radar_matched': matched,
                'radar_conf': radar_conf,
                'clip_feature': clip_feat,
            })

        detections_per_camera.append(cam_detections)

    # ---- Multi-view triangulation ----
    merged = triangulate_multi_view(detections_per_camera)

    # ---- Sort by confidence, keep top MAX_OBJECTS ----
    merged.sort(key=lambda d: d['conf'], reverse=True)
    n_objects = min(len(merged), MAX_OBJECTS)

    # ---- Fill feature array ----
    for i in range(n_objects):
        det = merged[i]
        cat_id = det['cat_id']
        pos = det['pos_ego']
        vx, vy = det['vx'], det['vy']

        # --- Dims 0-15: Structured features (same layout as annotation features) ---

        features[i, 0] = cat_id
        features[i, 1] = infer_attribute(cat_id, vx, vy)
        features[i, 2] = pos[0] / 50.0
        features[i, 3] = pos[1] / 50.0
        features[i, 4] = pos[2] / 5.0

        w, l, h = NUSCENES_SIZE_PRIORS.get(cat_id, (1.0, 1.0, 1.0))
        features[i, 5] = w / 10.0
        features[i, 6] = l / 10.0
        features[i, 7] = h / 5.0

        features[i, 8] = vx / 20.0
        features[i, 9] = vy / 20.0

        speed = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx) if speed > 0.5 else np.arctan2(pos[1], pos[0])
        features[i, 10] = np.sin(heading)
        features[i, 11] = np.cos(heading)

        dist = np.sqrt(pos[0]**2 + pos[1]**2)
        features[i, 12] = dist / 50.0

        angle = np.arctan2(pos[1], pos[0])
        features[i, 13] = np.sin(angle)
        features[i, 14] = np.cos(angle)

        # Confidence: combine detection conf with radar match confidence
        radar_bonus = 0.1 if det.get('radar_matched', False) else 0.0
        multiview_bonus = 0.05 * min(det.get('_num_views', 1) - 1, 3)
        features[i, 15] = min(1.0, det['conf'] + radar_bonus + multiview_bonus)

        # --- Dims 16-47: CLIP visual features (PCA-compressed + normalized) ---

        clip_feat = det.get('clip_feature')
        if clip_feat is not None and pca_model is not None:
            pca = pca_model['pca'] if isinstance(pca_model, dict) else pca_model
            clip_compressed = pca.transform(clip_feat.reshape(1, -1))[0]
            # Pre-normalize using dataset-wide stats
            if isinstance(pca_model, dict) and 'clip_mean' in pca_model:
                clip_compressed = (clip_compressed - pca_model['clip_mean']) / pca_model['clip_std']
                clip_compressed = np.clip(clip_compressed, -3.0, 3.0)  # prevent outliers
            features[i, STRUCT_DIM:STRUCT_DIM + CLIP_PCA_DIM] = clip_compressed.astype(np.float32)

    return features


# ============================================================
# PCA Fitting
# ============================================================

def fit_pca(nusc, yolo_model, clip_extractor, Quaternion, device='cuda',
            n_samples=850, n_components=CLIP_PCA_DIM):
    """
    Fit PCA on CLIP features from ALL scenes and ALL 6 cameras.
    Also computes and saves per-component normalization stats (mean, std)
    so CLIP features can be pre-normalized at extraction time.
    """
    from sklearn.decomposition import PCA

    n_samples = min(n_samples, len(nusc.scene))
    print(f"\n=== Fitting PCA on {n_samples} scenes, ALL 6 cameras ===")

    all_clip_features = []
    sample_tokens = [s['first_sample_token'] for s in nusc.scene[:n_samples]]

    for token in tqdm(sample_tokens, desc="PCA fitting"):
        sample = nusc.get('sample', token)

        for cam_ch in CAMERA_CHANNELS:  # ALL 6 cameras
            if cam_ch not in sample['data']:
                continue

            cam_sd = nusc.get('sample_data', sample['data'][cam_ch])
            img_path = os.path.join(nusc.dataroot, cam_sd['filename'])
            if not os.path.exists(img_path):
                continue

            image = Image.open(img_path).convert('RGB')
            with torch.no_grad():
                results = yolo_model(image, verbose=False, conf=CONF_THRESHOLD)

            if results is None or len(results) == 0:
                continue

            result = results[0]
            if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
                continue

            bboxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            valid_mask = np.array([int(c) in VALID_COCO_IDS for c in classes])

            if not valid_mask.any():
                continue

            valid_bboxes = bboxes[valid_mask]
            clip_bboxes = [(b[0], b[1], b[2], b[3]) for b in valid_bboxes]
            clip_features = clip_extractor.extract_crop_features(image, clip_bboxes)

            all_clip_features.append(clip_features)

    all_clip_features = np.vstack(all_clip_features)
    print(f"  Collected {len(all_clip_features)} CLIP features for PCA")

    pca = PCA(n_components=n_components)
    pca.fit(all_clip_features)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {n_components} components explain {explained_var*100:.1f}% variance")

    # Compute per-component normalization stats from the training set
    transformed = pca.transform(all_clip_features)
    clip_mean = transformed.mean(axis=0)
    clip_std = transformed.std(axis=0)
    clip_std[clip_std < 1e-6] = 1.0  # prevent division by zero

    # Save PCA model + normalization stats together
    save_data = {
        'pca': pca,
        'clip_mean': clip_mean,
        'clip_std': clip_std,
    }
    with open(PCA_MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"  Saved PCA model + normalization stats to {PCA_MODEL_PATH}")
    print(f"  CLIP component stats: mean range [{clip_mean.min():.3f}, {clip_mean.max():.3f}], "
          f"std range [{clip_std.min():.3f}, {clip_std.max():.3f}]")
    return save_data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RadarXFormer-inspired feature extraction for NuScenes VQA"
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--yolo-model", default=YOLO_MODEL)
    parser.add_argument("--conf-thresh", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=['fit-pca', 'extract', 'all'], default='all',
                        help="fit-pca: only fit PCA model; extract: extract features; all: both")
    parser.add_argument("--pca-samples", type=int, default=850,
                        help="Number of scenes to use for PCA fitting (default: all)")
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

    # --- Load CLIP ---
    clip_extractor = CLIPFeatureExtractor(device=args.device)

    # --- Step 1: Fit PCA ---
    pca_model = None
    if args.mode in ('fit-pca', 'all'):
        pca_model = fit_pca(nusc, yolo, clip_extractor, Quaternion,
                            device=args.device, n_samples=args.pca_samples)

    if args.mode == 'fit-pca':
        print("\nPCA fitting complete. Run with --mode extract to extract features.")
        return

    # --- Load PCA model ---
    if pca_model is None:
        if os.path.exists(PCA_MODEL_PATH):
            with open(PCA_MODEL_PATH, 'rb') as f:
                loaded = pickle.load(f)
            # Handle both old format (raw PCA) and new format (dict)
            if isinstance(loaded, dict):
                pca_model = loaded
            else:
                pca_model = {'pca': loaded, 'clip_mean': None, 'clip_std': None}
                print("  WARNING: Old PCA format without normalization stats. Re-run with --mode fit-pca.")
            print(f"Loaded PCA model from {PCA_MODEL_PATH}")
        else:
            print(f"ERROR: PCA model not found at {PCA_MODEL_PATH}")
            print("Run with --mode fit-pca first.")
            return

    # --- Step 2: Extract features ---
    existing = set()
    if os.path.exists(args.out_dir):
        existing = {f.replace('.npy', '') for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    print(f"Found {len(existing)} existing features, will skip")

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
                    nusc, sample_token, yolo, clip_extractor,
                    pca_model, Quaternion, device=args.device
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

    print(f"\n✅ RadarXFormer feature extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped (existing): {skipped}")
    print(f"   Errors: {errors}")
    print(f"   Output: {args.out_dir}")
    print(f"   Feature shape: ({MAX_OBJECTS}, {FEAT_DIM})")


if __name__ == "__main__":
    main()
