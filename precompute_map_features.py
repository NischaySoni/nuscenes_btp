#!/usr/bin/env python3
"""
Pre-compute MAP CONTEXT features for each detected object in NuScenes.

For each object at position (x, y) in global coordinates, queries the
NuScenes semantic map to extract spatial context:
  - Is on drivable area, walkway, road segment, parking area?
  - Near crosswalk or stop line?
  - Distance to road edge, crosswalk
  - Lane direction at that point
  - Number of nearby lanes

Output shape: (MAX_OBJECTS, MAP_DIM) per sample — aligned with RadarXF features
so each object gets its own map context.

Usage:
    python precompute_map_features.py
    python precompute_map_features.py --data-root /path/to/nuscenes
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
OUT_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/map_features"
RXF_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/radarxf_features_v2"
MAX_OBJECTS = 100
MAP_DIM = 11  # number of map context features per object

# Map layer names for querying
POINT_LAYERS = ['drivable_area', 'walkway', 'road_segment', 'parking', 'ped_crossing', 'stop_line']


# ============================================================
# Map Feature Extraction
# ============================================================

def get_map_features_for_point(nusc_map, x_global, y_global):
    """
    Query the semantic map at a single global (x, y) point.
    Returns a MAP_DIM-length feature vector.
    """
    features = np.zeros(MAP_DIM, dtype=np.float32)

    # --- Binary: is the point inside each layer? ---
    try:
        layers_at_point = nusc_map.layers_on_point(x_global, y_global)
    except Exception:
        layers_at_point = []

    layer_set = set(layers_at_point)
    features[0] = 1.0 if 'drivable_area' in layer_set else 0.0
    features[1] = 1.0 if 'walkway' in layer_set else 0.0
    features[2] = 1.0 if 'road_segment' in layer_set else 0.0
    features[3] = 1.0 if 'parking' in layer_set else 0.0

    # --- Distance-based: nearest crosswalk, stop line ---
    try:
        # Get records within a search radius
        nearby = nusc_map.get_records_in_radius(x_global, y_global, 20.0, ['ped_crossing', 'stop_line'])

        # Pedestrian crossing
        if 'ped_crossing' in nearby and len(nearby['ped_crossing']) > 0:
            features[4] = 1.0  # near crosswalk
            # Approximate distance (just use a flag + count)
            features[7] = min(1.0, 5.0 / (1.0 + len(nearby['ped_crossing'])))
        else:
            features[4] = 0.0
            features[7] = 1.0  # far from crosswalk (normalized)

        # Stop line
        if 'stop_line' in nearby and len(nearby['stop_line']) > 0:
            features[5] = 1.0
        else:
            features[5] = 0.0
    except Exception:
        features[4] = 0.0
        features[5] = 0.0
        features[7] = 1.0

    # --- Distance to road edge ---
    try:
        if 'road_segment' in layer_set:
            features[6] = 0.0  # on road = 0 distance
        else:
            # Not on road — compute approximate distance
            road_records = nusc_map.get_records_in_radius(x_global, y_global, 30.0, ['road_segment'])
            if 'road_segment' in road_records and len(road_records['road_segment']) > 0:
                features[6] = 0.3  # nearby road
            else:
                features[6] = 1.0  # far from road
    except Exception:
        features[6] = 0.5

    # --- Lane direction ---
    try:
        closest_lane = nusc_map.get_closest_lane(x_global, y_global, radius=10.0)
        if closest_lane:
            lane_record = nusc_map.get_arcline_path(closest_lane)
            if lane_record and len(lane_record) > 0:
                # Get direction from first segment
                pose = lane_record[0]
                heading = pose.get('heading', 0.0) if isinstance(pose, dict) else 0.0
                features[8] = np.sin(heading)
                features[9] = np.cos(heading)
    except Exception:
        features[8] = 0.0
        features[9] = 0.0

    # --- Number of lanes nearby ---
    try:
        nearby_lanes = nusc_map.get_records_in_radius(x_global, y_global, 10.0, ['lane'])
        if 'lane' in nearby_lanes:
            features[10] = min(1.0, len(nearby_lanes['lane']) / 6.0)  # normalized
        else:
            features[10] = 0.0
    except Exception:
        features[10] = 0.0

    return features


def extract_map_features_for_sample(nusc, nusc_maps, sample_token, rxf_feat):
    """
    Extract map features for all objects in a sample.

    Args:
        nusc: NuScenes instance
        nusc_maps: dict of {map_name: NuScenesMap}
        sample_token: sample token
        rxf_feat: (MAX_OBJECTS, 48) RadarXF features (positions at dims 2-4)

    Returns:
        (MAX_OBJECTS, MAP_DIM) map features
    """
    from pyquaternion import Quaternion

    map_features = np.zeros((MAX_OBJECTS, MAP_DIM), dtype=np.float32)

    sample = nusc.get('sample', sample_token)

    # Get ego pose from LIDAR_TOP reference
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])[:2]  # x, y
    ego_rotation = Quaternion(ego_pose['rotation'])

    # Get the correct map for this scene
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']

    if map_name not in nusc_maps:
        return map_features

    nusc_map = nusc_maps[map_name]

    # Process each detected object
    struct_dim = 16
    for i in range(MAX_OBJECTS):
        # Check if this is a valid object (non-zero features)
        if np.abs(rxf_feat[i, :struct_dim]).sum() < 1e-6:
            continue

        # Object position in ego frame (normalized in RadarXF features)
        # Dims 2-4 are x, y, z normalized by /50, /50, /5
        x_ego = rxf_feat[i, 2] * 50.0
        y_ego = rxf_feat[i, 3] * 50.0

        # Transform ego frame → global frame
        pos_ego = np.array([x_ego, y_ego, 0.0])
        pos_global = ego_rotation.rotate(pos_ego)[:2] + ego_translation

        x_global = float(pos_global[0])
        y_global = float(pos_global[1])

        # Query map
        map_features[i] = get_map_features_for_point(nusc_map, x_global, y_global)

    return map_features


import multiprocessing

# Globals for workers to avoid pickling overhead
g_nusc = None
g_nusc_maps = None
g_args = None

def init_worker(nusc_inst, nusc_maps_inst, args):
    global g_nusc, g_nusc_maps, g_args
    g_nusc = nusc_inst
    g_nusc_maps = nusc_maps_inst
    g_args = args

def process_sample(sample_token):
    try:
        # Load corresponding RadarXF features for object positions
        rxf_path = os.path.join(g_args.rxf_dir, f"{sample_token}.npy")
        if os.path.exists(rxf_path):
            rxf_feat = np.load(rxf_path)
        else:
            rxf_feat = np.zeros((MAX_OBJECTS, 48), dtype=np.float32)

        # Extract map features
        map_feat = extract_map_features_for_sample(
            g_nusc, g_nusc_maps, sample_token, rxf_feat
        )

        out_path = os.path.join(g_args.out_dir, f"{sample_token}.npy")
        np.save(out_path, map_feat.astype(np.float32))
        return (sample_token, True, None)
    except Exception as e:
        return (sample_token, False, str(e))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Extract map context features for NuScenes VQA")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--rxf-dir", default=RXF_DIR)
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load NuScenes ---
    from nuscenes.nuscenes import NuScenes
    print(f"Loading NuScenes {args.version} from {args.data_root}...")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    print(f"Loaded {len(nusc.scene)} scenes, {len(nusc.sample)} samples")

    # --- Load Maps ---
    from nuscenes.map_expansion.map_api import NuScenesMap

    map_locations = [
        'singapore-onenorth',
        'singapore-hollandvillage',
        'singapore-queenstown',
        'boston-seaport',
    ]

    nusc_maps = {}
    for loc in map_locations:
        try:
            nusc_maps[loc] = NuScenesMap(dataroot=args.data_root, map_name=loc)
            print(f"  Loaded map: {loc}")
        except Exception as e:
            print(f"  WARNING: Could not load map {loc}: {e}")

    if not nusc_maps:
        print("ERROR: No maps loaded! Download the map expansion pack.")
        print("  https://www.nuscenes.org/download")
        return

    # --- Check existing (for resume) ---
    existing = set()
    if os.path.exists(args.out_dir):
        existing = {f.replace('.npy', '') for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    print(f"Found {len(existing)} existing map features, will skip those")

    # --- Collect all tokens ---
    tokens_to_process = []
    for scene in nusc.scene:
        sample_token = scene["first_sample_token"]
        while sample_token:
            if sample_token not in existing:
                tokens_to_process.append(sample_token)
            sample_token = nusc.get("sample", sample_token)["next"]

    print(f"Processing {len(tokens_to_process)} samples with {args.workers} workers...")

    processed = 0
    errors = 0

    if args.workers > 1 and len(tokens_to_process) > 0:
        with multiprocessing.Pool(args.workers, initializer=init_worker, initargs=(nusc, nusc_maps, args)) as pool:
            results = list(tqdm(pool.imap_unordered(process_sample, tokens_to_process), total=len(tokens_to_process)))
            
            for res in results:
                token, success, err = res
                if success:
                    processed += 1
                else:
                    errors += 1
                    print(f"\n  Error on {token}: {err}")
    elif len(tokens_to_process) > 0:
        init_worker(nusc, nusc_maps, args)
        for token in tqdm(tokens_to_process):
            res = process_sample(token)
            if res[1]:
                processed += 1
            else:
                errors += 1
                print(f"\n  Error on {token}: {res[2]}")

    print(f"\n✅ Map feature extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Skipped (existing): {len(existing)}")
    print(f"   Errors: {errors}")
    print(f"   Output: {args.out_dir}")
    print(f"   Feature shape: ({MAX_OBJECTS}, {MAP_DIM})")


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()
