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
    python precompute_map_features.py --workers 16
    python precompute_map_features.py --data-root /path/to/nuscenes --workers 8
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

# Valid NuScenes polygon layers (verified from NuScenesMap source code)
POLYGON_LAYERS = ['drivable_area', 'road_segment', 'road_block', 'lane',
                  'ped_crossing', 'walkway', 'stop_line', 'carpark_area']


# ============================================================
# Map Feature Extraction
# ============================================================

def get_map_features_for_point(nusc_map, x_global, y_global):
    """
    Query the semantic map at a single global (x, y) point.
    Returns a MAP_DIM-length feature vector.

    IMPORTANT: layers_on_point() returns Dict[str, str] where:
      - key = layer name
      - value = token string (non-empty = point IS on layer, '' = NOT on layer)
    """
    features = np.zeros(MAP_DIM, dtype=np.float32)

    # --- Binary: is the point inside each polygon layer? ---
    try:
        # Returns {layer_name: token_or_empty_string}
        layers_result = nusc_map.layers_on_point(x_global, y_global)
    except Exception:
        layers_result = {}

    # A non-empty token string means the point IS on that layer
    features[0] = 1.0 if layers_result.get('drivable_area', '') != '' else 0.0
    features[1] = 1.0 if layers_result.get('walkway', '') != '' else 0.0
    features[2] = 1.0 if layers_result.get('road_segment', '') != '' else 0.0
    features[3] = 1.0 if layers_result.get('carpark_area', '') != '' else 0.0

    # --- Distance-based: nearest crosswalk, stop line ---
    try:
        # get_records_in_radius returns {layer_name: [list of tokens]}
        nearby = nusc_map.get_records_in_radius(x_global, y_global, 20.0, ['ped_crossing', 'stop_line'])

        # Pedestrian crossing
        ped_tokens = nearby.get('ped_crossing', [])
        if len(ped_tokens) > 0:
            features[4] = 1.0  # near crosswalk
            features[7] = min(1.0, 5.0 / (1.0 + len(ped_tokens)))
        else:
            features[4] = 0.0
            features[7] = 1.0  # far from crosswalk

        # Stop line
        stop_tokens = nearby.get('stop_line', [])
        features[5] = 1.0 if len(stop_tokens) > 0 else 0.0

    except Exception:
        features[4] = 0.0
        features[5] = 0.0
        features[7] = 1.0

    # --- Distance to road edge ---
    try:
        if features[2] > 0.5:  # already on road_segment
            features[6] = 0.0
        else:
            road_records = nusc_map.get_records_in_radius(x_global, y_global, 30.0, ['road_segment'])
            road_tokens = road_records.get('road_segment', [])
            if len(road_tokens) > 0:
                features[6] = 0.3  # nearby road
            else:
                features[6] = 1.0  # far from road
    except Exception:
        features[6] = 0.5

    # --- Lane direction (using discretized lane centerline) ---
    try:
        closest_lane = nusc_map.get_closest_lane(x_global, y_global, radius=10.0)
        if closest_lane:
            # get_arcline_path returns List[ArcLinePath dicts]
            # Use discretize_lanes to get actual (x, y, yaw) poses
            discrete = nusc_map.discretize_lanes([closest_lane], 0.5)
            if closest_lane in discrete and len(discrete[closest_lane]) > 1:
                poses = discrete[closest_lane]
                # Each pose is (x, y, yaw)
                # Find the pose closest to our query point
                poses_arr = np.array(poses)
                dists = np.linalg.norm(poses_arr[:, :2] - [x_global, y_global], axis=1)
                closest_idx = np.argmin(dists)
                yaw = poses_arr[closest_idx, 2]  # yaw/heading in radians
                features[8] = np.sin(yaw)
                features[9] = np.cos(yaw)
    except Exception:
        features[8] = 0.0
        features[9] = 0.0

    # --- Number of lanes nearby ---
    try:
        nearby_lanes = nusc_map.get_records_in_radius(x_global, y_global, 10.0, ['lane', 'lane_connector'])
        n_lanes = len(nearby_lanes.get('lane', [])) + len(nearby_lanes.get('lane_connector', []))
        features[10] = min(1.0, n_lanes / 6.0)  # normalized
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

    # --- Sanity check: verify map features work correctly ---
    print("\n--- Sanity Check ---")
    test_map = list(nusc_maps.values())[0]
    test_result = test_map.layers_on_point(300.0, 1700.0)
    print(f"  layers_on_point(300, 1700) = {test_result}")
    print(f"  Type: {type(test_result)}")
    for k, v in test_result.items():
        print(f"    {k}: '{v}' (on_layer={v != ''})")

    # --- Check existing (for resume) ---
    existing = set()
    if os.path.exists(args.out_dir):
        existing = {f.replace('.npy', '') for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    print(f"\nFound {len(existing)} existing map features, will skip those")

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
