#!/usr/bin/env python3
"""
Pre-compute LiDAR BEV features for NuScenes VQA Trimodal Fusion.

Converts each LiDAR point cloud to a BEV (Bird's-Eye-View) statistical
grid that complements the existing camera BEV features.

The ego-centric area (-50m to +50m in x and y) is divided into an 80-cell
1D radial grid (matching the camera BEV grid size). For each cell, 6
statistical channels are computed from the LiDAR points:

  [0] point_density     — log(1 + count) / 5      (how many points)
  [1] mean_height       — mean(z) / 5             (average elevation)
  [2] max_height        — max(z) / 5              (tallest structure)
  [3] height_var        — var(z) / 5              (shape complexity)
  [4] intensity_mean    — mean(intensity) / 255    (reflectivity)
  [5] intensity_var     — var(intensity) / 255     (surface texture)

Output shape: (80, 6) per sample  —  matches BEV grid rows for easy concat.

Usage:
    python precompute_lidar_features.py
    python precompute_lidar_features.py --data-root /path/to/nuscenes
    python precompute_lidar_features.py --check   # just verify files exist
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
OUT_DIR = "/media/nas_mount/anwar2/experiment/dataset/nuscenes/nischay/lidar_bev_features"

# Grid parameters — match camera BEV
GRID_CELLS = 80          # number of radial bins
RANGE_MAX = 50.0          # meters from ego in each direction
FEAT_DIM = 6             # channels per cell

# Point cloud filter
Z_MIN = -5.0              # filter floor points below this
Z_MAX = 10.0              # filter unreasonable height
INTENSITY_SCALE = 255.0   # NuScenes intensity normalization


# ============================================================
# LiDAR Point Cloud Loading
# ============================================================

def load_lidar_points(lidar_path):
    """
    Load NuScenes LiDAR point cloud from .pcd.bin file.
    Returns (N, 5) array: [x, y, z, intensity, ring_index]
    """
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    return points


def transform_to_ego(points, calibrated_sensor, ego_pose):
    """
    Transform points from sensor frame to ego frame.
    Uses NuScenes' calibrated_sensor (sensor→ego) transform.
    """
    from pyquaternion import Quaternion

    # Sensor → Ego
    rotation = Quaternion(calibrated_sensor['rotation']).rotation_matrix
    translation = np.array(calibrated_sensor['translation'])
    points_xyz = points[:, :3]
    points_ego = (rotation @ points_xyz.T).T + translation

    # Keep other channels
    result = np.copy(points)
    result[:, :3] = points_ego
    return result


# ============================================================
# BEV Grid Computation
# ============================================================

def compute_lidar_bev(points, grid_cells=GRID_CELLS, range_max=RANGE_MAX):
    """
    Convert point cloud to BEV statistical grid.

    Instead of a 2D grid (which would be very sparse at 80x80),
    we use an 80-bin RADIAL grid based on distance from ego.
    This matches the camera BEV representation.

    Args:
        points: (N, 5) array [x, y, z, intensity, ring]
        grid_cells: number of radial bins
        range_max: max distance in meters

    Returns:
        (grid_cells, 6) feature array
    """
    features = np.zeros((grid_cells, FEAT_DIM), dtype=np.float32)

    if len(points) == 0:
        return features

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3]

    # Filter by height range
    height_mask = (z >= Z_MIN) & (z <= Z_MAX)
    x, y, z, intensity = x[height_mask], y[height_mask], z[height_mask], intensity[height_mask]

    if len(x) == 0:
        return features

    # Compute radial distance from ego
    dist = np.sqrt(x**2 + y**2)

    # Bin by radial distance
    bin_size = range_max / grid_cells
    bin_idx = np.clip((dist / bin_size).astype(int), 0, grid_cells - 1)

    for i in range(grid_cells):
        mask = (bin_idx == i)
        count = mask.sum()

        if count == 0:
            continue

        z_cell = z[mask]
        int_cell = intensity[mask]

        # [0] Point density (log-scaled)
        features[i, 0] = np.log1p(count) / 5.0

        # [1] Mean height
        features[i, 1] = z_cell.mean() / 5.0

        # [2] Max height
        features[i, 2] = z_cell.max() / 5.0

        # [3] Height variance (shape complexity)
        features[i, 3] = z_cell.var() / 5.0 if count > 1 else 0.0

        # [4] Intensity mean (reflectivity)
        features[i, 4] = int_cell.mean() / INTENSITY_SCALE

        # [5] Intensity variance (surface texture)
        features[i, 5] = int_cell.var() / INTENSITY_SCALE if count > 1 else 0.0

    return features


# ============================================================
# Main Extraction
# ============================================================

def check_lidar_exists(data_root):
    """Check if LiDAR data files exist in the NuScenes dataset."""
    lidar_dir = os.path.join(data_root, "samples", "LIDAR_TOP")
    if not os.path.exists(lidar_dir):
        print(f"ERROR: LiDAR directory not found at {lidar_dir}")
        print("NuScenes v1.0-trainval should include LIDAR_TOP data.")
        print("Please ensure the full dataset is downloaded.")
        return False

    files = [f for f in os.listdir(lidar_dir) if f.endswith('.pcd.bin')]
    print(f"Found {len(files)} LiDAR point cloud files in {lidar_dir}")
    return len(files) > 0


def extract_all(data_root, version, out_dir):
    """Extract LiDAR BEV features for all samples."""
    from nuscenes.nuscenes import NuScenes

    print(f"Loading NuScenes {version} from {data_root}...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    os.makedirs(out_dir, exist_ok=True)

    # Count existing
    existing = set(f.replace('.npy', '') for f in os.listdir(out_dir) if f.endswith('.npy'))
    print(f"Found {len(existing)} existing features, {len(nusc.sample) - len(existing)} to extract")

    skipped = 0
    processed = 0

    for sample in tqdm(nusc.sample, desc="Extracting LiDAR BEV"):
        token = sample['token']

        if token in existing:
            skipped += 1
            continue

        # Get LiDAR sample data
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(nusc.dataroot, lidar_sd['filename'])

        if not os.path.exists(lidar_path):
            # Save zeros for missing data
            features = np.zeros((GRID_CELLS, FEAT_DIM), dtype=np.float32)
            np.save(os.path.join(out_dir, f"{token}.npy"), features)
            continue

        # Load point cloud
        points = load_lidar_points(lidar_path)

        # Transform to ego frame
        cal_sensor = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        points_ego = transform_to_ego(points, cal_sensor, None)

        # Compute BEV features
        features = compute_lidar_bev(points_ego)
        np.save(os.path.join(out_dir, f"{token}.npy"), features)
        processed += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Features saved to: {out_dir}")
    print(f"Feature shape: ({GRID_CELLS}, {FEAT_DIM})")


def main():
    parser = argparse.ArgumentParser(description="Extract LiDAR BEV features")
    parser.add_argument('--data-root', default=DATA_ROOT, help="NuScenes data root")
    parser.add_argument('--version', default=VERSION, help="NuScenes version")
    parser.add_argument('--out-dir', default=OUT_DIR, help="Output directory")
    parser.add_argument('--check', action='store_true', help="Only check if LiDAR data exists")
    args = parser.parse_args()

    if args.check:
        check_lidar_exists(args.data_root)
        return

    if not check_lidar_exists(args.data_root):
        print("\nCannot proceed without LiDAR data.")
        print("Check if your NuScenes dataset includes samples/LIDAR_TOP/")
        return

    extract_all(args.data_root, args.version, args.out_dir)


if __name__ == "__main__":
    main()
