"""LiDAR Motion Proxy Extraction.

Computes ego-motion compensated ICP residual as the LiDAR motion proxy.
"""
import numpy as np
import open3d as o3d


def compute_icp_residual(scan_curr, scan_prev, transform):
    """Compute ICP residual after ego-motion compensation.
    
    Args:
        scan_curr: (N, 3) array — current LiDAR scan (x, y, z)
        scan_prev: (M, 3) array — previous LiDAR scan
        transform: (4, 4) transformation matrix from OXTS pose
    
    Returns:
        float — mean nearest-neighbour distance after transformation
    """

    # ------------------------------------------------------------------
    # Downsample BEFORE any expensive operations
    # ------------------------------------------------------------------
    # scan_curr = scan_curr[::10]
    # scan_prev = scan_prev[::10]

    # Transform previous scan into current frame
    scan_prev_homog = np.hstack(
        [scan_prev, np.ones((scan_prev.shape[0], 1))]
    )

    scan_prev_transformed = (
        transform @ scan_prev_homog.T
    ).T[:, :3]

    # Build KD-tree on transformed previous scan
    pcd_prev = o3d.geometry.PointCloud()
    pcd_prev.points = o3d.utility.Vector3dVector(scan_prev_transformed)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_prev)

    # For each point in current scan, find nearest neighbour
    residuals = []

    for point in scan_curr:
        [_, _, dist2] = pcd_tree.search_knn_vector_3d(point, 1)
        residuals.append(np.sqrt(dist2[0]))

    return np.mean(residuals)


def get_oxts_transform(prev_frame, curr_frame):
    """Build 4x4 transformation matrix from OXTS pose data.

    Args:
        prev_frame: previous pykitti OXTS frame
        curr_frame: current pykitti OXTS frame

    Returns:
        (4, 4) numpy array
    """

    # KITTI already provides fused GPS/IMU poses through T_w_imu
    T_prev = prev_frame.T_w_imu
    T_curr = curr_frame.T_w_imu

    # Relative transform from previous frame to current frame
    transform = np.linalg.inv(T_curr) @ T_prev

    return transform


def extract_lidar_residuals(velo_scans, oxts_data, skip_first=1):
    """Extract LiDAR ICP residuals for all consecutive scan pairs.

    Args:
        velo_scans: list of (N, 4) arrays from pykitti
        oxts_data: pykitti oxts list
        skip_first: skip the first N scans for alignment

    Returns:
        np.array of shape (N-1,) — ICP residuals in meters
    """

    residuals = []

    total_scans = len(velo_scans)

    for i in range(skip_first, total_scans):

        if i % 20 == 0:
            print(f"Processing scan {i}/{total_scans}")

        scan_curr = velo_scans[i][:, :3]
        scan_prev = velo_scans[i - 1][:, :3]

        transform = get_oxts_transform(
            oxts_data[i - 1],
            oxts_data[i]
        )

        residual = compute_icp_residual(
            scan_curr,
            scan_prev,
            transform
        )

        residuals.append(residual)

    return np.array(residuals)


def extract_all_lidar_proxies(velo_scans, oxts_data):
    """Extract all LiDAR motion proxies.

    Returns dict with key: icp_residual
    """

    residuals = extract_lidar_residuals(
        velo_scans,
        oxts_data
    )

    return {
        'icp_residual': residuals
    }