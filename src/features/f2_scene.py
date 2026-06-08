"""Feature 2: LiDAR Scene-Change Consistency.

Compares GPS-reported speed with LiDAR ICP residual.
Physical basis: When a vehicle moves, the LiDAR point cloud changes.
If GPS says moving but LiDAR says static (or vice versa), something is spoofed.
"""
import numpy as np


def compute_f2(z_gps_delta_v, z_lidar_icp):
    """Compute GPS-LiDAR scene-change consistency.
    
    F2 = |z_gps_delta_v - z_lidar_icp|
    
    Args:
        z_gps_delta_v: z-scored GPS speed change
        z_lidar_icp:   z-scored LiDAR ICP residual
    
    Returns:
        np.array of same length — F2 values per frame
    """
    # Both are z-scored, so they're directly comparable
    f2 = np.abs(z_gps_delta_v - z_lidar_icp)
    return f2


def get_f2_stats(f2, labels=None):
    """Print statistics for F2.
    
    Args:
        f2: F2 feature array
        labels: optional binary labels (0=clean, 1=attacked) for per-class stats
    """
    valid = f2[~np.isnan(f2)]
    print(f"F2 (GPS-LiDAR Scene Consistency):")
    print(f"  mean={np.mean(valid):.4f}, std={np.std(valid):.4f}")
    print(f"  min={np.min(valid):.4f}, max={np.max(valid):.4f}")
    
    if labels is not None:
        clean = f2[labels == 0]
        attacked = f2[labels == 1]
        clean_valid = clean[~np.isnan(clean)]
        att_valid = attacked[~np.isnan(attacked)]
        if len(clean_valid) > 0:
            print(f"  Clean:    mean={np.mean(clean_valid):.4f}")
        if len(att_valid) > 0:
            print(f"  Attacked: mean={np.mean(att_valid):.4f}")
            if len(clean_valid) > 0:
                print(f"  Ratio:    {np.mean(att_valid)/np.mean(clean_valid):.1f}x")
