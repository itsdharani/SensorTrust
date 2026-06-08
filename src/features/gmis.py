"""Feature 3: Global Motion Inconsistency Score (GMIS).

Weighted RMS of all 6 pairwise sensor disagreements.
Measures how much all four sensors disagree with each other simultaneously.
"""
import numpy as np


# Sensor reliability weights (fixed design parameters)
WEIGHTS = {
    'gps':    1.0,
    'imu':    1.0,
    'lidar':  1.0,
    'camera': 0.5   # Optical flow is noisier at 10 Hz
}


def compute_gmis(z_gps_delta_v, z_imu_delta_v, z_lidar_icp, z_camera_flow):
    """Compute Global Motion Inconsistency Score.
    
    Step 1: Compute all 6 pairwise weighted differences
    Step 2: Weighted RMS = sqrt( Σ w_ij * (z_i - z_j)² / Σ w_ij )
    
    Args:
        z_gps_delta_v:  z-scored GPS speed change
        z_imu_delta_v:  z-scored IMU speed change
        z_lidar_icp:    z-scored LiDAR ICP residual
        z_camera_flow:  z-scored camera optical flow
    
    Returns:
        np.array of same length — GMIS values per frame
    """
    n = len(z_gps_delta_v)
    gmis = np.full(n, np.nan)
    
    for t in range(n):
        # Get z-scores at this timestep
        zg = z_gps_delta_v[t]
        zi = z_imu_delta_v[t]
        zl = z_lidar_icp[t]
        zc = z_camera_flow[t]
        
        # Skip if any value is NaN
        if np.isnan([zg, zi, zl, zc]).any():
            continue
        
        # All 6 pairwise weighted squared differences
        weighted_sum = (
            WEIGHTS['gps'] * WEIGHTS['imu']     * (zg - zi)**2 +
            WEIGHTS['gps'] * WEIGHTS['lidar']   * (zg - zl)**2 +
            WEIGHTS['gps'] * WEIGHTS['camera']  * (zg - zc)**2 +
            WEIGHTS['imu'] * WEIGHTS['lidar']   * (zi - zl)**2 +
            WEIGHTS['imu'] * WEIGHTS['camera']  * (zi - zc)**2 +
            WEIGHTS['lidar'] * WEIGHTS['camera']* (zl - zc)**2
        )
        
        # Sum of weights
        weight_sum = (
            WEIGHTS['gps'] * WEIGHTS['imu'] +
            WEIGHTS['gps'] * WEIGHTS['lidar'] +
            WEIGHTS['gps'] * WEIGHTS['camera'] +
            WEIGHTS['imu'] * WEIGHTS['lidar'] +
            WEIGHTS['imu'] * WEIGHTS['camera'] +
            WEIGHTS['lidar'] * WEIGHTS['camera']
        )
        
        gmis[t] = np.sqrt(weighted_sum / weight_sum)
    
    return gmis


def get_gmis_stats(gmis, labels=None):
    """Print statistics for GMIS."""
    valid = gmis[~np.isnan(gmis)]
    print(f"GMIS (Global Motion Inconsistency Score):")
    print(f"  mean={np.mean(valid):.4f}, std={np.std(valid):.4f}")
    print(f"  min={np.min(valid):.4f}, max={np.max(valid):.4f}")
    
    if labels is not None:
        clean = gmis[labels == 0]
        attacked = gmis[labels == 1]
        clean_valid = clean[~np.isnan(clean)]
        att_valid = attacked[~np.isnan(attacked)]
        if len(clean_valid) > 0:
            print(f"  Clean:    mean={np.mean(clean_valid):.4f}")
        if len(att_valid) > 0:
            print(f"  Attacked: mean={np.mean(att_valid):.4f}")
            if len(clean_valid) > 0:
                print(f"  Ratio:    {np.mean(att_valid)/np.mean(clean_valid):.1f}x")


def get_pairwise_contributions(z_gps_delta_v, z_imu_delta_v, z_lidar_icp, z_camera_flow, t):
    """Return the 6 pairwise disagreement values at a single timestep.
    
    Useful for diagnosing which sensor pair is driving the GMIS spike.
    """
    zg = z_gps_delta_v[t]
    zi = z_imu_delta_v[t]
    zl = z_lidar_icp[t]
    zc = z_camera_flow[t]
    
    contributions = {
        'GPS-IMU':     WEIGHTS['gps'] * WEIGHTS['imu']     * abs(zg - zi),
        'GPS-LiDAR':   WEIGHTS['gps'] * WEIGHTS['lidar']   * abs(zg - zl),
        'GPS-Camera':  WEIGHTS['gps'] * WEIGHTS['camera']  * abs(zg - zc),
        'IMU-LiDAR':   WEIGHTS['imu'] * WEIGHTS['lidar']   * abs(zi - zl),
        'IMU-Camera':  WEIGHTS['imu'] * WEIGHTS['camera']  * abs(zi - zc),
        'LiDAR-Camera':WEIGHTS['lidar']* WEIGHTS['camera'] * abs(zl - zc),
    }
    
    return contributions
