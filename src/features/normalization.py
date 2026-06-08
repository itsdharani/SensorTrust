"""Z-Score Normalization for Cross-Modal Motion Proxies.

Computes and stores normalization parameters (μ, σ) from clean training data.
Applies z-score normalization to make all motion proxies unitless and comparable.
"""
import numpy as np
import pickle
from pathlib import Path


class MotionNormalizer:
    """Handles z-score normalization for all four sensor motion proxies."""
    
    def __init__(self):
        self.stats = {}  # {proxy_name: {'mu': float, 'sigma': float}}
        self.fitted = False
    
    def fit(self, gps_proxies, imu_proxies, camera_proxies, lidar_proxies):
        """Compute μ and σ from clean training data.
        
        Args:
            gps_proxies: dict with keys 'delta_v', 'heading_rate'
            imu_proxies: dict with keys 'delta_v', 'yaw_rate'
            camera_proxies: dict with key 'flow_magnitude'
            lidar_proxies: dict with key 'icp_residual'
        """
        proxy_sources = {
            'gps_delta_v':      gps_proxies['delta_v'],
            'gps_heading_rate': gps_proxies['heading_rate'],
            'imu_delta_v':      imu_proxies['delta_v'],
            'imu_yaw_rate':     imu_proxies['yaw_rate'],
            'camera_flow':      camera_proxies['flow_magnitude'],
            'lidar_icp':        lidar_proxies['icp_residual'],
        }
        
        for name, data in proxy_sources.items():
            valid = data[~np.isnan(data)]
            self.stats[name] = {
                'mu': float(np.mean(valid)),
                'sigma': float(np.std(valid))
            }
        
        self.fitted = True
        print("Normalization parameters fitted:")
        for name, s in self.stats.items():
            print(f"  {name:20s}: μ={s['mu']:8.4f}, σ={s['sigma']:8.4f}")
    
    def transform(self, gps_proxies, imu_proxies, camera_proxies, lidar_proxies):
        """Apply z-score normalization.
        
        Returns dict of z-scored proxies with same keys as input + normalized versions.
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        proxy_sources = {
            'gps_delta_v':      gps_proxies['delta_v'],
            'gps_heading_rate': gps_proxies['heading_rate'],
            'imu_delta_v':      imu_proxies['delta_v'],
            'imu_yaw_rate':     imu_proxies['yaw_rate'],
            'camera_flow':      camera_proxies['flow_magnitude'],
            'lidar_icp':        lidar_proxies['icp_residual'],
        }
        
        z_scored = {}
        for name, data in proxy_sources.items():
            mu = self.stats[name]['mu']
            sigma = self.stats[name]['sigma']
            z_scored[name] = (data - mu) / sigma
        
        return z_scored
    
    def save(self, filepath):
        """Save normalization parameters to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'stats': self.stats, 'fitted': self.fitted}, f)
        print(f"Normalizer saved to {filepath}")
    
    def load(self, filepath):
        """Load normalization parameters from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.stats = data['stats']
        self.fitted = data['fitted']
        print(f"Normalizer loaded from {filepath}")
    
    def get_stats(self):
        """Return stored statistics as a formatted string."""
        lines = ["Normalization Parameters:", "-" * 50]
        for name, s in self.stats.items():
            lines.append(f"  {name:20s}: μ={s['mu']:8.4f}, σ={s['sigma']:8.4f}")
        return "\n".join(lines)
