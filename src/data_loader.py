"""Load and validate KITTI raw data sequences."""
import pykitti
import numpy as np
from pathlib import Path
from utils import get_kitti_base_path, get_sync_path, load_timestamps_from_file
import os

class KITTILoader:
    """Loads and validates a single KITTI raw sequence."""
    
    def __init__(self, date='2011_09_26', drive='0009'):
        self.date = date
        self.drive = drive
        self.base_path = get_kitti_base_path()
        
        # Load via pykitti
        self.raw_data = pykitti.raw(
            base_path=str(self.base_path),
            date=self.date,
            drive=self.drive
        )
        
        # Load timestamps from files (more precise than pykitti's built-in)
        self.sync_path = get_sync_path(self.date, self.drive)
        self.timestamps = self._load_all_timestamps()
        
    def _load_all_timestamps(self):
        """Load timestamps from KITTI sync files."""
        return {
            'oxts': load_timestamps_from_file(
                os.path.join(self.sync_path, 'oxts', 'timestamps.txt')),
            'velo': load_timestamps_from_file(
                os.path.join(self.sync_path, 'velodyne_points', 'timestamps.txt')),
            'cam2': load_timestamps_from_file(
                os.path.join(self.sync_path, 'image_02', 'timestamps.txt'))
        }
    
    def get_statistics(self):
        """Return summary statistics about the loaded sequence."""
        stats = {
            'num_gps_imu': len(self.raw_data.oxts),
            'num_lidar': len(self.raw_data.velo_files),
            'num_camera': len(self.raw_data.cam2_files),
            'gps_imu_hz': 1.0 / np.mean(np.diff(self.timestamps['oxts'])),
            'lidar_hz': 1.0 / np.mean(np.diff(self.timestamps['velo'])),
            'camera_hz': 1.0 / np.mean(np.diff(self.timestamps['cam2'])),
        }
        return stats
    
    def get_gps_reading(self, index):
        """Return a single GPS/IMU reading as a dictionary."""
        frame = self.raw_data.oxts[index]
        return {
            'lat': frame.packet.lat,
            'lon': frame.packet.lon,
            'alt': frame.packet.alt,
            'vf': frame.packet.vf,   # Forward velocity
            'vl': frame.packet.vl,   # Leftward velocity
            'vu': frame.packet.vu,   # Upward velocity
            'ax': frame.packet.ax,   # Acceleration x
            'ay': frame.packet.ay,   # Acceleration y
            'az': frame.packet.az,   # Acceleration z
            'wx': frame.packet.wx,   # Angular rate x
            'wy': frame.packet.wy,   # Angular rate y
            'wz': frame.packet.wz,   # Angular rate z
            'roll': frame.packet.roll,
            'pitch': frame.packet.pitch,
            'yaw': frame.packet.yaw,
        }
    
    def get_lidar_scan(self, index):
        """Return a single LiDAR scan as (N, 4) array."""
        return self.raw_data.get_velo(index)
    
    def get_camera_image(self, index, camera='cam2'):
        """Return a single camera image as numpy array."""
        if camera == 'cam2':
            return self.raw_data.get_cam2(index)
        elif camera == 'cam3':
            return self.raw_data.get_cam3(index)
        raise ValueError(f"Unknown camera: {camera}")

# Quick test when run directly
if __name__ == '__main__':
    loader = KITTILoader('2011_09_26', '0009')
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
