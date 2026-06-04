"""Temporal alignment of multi-sensor streams."""
import numpy as np
from utils import get_sync_path, load_timestamps_from_file
import os

class TemporalAligner:
    """
    Aligns GPS/IMU, LiDAR, and Camera streams to a common timeline.
    IMU timestamps are used as the reference (highest frequency).
    """
    
    def __init__(self, loader):
        self.loader = loader
        self.timestamps = loader.timestamps
        self.alignment_quality = {}
        
    def align_all(self, tolerance_ms=50.0):
        """
        Align LiDAR and Camera to the nearest IMU timestamp.
        Returns alignment indices and quality statistics.
        """
        imu_times = self.timestamps['oxts']
        velo_times = self.timestamps['velo']
        cam_times = self.timestamps['cam2']
        
        tolerance = tolerance_ms / 1000.0  # Convert to seconds
        
        # Align LiDAR to IMU
        self.lidar_to_imu, self.lidar_gaps = self._find_nearest(
            imu_times, velo_times, tolerance
        )
        
        # Align Camera to IMU
        self.cam_to_imu, self.cam_gaps = self._find_nearest(
            imu_times, cam_times, tolerance
        )
        
        # Compute quality statistics
        self._compute_quality()
        
        return self.lidar_to_imu, self.cam_to_imu
    
    def _find_nearest(self, reference_times, query_times, tolerance):
        """For each query timestamp, find the nearest reference timestamp."""
        indices = []
        gaps = []
        for t in query_times:
            diffs = np.abs(reference_times - t)
            idx = np.argmin(diffs)
            gap = diffs[idx]
            if gap <= tolerance:
                indices.append(idx)
                gaps.append(gap)
            else:
                indices.append(-1)
                gaps.append(gap)
        return np.array(indices), np.array(gaps)
    
    def _compute_quality(self):
        """Compute and store alignment quality metrics."""
        valid_lidar = self.lidar_to_imu != -1
        valid_cam = self.cam_to_imu != -1
        
        self.alignment_quality = {
            'total_imu_frames': len(self.timestamps['oxts']),
            'total_lidar_frames': len(self.timestamps['velo']),
            'total_camera_frames': len(self.timestamps['cam2']),
            'valid_lidar_alignments': int(np.sum(valid_lidar)),
            'valid_camera_alignments': int(np.sum(valid_cam)),
            'lidar_drop_rate': 1.0 - np.mean(valid_lidar),
            'camera_drop_rate': 1.0 - np.mean(valid_cam),
            'mean_lidar_gap_ms': np.mean(self.lidar_gaps[valid_lidar]) * 1000 if np.any(valid_lidar) else None,
            'max_lidar_gap_ms': np.max(self.lidar_gaps[valid_lidar]) * 1000 if np.any(valid_lidar) else None,
            'mean_camera_gap_ms': np.mean(self.cam_gaps[valid_cam]) * 1000 if np.any(valid_cam) else None,
            'max_camera_gap_ms': np.max(self.cam_gaps[valid_cam]) * 1000 if np.any(valid_cam) else None,
        }
    
    def get_quality_report(self):
        """Return a formatted string of alignment quality."""
        q = self.alignment_quality
        report = f"""
Temporal Alignment Quality Report
---------------------------------
IMU frames:           {q['total_imu_frames']}
LiDAR frames:         {q['total_lidar_frames']} 
  -> Valid matches:   {q['valid_lidar_alignments']} ({q['lidar_drop_rate']*100:.1f}% dropped)
  -> Mean gap:        {q['mean_lidar_gap_ms']:.2f} ms
  -> Max gap:         {q['max_lidar_gap_ms']:.2f} ms
Camera frames:        {q['total_camera_frames']}
  -> Valid matches:   {q['valid_camera_alignments']} ({q['camera_drop_rate']*100:.1f}% dropped)
  -> Mean gap:        {q['mean_camera_gap_ms']:.2f} ms
  -> Max gap:         {q['max_camera_gap_ms']:.2f} ms
"""
        return report

# Quick test
if __name__ == '__main__':
    from .data_loader import KITTILoader
    loader = KITTILoader('2011_09_26', '0009')
    aligner = TemporalAligner(loader)
    aligner.align_all()
    print(aligner.get_quality_report())
