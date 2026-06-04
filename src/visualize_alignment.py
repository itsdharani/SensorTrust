"""Visualize temporal alignment between LiDAR and GPS/IMU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from data_loader import KITTILoader
from alignment import TemporalAligner

def main():
    print("Loading and aligning...")
    loader = KITTILoader('2011_09_26', '0009')
    aligner = TemporalAligner(loader)
    lidar_to_imu, cam_to_imu = aligner.align_all(tolerance_ms=50.0)
    
    imu_times = loader.timestamps['oxts']
    velo_times = loader.timestamps['velo']
    cam_times = loader.timestamps['cam2']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # ==================================================================
    # Plot 1: LiDAR alignment gaps over the whole sequence
    # ==================================================================
    ax = axes[0]
    valid_lidar = lidar_to_imu != -1
    lidar_indices = np.arange(len(velo_times))
    lidar_gaps_ms = aligner.lidar_gaps * 1000  # Convert to ms
    
    ax.bar(lidar_indices[valid_lidar], lidar_gaps_ms[valid_lidar], 
           width=1.0, color='steelblue', alpha=0.7, label='Alignment Gap')
    ax.axhline(y=lidar_gaps_ms[valid_lidar].mean(), color='red', 
               linestyle='--', linewidth=1.5, label=f'Mean: {lidar_gaps_ms[valid_lidar].mean():.1f} ms')
    ax.axhline(y=50.0, color='gray', linestyle=':', linewidth=1, label='Tolerance (50 ms)')
    ax.set_xlabel('LiDAR Scan Index')
    ax.set_ylabel('Gap (milliseconds)')
    ax.set_title('Alignment Gap: Each LiDAR Scan to its Nearest GPS/IMU Reading')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ==================================================================
    # Plot 2: Zoomed view - first 50 LiDAR scans with exact timestamps
    # ==================================================================
    ax = axes[1]
    n_zoom = min(50, len(velo_times))
    
    for i in range(n_zoom):
        lidar_t = velo_times[i]
        imu_idx = lidar_to_imu[i]
        imu_t = imu_times[imu_idx]
        
        # Draw a line connecting the paired timestamps
        ax.plot([i, i], [lidar_t, imu_t], 'k-', linewidth=0.8, alpha=0.5)
        ax.plot(i, lidar_t, 'bo', markersize=4, label='LiDAR' if i == 0 else '')
        ax.plot(i, imu_t, 'rs', markersize=4, label='GPS/IMU' if i == 0 else '')
    
    ax.set_xlabel('Scan Index')
    ax.set_ylabel('Timestamp (seconds)')
    ax.set_title('First 50 Scans: LiDAR vs Paired GPS/IMU Timestamps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ==================================================================
    # Plot 3: Distribution of alignment gaps
    # ==================================================================
    ax = axes[2]
    ax.hist(lidar_gaps_ms[valid_lidar], bins=40, color='steelblue', 
            edgecolor='black', alpha=0.7)
    ax.axvline(x=lidar_gaps_ms[valid_lidar].mean(), color='red', 
               linestyle='--', linewidth=2, label=f'Mean = {lidar_gaps_ms[valid_lidar].mean():.1f} ms')
    ax.axvline(x=50.0, color='gray', linestyle=':', linewidth=2, label='Tolerance = 50 ms')
    ax.set_xlabel('Alignment Gap (milliseconds)')
    ax.set_ylabel('Number of Scans')
    ax.set_title('Distribution: How Far Apart Are LiDAR and GPS/IMU Readings?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ==================================================================
    # Statistics annotation
    # ==================================================================
    stats_text = f"""
    Total LiDAR scans:     {len(velo_times)}
    Successfully aligned:  {np.sum(valid_lidar)}
    Dropped:               {np.sum(~valid_lidar)}
    
    Mean gap:  {lidar_gaps_ms[valid_lidar].mean():.2f} ms
    Median gap: {np.median(lidar_gaps_ms[valid_lidar]):.2f} ms
    Min gap:   {lidar_gaps_ms[valid_lidar].min():.2f} ms
    Max gap:   {lidar_gaps_ms[valid_lidar].max():.2f} ms
    
    LiDAR period: 100 ms (10 Hz)
    Mean gap is {lidar_gaps_ms[valid_lidar].mean()/100*100:.1f}% of one LiDAR frame.
    → Data are effectively from the SAME MOMENT.
    """
    fig.text(0.02, 0.02, stats_text, fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('alignment_visualization.png', dpi=200, bbox_inches='tight')
    print("Saved: alignment_visualization.png")
    plt.show()

if __name__ == '__main__':
    main()
