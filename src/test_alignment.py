"""Verify temporal alignment quality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import KITTILoader
from alignment import TemporalAligner

def main():
    print("Loading KITTI sequence...")
    loader = KITTILoader('2011_09_26', '0009')
    
    print("Running temporal alignment...")
    aligner = TemporalAligner(loader)
    aligner.align_all(tolerance_ms=50.0)
    
    # Print the formatted report
    print(aligner.get_quality_report())
    
    # Detailed checks
    q = aligner.alignment_quality
    
    print("\n=== PASS/FAIL CHECKS ===")
    
    checks = []
    
    # Check 1: Drop rate must be low
    lidar_drop_ok = q['lidar_drop_rate'] < 0.10  # Less than 10% dropped
    cam_drop_ok = q['camera_drop_rate'] < 0.10
    checks.append(("LiDAR drop rate < 10%", lidar_drop_ok, q['lidar_drop_rate']))
    checks.append(("Camera drop rate < 10%", cam_drop_ok, q['camera_drop_rate']))
    
    # Check 2: Mean alignment gap must be small
    lidar_gap_ok = q['mean_lidar_gap_ms'] is not None and q['mean_lidar_gap_ms'] < 25.0
    cam_gap_ok = q['mean_camera_gap_ms'] is not None and q['mean_camera_gap_ms'] < 25.0
    checks.append(("LiDAR mean gap < 25 ms", lidar_gap_ok, q['mean_lidar_gap_ms']))
    checks.append(("Camera mean gap < 25 ms", cam_gap_ok, q['mean_camera_gap_ms']))
    
    # Check 3: Max gap must be within tolerance
    lidar_max_ok = q['max_lidar_gap_ms'] is not None and q['max_lidar_gap_ms'] <= 50.0
    cam_max_ok = q['max_camera_gap_ms'] is not None and q['max_camera_gap_ms'] <= 50.0
    checks.append(("LiDAR max gap <= 50 ms", lidar_max_ok, q['max_lidar_gap_ms']))
    checks.append(("Camera max gap <= 50 ms", cam_max_ok, q['max_camera_gap_ms']))
    
    # Check 4: At least 90% of frames aligned
    lidar_ratio = q['valid_lidar_alignments'] / q['total_lidar_frames']
    cam_ratio = q['valid_camera_alignments'] / q['total_camera_frames']
    checks.append(("LiDAR alignment > 90%", lidar_ratio > 0.90, lidar_ratio))
    checks.append(("Camera alignment > 90%", cam_ratio > 0.90, cam_ratio))
    
    # Print results
    all_pass = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        if isinstance(value, float):
            print(f"  [{status}] {name}: {value:.4f}")
        else:
            print(f"  [{status}] {name}: {value}")
    
    print("\n=== RESULT ===")
    if all_pass:
        print("ALL CHECKS PASSED - Alignment is working correctly.")
        print("You can proceed to feature extraction (Week 4).")
    else:
        print("SOME CHECKS FAILED - Fix alignment before proceeding.")
        print("Common fixes:")
        print("  1. Increase tolerance_ms (try 100ms)")
        print("  2. Check timestamp files are being read correctly")
        print("  3. Verify you're using the _sync dataset, not _extract")

if __name__ == '__main__':
    main()
