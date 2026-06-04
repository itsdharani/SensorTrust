"""Shared utilities for the sensor-trust project."""
from pathlib import Path
from datetime import datetime
import numpy as np
import os

def get_kitti_base_path():
    """Return the KITTI dataset base path. Works cross-platform."""
    return Path.home() / "Project" / "datasets" / "kitti"

def load_timestamps_from_file(filepath):
    """Load timestamps from a KITTI timestamps.txt file."""
    timestamps = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                date_part, time_part = line.split(' ')
                whole_secs, frac = time_part.split('.')
                frac = frac[:6]  # Truncate to microseconds
                clean_time = f"{date_part} {whole_secs}.{frac}"
                dt = datetime.strptime(clean_time, '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(dt.timestamp())
            except Exception as e:
                print(f"Error parsing line: {repr(line)}")
                raise
    return np.array(timestamps)

def get_sync_path(date, drive):
    """Get path to synchronized data for a specific sequence."""
    base = get_kitti_base_path()
    return base / date / f"{date}_drive_{drive}_sync"
