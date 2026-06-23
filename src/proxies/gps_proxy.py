"""GPS Motion Proxy Extraction.

Extracts motion signals from OXTS GPS/IMU data:
    - gps_speed:        forward velocity magnitude (m/s)
    - gps_heading:      bearing from consecutive positions (radians)
    - gps_heading_rate: rate of heading change (rad/s)
"""
import numpy as np

#  helper functions
def get_vf(frame):
    if isinstance(frame, dict):
        return frame["vf"]
    return frame.packet.vf


def get_lat(frame):
    if isinstance(frame, dict):
        return frame["lat"]
    return frame.packet.lat


def get_lon(frame):
    if isinstance(frame, dict):
        return frame["lon"]
    return frame.packet.lon



def extract_gps_speed(oxts_data):
    return np.array([get_vf(frame) for frame in oxts_data])

def extract_gps_heading(oxts_data):
    """Compute heading (bearing) from consecutive GPS positions.
    
    Returns array of shape (N,) in radians [0, 2π).
    First element is NaN.
    """
    lats = np.array([get_lat(frame) for frame in oxts_data])
    lons = np.array([get_lon(frame) for frame in oxts_data])
    
    dlat = np.diff(lats)
    dlon = np.diff(lons)
    
    headings = np.arctan2(dlon, dlat)
    headings = np.mod(headings, 2 * np.pi)
    headings = np.insert(headings, 0, np.nan)
    
    return headings


def extract_gps_heading_rate(oxts_data, dt=0.1035):
    """Compute heading change rate from consecutive headings.
    
    Uses dt = 0.1035s based on your measured ~9.7 Hz GPS frequency.
    First two elements are NaN.
    """
    headings = extract_gps_heading(oxts_data)
    
    heading_diff = np.diff(headings)
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
    
    heading_rate = heading_diff / dt
    heading_rate = np.insert(heading_rate, 0, [np.nan, np.nan])
    
    return heading_rate


def extract_all_gps_proxies(oxts_data, dt=0.1035, window=5):
    """Extract all GPS motion proxies."""
    return {
        'speed': extract_gps_speed(oxts_data),
        'delta_v': extract_gps_delta_v(oxts_data, window),
        'heading': extract_gps_heading(oxts_data),
        'heading_rate': extract_gps_heading_rate(oxts_data, dt)
    }

def get_heading_gate_mask(gps_speed, min_speed=2.0):
    """Return boolean mask: True where heading data is reliable.

    GPS heading from consecutive positions is unreliable below ~2 m/s.
    This mask gates the heading component of F1.

    Args:
        gps_speed: array of GPS forward speeds
        min_speed: minimum speed for reliable heading (m/s)

    Returns:
        np.array of bool, same shape as gps_speed
    """
    return np.abs(gps_speed) >= min_speed

def extract_gps_delta_v(oxts_data, window=5):
    """Compute GPS-measured speed change over a sliding window.
    
    Args:
        oxts_data: pykitti oxts list
        window: number of frames for change computation
    
    Returns:
        np.array of shape (N,) — speed change over window (m/s)
        First `window` elements are NaN.
    """
    speed = extract_gps_speed(oxts_data)
    delta_v = np.full(len(speed), np.nan)
    
    for i in range(window, len(speed)):
        delta_v[i] = speed[i] - speed[i - window]
    
    return delta_v
