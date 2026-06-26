"""GPS Spoofing Attack Injection.

Implements:
    - gps_step_offset:    Sudden position jump
    - gps_slow_drift:     Gradual position drift below naive detection thresholds
    - gps_speed_spoof:    Constant forward speed offset
    - gps_speed_ramp:     Linearly increasing speed offset (creates delta-v spike)
    - gps_position_spoof: Position shift that causes heading + velocity change
"""
import numpy as np


def gps_step_offset(oxts_data, start_frame, lat_offset=0.001, lon_offset=0.002, duration=None):
    """Sudden GPS position jump (step attack)."""
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            entry['lat'] += lat_offset
            entry['lon'] += lon_offset
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels


def gps_slow_drift(oxts_data, start_frame, drift_per_frame=0.00001, duration=None):
    """Gradual GPS position drift (evasive)."""
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    drift = 0.0
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            drift += drift_per_frame
            entry['lat'] += drift
            entry['lon'] += drift
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels


def gps_speed_spoof(oxts_data, start_frame, speed_offset=5.0, duration=None):
    """Constant forward speed spoofing. Adds fixed offset to vf."""
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            entry['vf'] += speed_offset
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels


def gps_speed_ramp(oxts_data, start_frame, ramp_rate=2.0, duration=None):
    """GPS speed ramp — creates sustained delta-v disagreement with IMU.
    
    Linearly increases forward speed each frame so that the 5-frame
    delta-v window captures the growing gap between GPS-reported
    speed change and IMU-measured acceleration.
    
    Args:
        oxts_data: pykitti oxts list
        start_frame: frame index where ramp begins
        ramp_rate: m/s added per frame (2.0 = +2 m/s every frame)
        duration: number of frames
    
    Returns:
        attacked: list of dicts with spoofed GPS values
        labels: binary array (0=clean, 1=attacked)
    """
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    offset = 0.0
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            offset += ramp_rate
            entry['vf'] = pkt.vf + offset
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels


def gps_position_spoof(oxts_data, start_frame, lat_offset=0.001, lon_offset=0.002,
                       speed_offset=5.0, duration=None):
    """Combined position + speed spoofing attack.
    
    Shifts both position and forward velocity simultaneously.
    Triggers F1 (speed mismatch), F2 (LiDAR scene mismatch), and GMIS.
    """
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            entry['lat'] += lat_offset
            entry['lon'] += lon_offset
            entry['vf'] += speed_offset
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels
