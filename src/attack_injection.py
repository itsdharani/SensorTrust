import numpy as np

def gps_step_drift(oxts_data, start_frame, lat_offset=0.001, lon_offset=0.002):
    """
    GPS step-function drift attack.
    Adds a sudden fixed offset to latitude and longitude from start_frame onwards.
    Returns modified oxts data and label array.
    """
    modified = []
    labels = []
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        if i >= start_frame:
            new_lat = pkt.lat + lat_offset
            new_lon = pkt.lon + lon_offset
            labels.append(1)   # 1 = GPS attacked
        else:
            new_lat = pkt.lat
            new_lon = pkt.lon
            labels.append(0)   # 0 = clean
        modified.append({'lat': new_lat, 'lon': new_lon,
                         'vf': pkt.vf, 'ax': pkt.ax, 'ay': pkt.ay})
    return modified, np.array(labels)


def gps_ramp_drift(oxts_data, start_frame, drift_per_frame=0.00001):
    """
    GPS slow-ramp drift attack (evasive).
    Gradually shifts GPS coordinates — invisible to naive thresholds.
    """
    modified = []
    labels = []
    drift = 0.0
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        if i >= start_frame:
            drift += drift_per_frame
            labels.append(1)
        else:
            labels.append(0)
        modified.append({'lat': pkt.lat + drift, 'lon': pkt.lon + drift,
                         'vf': pkt.vf, 'ax': pkt.ax, 'ay': pkt.ay})
    return modified, np.array(labels)


def imu_gaussian_noise(oxts_data, start_frame, std=0.5):
    """
    IMU Gaussian noise injection.
    Adds random noise to acceleration values.
    """
    modified = []
    labels = []
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        if i >= start_frame:
            noise_x = np.random.normal(0, std)
            noise_y = np.random.normal(0, std)
            labels.append(2)   # 2 = IMU attacked
        else:
            noise_x = 0
            noise_y = 0
            labels.append(0)
        modified.append({'lat': pkt.lat, 'lon': pkt.lon,
                         'vf': pkt.vf,
                         'ax': pkt.ax + noise_x,
                         'ay': pkt.ay + noise_y})
    return modified, np.array(labels)


def lidar_phantom_inject(scan, n_points=100, distance=5.0):
    """
    LiDAR phantom obstacle injection.
    Inserts a cluster of fake points at a fixed distance ahead.
    scan: numpy array of shape (N, 4) — x, y, z, intensity
    Returns modified scan.
    """
    phantom = np.zeros((n_points, 4))
    phantom[:, 0] = distance                          # x = ahead
    phantom[:, 1] = np.random.uniform(-0.5, 0.5, n_points)  # y = width
    phantom[:, 2] = np.random.uniform(0, 1.5, n_points)     # z = height
    phantom[:, 3] = 0.5                               # intensity
    return np.vstack([scan, phantom])


def coordinated_gps_imu(oxts_data, start_frame):
    """
    Coordinated GPS + IMU attack.
    GPS drifts slowly; IMU bias partially matches it — each looks individually plausible.
    """
    modified = []
    labels = []
    drift = 0.0
    bias = 0.0
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        if i >= start_frame:
            drift += 0.00001
            bias += 0.001
            labels.append(5)   # 5 = coordinated GPS+IMU
        else:
            labels.append(0)
        modified.append({'lat': pkt.lat + drift, 'lon': pkt.lon + drift,
                         'vf': pkt.vf,
                         'ax': pkt.ax + bias,
                         'ay': pkt.ay})
    return modified, np.array(labels)
