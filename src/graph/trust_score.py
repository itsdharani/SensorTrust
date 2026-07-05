import numpy as np

BETA = 0.7


def compute_node_inconsistency(graph):
    edges = {
        "gps":    ["gps_imu", "gps_lidar", "gps_camera"],
        "imu":    ["gps_imu", "imu_lidar", "imu_camera"],
        "lidar":  ["gps_lidar", "imu_lidar", "lidar_camera"],
        "camera": ["gps_camera", "imu_camera", "lidar_camera"],
    }
    return {
        sensor: np.median([graph[e] for e in edge_list], axis=0)
        for sensor, edge_list in edges.items()
    }


def compute_trust_scores(inconsistency):

    return {
        sensor: float(np.exp(-BETA * np.nanmean(value)))
        for sensor, value in inconsistency.items()
    }
