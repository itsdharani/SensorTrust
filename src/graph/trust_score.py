import numpy as np

BETA = 0.7


def compute_node_inconsistency(graph):

    inconsistency = {

        "gps":
            graph["gps_imu"]
            + graph["gps_lidar"]
            + graph["gps_camera"],

        "imu":
            graph["gps_imu"]
            + graph["imu_lidar"]
            + graph["imu_camera"],

        "lidar":
            graph["gps_lidar"]
            + graph["imu_lidar"]
            + graph["lidar_camera"],

        "camera":
            graph["gps_camera"]
            + graph["imu_camera"]
            + graph["lidar_camera"]
    }

    return inconsistency


def compute_trust_scores(
        inconsistency
):

    trust = {}

    for sensor, value in inconsistency.items():

        trust[sensor] = np.exp(
            -BETA * value
        )

    return trust