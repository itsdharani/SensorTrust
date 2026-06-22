import numpy as np


def build_disagreement_graph(
        z_gps,
        z_imu,
        z_lidar,
        z_camera
):

    return {

        "gps_imu":
            np.abs(z_gps - z_imu),

        "gps_lidar":
            np.abs(z_gps - z_lidar),

        "gps_camera":
            np.abs(z_gps - z_camera),

        "imu_lidar":
            np.abs(z_imu - z_lidar),

        "imu_camera":
            np.abs(z_imu - z_camera),

        "lidar_camera":
            np.abs(z_lidar - z_camera)
    }