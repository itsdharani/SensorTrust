"""
Load and validate KITTI raw data sequences.
"""

import os
import pykitti
import numpy as np

from src.utils import (
    get_kitti_base_path,
    get_sync_path,
    load_timestamps_from_file
)


class KITTILoader:
    """
    Loads and validates a KITTI raw sequence.
    """

    def __init__(
        self,
        date="2011_09_26",
        drive="0009"
    ):

        self.date = date
        self.drive = drive

        self.base_path = get_kitti_base_path()

        self.raw_data = pykitti.raw(
            base_path=str(self.base_path),
            date=self.date,
            drive=self.drive
        )

        self.sync_path = get_sync_path(
            self.date,
            self.drive
        )

        self.timestamps = self._load_all_timestamps()

    # --------------------------------------------------
    # Timestamp loading
    # --------------------------------------------------

    def _load_all_timestamps(self):

        return {

            "oxts":
                load_timestamps_from_file(
                    os.path.join(
                        self.sync_path,
                        "oxts",
                        "timestamps.txt"
                    )
                ),

            "velo":
                load_timestamps_from_file(
                    os.path.join(
                        self.sync_path,
                        "velodyne_points",
                        "timestamps.txt"
                    )
                ),

            "cam2":
                load_timestamps_from_file(
                    os.path.join(
                        self.sync_path,
                        "image_02",
                        "timestamps.txt"
                    )
                )
        }

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def get_statistics(self):

        return {

            "drive":
                self.drive,

            "num_gps_imu":
                len(self.raw_data.oxts),

            "num_lidar":
                len(self.raw_data.velo_files),

            "num_camera":
                len(self.raw_data.cam2_files),

            "gps_imu_hz":
                1.0 /
                np.mean(
                    np.diff(
                        self.timestamps["oxts"]
                    )
                ),

            "lidar_hz":
                1.0 /
                np.mean(
                    np.diff(
                        self.timestamps["velo"]
                    )
                ),

            "camera_hz":
                1.0 /
                np.mean(
                    np.diff(
                        self.timestamps["cam2"]
                    )
                )
        }

    # --------------------------------------------------
    # dt
    # --------------------------------------------------

    def get_dt(self):

        return np.mean(
            np.diff(
                self.timestamps["oxts"]
            )
        )

    # --------------------------------------------------
    # GPS / IMU
    # --------------------------------------------------

    def get_gps_reading(self, index):

        frame = self.raw_data.oxts[index]

        return {

            "lat": frame.packet.lat,
            "lon": frame.packet.lon,
            "alt": frame.packet.alt,

            "vf": frame.packet.vf,
            "vl": frame.packet.vl,
            "vu": frame.packet.vu,

            "ax": frame.packet.ax,
            "ay": frame.packet.ay,
            "az": frame.packet.az,

            "wx": frame.packet.wx,
            "wy": frame.packet.wy,
            "wz": frame.packet.wz,

            "roll": frame.packet.roll,
            "pitch": frame.packet.pitch,
            "yaw": frame.packet.yaw
        }

    # --------------------------------------------------
    # LiDAR
    # --------------------------------------------------

    def get_lidar_scan(self, index):

        return self.raw_data.get_velo(index)

    # --------------------------------------------------
    # Camera
    # --------------------------------------------------

    def get_camera_image(
        self,
        index,
        camera="cam2"
    ):

        if camera == "cam2":
            return self.raw_data.get_cam2(index)

        elif camera == "cam3":
            return self.raw_data.get_cam3(index)

        raise ValueError(
            f"Unknown camera: {camera}"
        )

    # --------------------------------------------------
    # MULTI DRIVE LOADER
    # --------------------------------------------------

    @staticmethod
    def load_multiple_drives(
        date="2011_09_26",
        drives=None
    ):

        if drives is None:

            drives = [
                "0009",
                "0015",
                "0051",
                "0091"
            ]

        datasets = {}

        for drive in drives:

            print(
                f"Loading drive {drive}..."
            )

            datasets[drive] = KITTILoader(
                date=date,
                drive=drive
            )

        print(
            f"\nLoaded {len(datasets)} drives."
        )

        return datasets


if __name__ == "__main__":

    loader = KITTILoader()

    print(
        loader.get_statistics()
    )