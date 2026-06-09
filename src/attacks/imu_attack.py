"""
IMU Attack Injection Module.

Provides multiple attack types for evaluating SensorTrust's
ability to detect compromised IMU measurements.
"""

import numpy as np


def constant_bias(signal, bias=0.05):
    """
    Add constant bias to IMU signal.
    signal + bias
    same bias added everywhere
    Example:
        yaw_rate += 0.05 rad/s
    attacks it simulates: Calibration error, Sensor offset, Gyroscope bias
    sensor will read 0.05 ras/s even when vehicle stationary

    gps says vehicle moving straight imu says turning, ***F1 will increase***
    """
    return signal + bias


def linear_drift(signal, final_bias=0.1):
    """
    Gradually increasing bias.
    not constant like 0.5 but gradually change from suppose 0.00 to 0.10
    attacks it simulates: Temperature changes, Sensor ages, Bias accumulates and error grows
    harder to detect than constant bias as initially it looks normal and drift not visible
    Simulates sensor calibration drift.
    """
    drift = np.linspace(
        0,
        final_bias,
        len(signal)
    )

    return signal + drift


def gaussian_noise(signal, std=0.02):
    """
    Add Gaussian noise.
    attacks it simulates: Electromagnetic interference, Acoustic attacks, Cheap sensor noise, Bad wiring
    random noise added everywhere
    Simulates noisy IMU measurements.
    """
    noise = np.random.normal(
        0,
        std,
        len(signal)
    )

    return signal + noise


def burst_noise(
        signal,
        start_frame=200,
        duration=30,
        amplitude=0.3
):
    """
    Inject short burst of large noise. Only attacks a short segment.
    not permanent, can detector find short attacks?
    attacks it simulates: Simulates temporary sensor disturbance like EM interference, Vibration, Shock
    Suppose the following frames:
    0-199 clean
    200-230 corrupted
    231-end clean
    """
    attacked = signal.copy()

    end_frame = min(
        start_frame + duration,
        len(signal)
    )

    attacked[start_frame:end_frame] += np.random.normal(
        0,
        amplitude,
        end_frame - start_frame
    )

    return attacked


def scale_factor_attack(
        signal,
        scale=1.2
):
    """
    Scale all measurements.
    signal * scale
    eg: scale=1.2 real=0.10 reported=real*scale=0.12 sensor assumes  turning is 12 deg instead of 10 deg
    attacks it simulates gain calibration error, sensor amplification error
    """
    return signal * scale


def attack_imu_proxies(
        imu_proxies,
        attack_type="drift",
        **kwargs
):
    """
    Apply attack to IMU proxy dictionary.
    """

    attacked = {
        key: value.copy()
        for key, value in imu_proxies.items()
    }

    targets = [
        "delta_v",
        "yaw_rate"
    ]

    for key in targets:

        if key not in attacked:
            continue

        signal = attacked[key]

        if attack_type == "bias":
            attacked[key] = constant_bias(
                signal,
                kwargs.get("bias", 0.05)
            )

        elif attack_type == "drift":
            attacked[key] = linear_drift(
                signal,
                kwargs.get("final_bias", 0.1)
            )

        elif attack_type == "noise":
            attacked[key] = gaussian_noise(
                signal,
                kwargs.get("std", 0.02)
            )

        elif attack_type == "burst":
            attacked[key] = burst_noise(
                signal,
                kwargs.get("start_frame", 200),
                kwargs.get("duration", 30),
                kwargs.get("amplitude", 0.3)
            )

        elif attack_type == "scale":
            attacked[key] = scale_factor_attack(
                signal,
                kwargs.get("scale", 1.2)
            )

        else:
            raise ValueError(
                f"Unknown attack type: {attack_type}"
            )

    return attacked