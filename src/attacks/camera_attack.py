"""
camera contributes to optical flow which estimates motion from images
Camera Attack Injection Module.

Provides image-level attacks for evaluating
SensorTrust's camera motion proxy robustness.
"""

import cv2
import numpy as np


def gaussian_noise(
        image,
        std=15
):
    """
    Add Gaussian image noise.
    image + noise produces grainy images
    simulates bad lighting, electronic noise, adversarial corruption
    """
    noise = np.random.normal(
        0,
        std,
        image.shape
    )

    attacked = image.astype(np.float32) + noise

    return np.clip(
        attacked,
        0,
        255
    ).astype(np.uint8)


def salt_pepper_noise(
        image,
        probability=0.01
):
    """
    Salt and pepper corruption
    random pixels attacked, become black and white (0 or 255)
    """
    attacked = image.copy()

    rnd = np.random.rand(
        image.shape[0],
        image.shape[1]
    )

    attacked[rnd < probability / 2] = 0
    attacked[rnd > 1 - probability / 2] = 255

    return attacked


def brightness_shift(
        image,
        shift=40
):
    """
    Global brightness manipulation.
    image += shift image becomes too dark or too bright
    """
    attacked = image.astype(np.int16)

    attacked += shift

    return np.clip(
        attacked,
        0,
        255
    ).astype(np.uint8)


def motion_blur(
        image,
        kernel_size=15
):
    """
    Simulates fast motion causing motion blur, smeared image
    """
    kernel = np.zeros(
        (kernel_size, kernel_size)
    )

    kernel[
        kernel_size // 2,
        :
    ] = np.ones(kernel_size)

    kernel /= kernel_size

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def occlusion(
        image,
        x=200,
        y=100,
        w=200,
        h=150
):
    """
    Add black rectangular occlusion (like a black rectangle added to image) covers part of the lens
    simulates mud or tape or some physical obstruction
    """
    attacked = image.copy()

    attacked[
        y:y+h,
        x:x+w
    ] = 0

    return attacked


def freeze_frames(
        frames,
        start_frame=200,
        duration=20
):
    """
    Freeze video stream.
    suppose vehicle is moving but camera shows no motion nothing changed because it keeps showing same frame
    """

    attacked = [
        frame.copy()
        for frame in frames
    ]

    frozen = attacked[start_frame].copy()

    end_frame = min(
        start_frame + duration,
        len(attacked)
    )

    for i in range(
            start_frame,
            end_frame
    ):
        attacked[i] = frozen.copy()

    return attacked


def drop_frames(
        frames,
        interval=5
):
    """
    Periodically duplicate frames.
    randomly drop the other frames vehicle seems to be stationary sometimes speeding
    simulates network loss dropped packets (frames?) transmission failures
    """

    attacked = [
        frame.copy()
        for frame in frames
    ]

    for i in range(
            interval,
            len(attacked)
    ):
        attacked[i] = attacked[i - 1].copy()

    return attacked