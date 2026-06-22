"""
Global anomaly detector
"""

import numpy as np


def detect_anomalies(
        mahalanobis_scores,
        threshold=3.0
):

    alerts = (
        mahalanobis_scores
        >
        threshold
    )

    return alerts