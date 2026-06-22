"""
Mahalanobis Anomaly Detector

Input:
    F1
    F2
    GMIS

Output:
    Mahalanobis distance per frame
"""

import numpy as np


class MahalanobisDetector:

    def __init__(self):
        self.mu = None
        self.cov_inv = None
        self.fitted = False

    def fit(self, f1, f2, gmis):

        X = np.column_stack([
            f1,
            f2,
            gmis
        ])

        X = X[~np.isnan(X).any(axis=1)]

        self.mu = np.mean(X, axis=0)

        cov = np.cov(
            X,
            rowvar=False
        )

        cov += np.eye(
            cov.shape[0]
        ) * 1e-6

        self.cov_inv = np.linalg.inv(cov)

        self.fitted = True

        print(
            f"Mahalanobis fitted on "
            f"{len(X)} samples"
        )

    def score(self, f1, f2, gmis):

        if not self.fitted:
            raise RuntimeError(
                "Call fit() first."
            )

        X = np.column_stack([
            f1,
            f2,
            gmis
        ])

        scores = np.full(
            len(X),
            np.nan
        )

        for i, x in enumerate(X):

            if np.isnan(x).any():
                continue

            d = x - self.mu

            scores[i] = np.sqrt(
                d.T
                @ self.cov_inv
                @ d
            )

        return scores