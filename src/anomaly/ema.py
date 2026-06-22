"""
EMA Adaptive Baseline
"""

import numpy as np


class EMABaseline:

    def __init__(
            self,
            alpha=0.05
    ):

        self.alpha = alpha

        self.mean = None
        self.var = None

    def update(
            self,
            x
    ):

        if self.mean is None:

            self.mean = x
            self.var = 0.0

            return (
                self.mean,
                self.var
            )

        self.mean = (
            self.alpha * x
            +
            (1-self.alpha)
            * self.mean
        )

        self.var = (
            self.alpha
            * (x-self.mean)**2
            +
            (1-self.alpha)
            * self.var
        )

        return (
            self.mean,
            self.var
        )

    def residual(
            self,
            x
    ):

        return abs(
            x - self.mean
        )