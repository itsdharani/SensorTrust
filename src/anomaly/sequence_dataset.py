"""
Sequence generation utilities for LSTM Autoencoder.
"""

import numpy as np


def create_sequences(X, seq_len=20):
    """
    Convert feature matrix into overlapping sequences.

    Args:
        X : np.ndarray, shape (N, F)
            Feature matrix

        seq_len : int
            Sequence length

    Returns:
        np.ndarray, shape (N-seq_len+1, seq_len, F)
    """

    sequences = []

    for i in range(len(X) - seq_len + 1):
        sequences.append(
            X[i:i + seq_len]
        )

    return np.array(sequences)