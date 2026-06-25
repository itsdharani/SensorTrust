"""
LSTM Autoencoder for anomaly detection.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder.
    """

    def __init__(
        self,
        n_features,
        hidden_size=32,
        latent_size=16
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.latent = nn.Linear(
            hidden_size,
            latent_size
        )

        self.expand = nn.Linear(
            latent_size,
            hidden_size
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):
        """
        x shape:
        (batch, seq_len, n_features)
        """

        _, (hidden, _) = self.encoder(x)

        hidden = hidden[-1]

        latent = self.latent(hidden)

        decoded_hidden = self.expand(latent)

        repeated = decoded_hidden.unsqueeze(1).repeat(
            1,
            x.size(1),
            1
        )

        reconstructed, _ = self.decoder(repeated)

        return reconstructed