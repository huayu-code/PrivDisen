"""Decoder module for PrivDisen.

Reconstructs the input from the concatenation of the shared and private
latent vectors: x̂ = Decoder(z_s ‖ z_p).
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Reconstructs input from shared + private representations.

    Args:
        shared_dim: Dimensionality of the shared latent vector.
        private_dim: Dimensionality of the private latent vector.
        hidden_dim: Width of the hidden layers.
        output_dim: Dimensionality of the reconstructed output (= input_dim).
    """

    def __init__(
        self,
        shared_dim: int,
        private_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim + private_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_s: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from shared and private latent codes.

        Args:
            z_s: Shared representation, shape (batch, shared_dim).
            z_p: Private representation, shape (batch, private_dim).

        Returns:
            Reconstructed input, shape (batch, output_dim).
        """
        z = torch.cat([z_s, z_p], dim=-1)
        return self.net(z)
