"""PrivDisen: full model combining all sub-modules.

This module wires together the shared encoder, private encoder, task head,
decoder, and privacy discriminator into a single ``nn.Module`` with a
convenient ``forward`` that returns all quantities needed for loss computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import PrivateEncoder, SharedEncoder, TaskHead
from .decoder import Decoder
from .discriminator import PrivacyDiscriminator


def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Apply the VAE reparameterisation trick.

    z = mu + eps * std,  eps ~ N(0, I)

    Args:
        mu: Mean of the approximate posterior, shape (batch, dim).
        logvar: Log-variance, shape (batch, dim).

    Returns:
        Sampled latent vector with the same shape as *mu*.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class PrivDisen(nn.Module):
    """Privacy-Preserving Disentangled Representation Learning model.

    Args:
        input_dim: Dimensionality of the flattened input.
        hidden_dim: Width of the hidden layers shared across sub-networks.
        shared_dim: Dimensionality of the shared (task) latent vector.
        private_dim: Dimensionality of the private (sensitive) latent vector.
        num_classes: Number of downstream task classes.
        num_sensitive: Number of sensitive attribute classes.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        shared_dim: int = 64,
        private_dim: int = 64,
        num_classes: int = 10,
        num_sensitive: int = 2,
    ) -> None:
        super().__init__()
        self.shared_encoder = SharedEncoder(input_dim, hidden_dim, shared_dim)
        self.private_encoder = PrivateEncoder(input_dim, hidden_dim, private_dim)
        self.task_head = TaskHead(shared_dim, num_classes)
        self.decoder = Decoder(shared_dim, private_dim, hidden_dim, input_dim)
        self.discriminator = PrivacyDiscriminator(shared_dim, num_sensitive)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def encoder_parameters(self):
        """Parameters belonging to the encoding side (excluding discriminator)."""
        return (
            list(self.shared_encoder.parameters())
            + list(self.private_encoder.parameters())
            + list(self.task_head.parameters())
            + list(self.decoder.parameters())
        )

    @property
    def discriminator_parameters(self):
        """Parameters of the privacy discriminator only."""
        return list(self.discriminator.parameters())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """Run a full forward pass.

        Args:
            x: Input tensor, shape (batch, input_dim).

        Returns:
            A dict with the following keys:

            - ``x_recon``: Reconstructed input, shape (batch, input_dim).
            - ``y_pred``: Task logits, shape (batch, num_classes).
            - ``s_pred``: Sensitive attribute logits from z_s (adversarial),
              shape (batch, num_sensitive).
            - ``z_s``, ``z_p``: Sampled latent vectors.
            - ``mu_s``, ``logvar_s``: Shared encoder posterior parameters.
            - ``mu_p``, ``logvar_p``: Private encoder posterior parameters.
        """
        # Encode into shared and private posteriors
        mu_s, logvar_s = self.shared_encoder(x)
        mu_p, logvar_p = self.private_encoder(x)

        # Sample latent codes
        z_s = reparameterise(mu_s, logvar_s)
        z_p = reparameterise(mu_p, logvar_p)

        # Decode / predict
        x_recon = self.decoder(z_s, z_p)
        y_pred = self.task_head(z_s)
        s_pred = self.discriminator(z_s)

        return {
            "x_recon": x_recon,
            "y_pred": y_pred,
            "s_pred": s_pred,
            "z_s": z_s,
            "z_p": z_p,
            "mu_s": mu_s,
            "logvar_s": logvar_s,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
        }
