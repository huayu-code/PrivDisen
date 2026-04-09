"""Privacy discriminator for PrivDisen.

The discriminator tries to predict the sensitive attribute from the *shared*
representation z_s.  By training the main encoder adversarially against this
discriminator we suppress sensitive information from z_s (adversarial privacy).
"""

import torch
import torch.nn as nn


class PrivacyDiscriminator(nn.Module):
    """Predicts sensitive attribute from the shared representation.

    Used adversarially: the encoder minimises the discriminator's accuracy
    while the discriminator is simultaneously trained to maximise it.

    Args:
        shared_dim: Dimensionality of the shared latent vector.
        num_sensitive: Number of sensitive attribute classes.
    """

    def __init__(self, shared_dim: int, num_sensitive: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, num_sensitive),
        )

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        """Predict sensitive class logits from shared representation.

        Args:
            z_s: Shared representation, shape (batch, shared_dim).

        Returns:
            Logits over sensitive classes, shape (batch, num_sensitive).
        """
        return self.net(z_s)
