"""Encoder modules for PrivDisen.

Each data sample x is encoded into two independent latent vectors:
  - z_s: shared (task-relevant) representation
  - z_p: private (sensitive) representation

Both branches output mean and log-variance for the reparameterisation trick
used in the VAE training objective.
"""

import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """Encodes input into the shared (task-relevant) latent space.

    Args:
        input_dim: Dimensionality of the flattened input.
        hidden_dim: Width of the hidden layers.
        shared_dim: Dimensionality of the shared latent vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, shared_dim)
        self.fc_logvar = nn.Linear(hidden_dim, shared_dim)

    def forward(self, x: torch.Tensor):
        """Return (mu, logvar) for the shared representation."""
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class PrivateEncoder(nn.Module):
    """Encodes input into the private (sensitive) latent space.

    Args:
        input_dim: Dimensionality of the flattened input.
        hidden_dim: Width of the hidden layers.
        private_dim: Dimensionality of the private latent vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, private_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, private_dim)
        self.fc_logvar = nn.Linear(hidden_dim, private_dim)

    def forward(self, x: torch.Tensor):
        """Return (mu, logvar) for the private representation."""
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class TaskHead(nn.Module):
    """Linear classifier that operates on the shared representation.

    Args:
        shared_dim: Dimensionality of the shared latent vector.
        num_classes: Number of output classes.
    """

    def __init__(self, shared_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(shared_dim, num_classes)

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.fc(z_s)
