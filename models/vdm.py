"""
Variational Disentanglement Module (VDM).

Core innovation of PrivDisen: splits bottom-model embeddings into
  - Z_task:    task-relevant representation (transmitted to active party)
  - Z_private: label-sensitive representation (kept locally)

Uses reparameterized Gaussian for soft, differentiable separation.
"""

import torch
import torch.nn as nn


class VariationalDisentangleModule(nn.Module):
    """
    Variational Disentanglement Module.

    Given a bottom-model embedding h ∈ R^{input_dim}, produces:
      - Z_task    ~ N(μ_task, σ²_task)   — transmitted
      - Z_private ~ N(μ_priv, σ²_priv)   — kept local
      - distribution params for MI / KL losses

    Architecture:
      h → shared_encoder → [task_branch, private_branch]
                            each branch outputs (μ, log σ²)
    """

    def __init__(
        self,
        input_dim: int,
        task_dim: int = 128,
        private_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.task_dim = task_dim
        self.private_dim = private_dim

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Task branch: μ and log(σ²)
        self.task_mu = nn.Linear(hidden_dim, task_dim)
        self.task_logvar = nn.Linear(hidden_dim, task_dim)

        # Private branch: μ and log(σ²)
        self.priv_mu = nn.Linear(hidden_dim, private_dim)
        self.priv_logvar = nn.Linear(hidden_dim, private_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε, ε ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: bottom-model embedding, shape (B, input_dim)

        Returns:
            z_task:    (B, task_dim)
            z_private: (B, private_dim)
            dist_params: dict with keys
                'task_mu', 'task_logvar', 'priv_mu', 'priv_logvar'
        """
        shared = self.shared_encoder(h)

        # Task representation
        t_mu = self.task_mu(shared)
        t_logvar = self.task_logvar(shared)
        z_task = self.reparameterize(t_mu, t_logvar)

        # Private representation
        p_mu = self.priv_mu(shared)
        p_logvar = self.priv_logvar(shared)
        z_private = self.reparameterize(p_mu, p_logvar)

        dist_params = {
            "task_mu": t_mu,
            "task_logvar": t_logvar,
            "priv_mu": p_mu,
            "priv_logvar": p_logvar,
        }

        return z_task, z_private, dist_params
