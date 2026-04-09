"""
Reconstruction decoder: ensures Z_task + Z_private can reconstruct
the original bottom-model embedding h (information completeness).
"""

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """
    Decode concatenated [Z_task, Z_private] back to the original
    embedding space of the bottom model.
    """

    def __init__(
        self,
        task_dim: int,
        private_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = task_dim + private_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, z_task: torch.Tensor, z_private: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_task:    (B, task_dim)
            z_private: (B, private_dim)

        Returns:
            h_recon: (B, output_dim)  — reconstruction of original embedding
        """
        z_cat = torch.cat([z_task, z_private], dim=-1)
        return self.decoder(z_cat)
