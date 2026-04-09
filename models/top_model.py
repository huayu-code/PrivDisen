"""
Top model for the active party.

Takes concatenated Z_task from all passive parties, fuses them, and predicts.
"""

import torch
import torch.nn as nn


class TopModel(nn.Module):
    """Active party's top model: fusion + classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: concatenated Z_task from all passive parties, shape (B, D_total)
        Returns:
            logits, shape (B, num_classes)
        """
        return self.classifier(z)
