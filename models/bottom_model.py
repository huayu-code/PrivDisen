"""
Bottom models for passive parties.

- MLP for tabular data
- CNN for image data (operates on flattened VFL partitions)
"""

import torch
import torch.nn as nn


class BottomModelMLP(nn.Module):
    """Simple MLP bottom model for tabular / flattened features."""

    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BottomModelCNN(nn.Module):
    """
    Small CNN bottom model.
    Since VFL partitions images into flattened vectors, we first reshape,
    then apply conv layers.

    This model assumes the input is a flattened partition of an image.
    For simplicity, we use an MLP-like architecture with larger capacity.
    """

    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_bottom_model(
    model_type: str, input_dim: int, output_dim: int = 128
) -> nn.Module:
    """Factory for bottom models."""
    if model_type == "mlp":
        return BottomModelMLP(input_dim, output_dim)
    elif model_type == "cnn":
        return BottomModelCNN(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown bottom model type: {model_type}")
