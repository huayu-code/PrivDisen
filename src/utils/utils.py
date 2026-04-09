"""Utility helpers for PrivDisen."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the ``.yaml`` config file.

    Returns:
        Nested dict with configuration values.
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    state: dict,
    checkpoint_dir: str | Path,
    filename: str = "checkpoint.pt",
) -> None:
    """Save a training checkpoint.

    Args:
        state: Dict containing model state, optimizer state, epoch, etc.
        checkpoint_dir: Directory where the checkpoint will be saved.
        filename: Name of the checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_dir / filename)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint ``.pt`` file.
        device: Device to map tensors onto.

    Returns:
        The saved state dict.
    """
    return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# Accuracy / metrics
# ---------------------------------------------------------------------------


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy.

    Args:
        logits: Model output logits, shape (batch, num_classes).
        labels: Ground-truth labels, shape (batch,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def average_meter(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of floats."""
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------
# AverageMeter class (useful for tracking running averages during training)
# ---------------------------------------------------------------------------


class AverageMeter:
    """Track and compute the running average of a scalar value."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.name}: {self.avg:.4f}"
