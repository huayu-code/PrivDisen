"""
Standard task loss (cross-entropy).
"""

import torch
import torch.nn.functional as F


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for the main classification task."""
    return F.cross_entropy(logits, targets)
