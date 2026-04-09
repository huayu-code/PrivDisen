"""
Adversarial Label Classifier (ALC) with Gradient Reversal Layer (GRL).

The ALC tries to predict labels from Z_task.
The GRL reverses gradients during backprop so that the bottom model + VDM
learn to produce Z_task that *fools* the ALC (= removes label signal).
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function


# ======================================================================
# Gradient Reversal Layer
# ======================================================================

class _GradientReversalFn(Function):
    """Reverse gradients scaled by lambda during backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps _GradientReversalFn as an nn.Module."""

    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambda_)


# ======================================================================
# Adversarial Label Classifier
# ======================================================================

class AdversarialLabelClassifier(nn.Module):
    """
    MLP classifier that tries to infer labels from Z_task.
    Trained WITH gradient reversal — so the encoder learns to
    make Z_task label-uninformative.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.grl = GradientReversalLayer()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambda_: float):
        """Update GRL reversal strength."""
        self.grl.set_lambda(lambda_)

    def forward(self, z_task: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_task: (B, task_dim) — task representation from VDM

        Returns:
            logits: (B, num_classes)
        """
        reversed_z = self.grl(z_task)
        return self.classifier(reversed_z)


# ======================================================================
# Alpha scheduling utilities
# ======================================================================

def compute_alpha_dann(epoch: int, max_epoch: int, gamma: float = 10.0) -> float:
    """DANN-style progressive alpha: 0 → 1 with sigmoid schedule."""
    p = epoch / max_epoch
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def compute_alpha_linear(epoch: int, max_epoch: int) -> float:
    """Linear ramp from 0 to 1."""
    return min(1.0, epoch / max_epoch)


def compute_alpha(
    epoch: int, max_epoch: int, schedule: str = "dann", alpha_max: float = 1.0
) -> float:
    """Compute alpha (adversarial strength) for the current epoch."""
    if schedule == "dann":
        raw = compute_alpha_dann(epoch, max_epoch)
    elif schedule == "linear":
        raw = compute_alpha_linear(epoch, max_epoch)
    elif schedule == "constant":
        raw = 1.0
    else:
        raise ValueError(f"Unknown alpha schedule: {schedule}")
    return raw * alpha_max
