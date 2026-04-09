"""
Gradient Purification Module (optional enhancement).

Projects back-propagated gradients to remove label-correlated components,
providing backward-path protection in addition to forward-path (VDM).
"""

import torch
import torch.nn as nn


class GradientPurifier(nn.Module):
    """
    Removes the label-direction component from gradients via projection.

    Maintains an exponential moving average of per-class gradient centroids,
    then projects out the label-discriminative direction.
    """

    def __init__(self, dim: int, num_classes: int, momentum: float = 0.9):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.momentum = momentum

        # Running centroid per class: (num_classes, dim)
        self.register_buffer(
            "centroids", torch.zeros(num_classes, dim)
        )
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update_centroids(self, grads: torch.Tensor, labels: torch.Tensor):
        """Update per-class gradient centroids with EMA."""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            class_mean = grads[mask].mean(dim=0)
            if not self.initialized:
                self.centroids[c] = class_mean
            else:
                self.centroids[c] = (
                    self.momentum * self.centroids[c]
                    + (1 - self.momentum) * class_mean
                )
        self.initialized.fill_(True)

    def purify(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Remove the label-discriminative direction from gradients.

        The label direction v_y is estimated as the first principal
        component of the class centroids.
        """
        if not self.initialized:
            return grads

        # Compute label direction via SVD on centroids
        centered = self.centroids - self.centroids.mean(dim=0, keepdim=True)
        # (num_classes, dim) -> SVD -> first right singular vector
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            v_y = Vh[0]  # first principal direction, shape (dim,)
        except RuntimeError:
            return grads

        # Project out: g' = g - (g · v_y) * v_y
        proj = torch.einsum("bd, d -> b", grads, v_y).unsqueeze(-1) * v_y.unsqueeze(0)
        return grads - proj
