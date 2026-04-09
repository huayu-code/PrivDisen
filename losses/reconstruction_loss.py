"""
Reconstruction loss: ensures [Z_task, Z_private] can recover the
original bottom-model embedding h (information completeness).
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    h_original: torch.Tensor,
    h_reconstructed: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between original embedding and reconstructed embedding.

    Args:
        h_original:      (B, D) — original bottom-model output (detached)
        h_reconstructed: (B, D) — decoder output from [Z_task, Z_private]

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(h_reconstructed, h_original.detach())
