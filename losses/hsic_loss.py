"""
HSIC (Hilbert-Schmidt Independence Criterion) loss.

Measures statistical dependence between Z_task and Z_private.
HSIC ≈ 0 ⟹ the two representations are statistically independent.
"""

import torch


def _rbf_kernel(X: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF (Gaussian) kernel matrix."""
    # X: (B, D)
    dists = torch.cdist(X, X, p=2).pow(2)  # (B, B)
    return torch.exp(-dists / (2.0 * sigma ** 2))


def hsic_loss(
    z_task: torch.Tensor,
    z_private: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Compute HSIC between Z_task and Z_private using RBF kernels.

    A smaller HSIC means the two representations are more independent.

    Args:
        z_task:    (B, task_dim)
        z_private: (B, private_dim)
        sigma:     RBF kernel bandwidth

    Returns:
        Scalar HSIC value (non-negative).
    """
    n = z_task.shape[0]
    if n < 4:
        return torch.tensor(0.0, device=z_task.device)

    K = _rbf_kernel(z_task, sigma)      # (B, B)
    L = _rbf_kernel(z_private, sigma)   # (B, B)

    # Centering matrix H = I - 1/n * 11^T
    H = torch.eye(n, device=z_task.device) - 1.0 / n

    # HSIC = 1/(n-1)^2 * tr(KHLH)
    KH = K @ H
    LH = L @ H
    hsic = torch.trace(KH @ LH) / ((n - 1) ** 2)

    return hsic
