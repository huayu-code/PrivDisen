"""
Mutual Information upper-bound loss based on KL divergence (VIB).

I(Z_task; X) <= KL( q(Z_task|X) || p(Z_task) )

where p(Z_task) = N(0, I) is the standard-normal prior.
By the Data Processing Inequality, I(Z_task; Y) <= I(Z_task; X),
so bounding I(Z_task; X) also bounds label leakage.
"""

import torch


def mi_loss(task_mu: torch.Tensor, task_logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from q(Z_task|X) = N(μ, σ²) to p(Z_task) = N(0, I).

    KL = 0.5 * Σ (μ² + σ² - log(σ²) - 1)

    Args:
        task_mu:     (B, task_dim)
        task_logvar: (B, task_dim)  — log of variance

    Returns:
        Scalar KL divergence, averaged over batch.
    """
    kl = -0.5 * torch.mean(
        1.0 + task_logvar - task_mu.pow(2) - task_logvar.exp()
    )
    return kl
