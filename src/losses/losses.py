"""Loss functions for PrivDisen.

The total training objective is:

    L = lambda_recon * L_recon
      + alpha        * L_task
      + beta         * L_kl
      + gamma        * L_mi
      - delta        * L_adv   ← encoder *minimises* adversarial loss

The discriminator is trained separately to *maximise* L_adv (cross-entropy on
the sensitive attribute prediction from z_s).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------


def reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy reconstruction loss (element-mean).

    Assumes both *x* and *x_recon* are in [0, 1].

    Args:
        x: Original input, shape (batch, input_dim).
        x_recon: Reconstructed input, same shape.

    Returns:
        Scalar loss.
    """
    return F.binary_cross_entropy(x_recon, x, reduction="mean")


def kl_divergence(
    mu: torch.Tensor, logvar: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor
) -> torch.Tensor:
    """KL divergence KL(q(z|x) ‖ p(z)) summed over all latent dimensions.

    For a standard Gaussian prior p(z)=N(0,I) the closed-form is:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Both the shared and private posteriors are included.

    Args:
        mu: Shared encoder mean.
        logvar: Shared encoder log-variance.
        mu2: Private encoder mean.
        logvar2: Private encoder log-variance.

    Returns:
        Scalar KL loss (mean over batch).
    """
    kl_s = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_p = -0.5 * torch.mean(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    return kl_s + kl_p


def task_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Cross-entropy task classification loss.

    Args:
        y_pred: Logits, shape (batch, num_classes).
        y_true: Ground-truth class indices, shape (batch,).

    Returns:
        Scalar loss.
    """
    return F.cross_entropy(y_pred, y_true)


def mutual_information_penalty(z_s: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
    """Approximate mutual information penalty between z_s and z_p.

    We use the squared cosine similarity as a differentiable proxy for
    linear dependence between the two representations.  A value of 0 means
    the representations are orthogonal (fully disentangled).

    Args:
        z_s: Shared latent, shape (batch, shared_dim).
        z_p: Private latent, shape (batch, private_dim).

    Returns:
        Scalar penalty (mean over batch).
    """
    # Project z_p to z_s space via a dot-product if dims differ
    min_dim = min(z_s.size(-1), z_p.size(-1))
    z_s_trunc = z_s[..., :min_dim]
    z_p_trunc = z_p[..., :min_dim]

    z_s_norm = F.normalize(z_s_trunc, dim=-1)
    z_p_norm = F.normalize(z_p_trunc, dim=-1)
    cosine_sim = (z_s_norm * z_p_norm).sum(dim=-1)  # shape (batch,)
    return cosine_sim.pow(2).mean()


def adversarial_privacy_loss(
    s_pred: torch.Tensor, s_true: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy loss for the sensitive attribute prediction.

    *Discriminator* step : maximise this (standard CE optimisation).
    *Encoder* step       : minimise the negative of this (i.e. subtract it
                           from the total loss with weight ``delta``).

    Args:
        s_pred: Logits from the privacy discriminator, shape (batch, num_sensitive).
        s_true: Ground-truth sensitive labels, shape (batch,).

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(s_pred, s_true)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------


def compute_total_loss(
    outputs: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    s: torch.Tensor,
    *,
    lambda_recon: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.5,
    delta: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Compute all loss components and aggregate the encoder total loss.

    Note: The discriminator is trained separately using only
    ``adversarial_privacy_loss``.

    Args:
        outputs: Dict returned by ``PrivDisen.forward``.
        x: Original input, shape (batch, input_dim).
        y: Task labels, shape (batch,).
        s: Sensitive labels, shape (batch,).
        lambda_recon: Weight for reconstruction loss.
        alpha: Weight for task loss.
        beta: Weight for KL divergence.
        gamma: Weight for mutual information penalty.
        delta: Weight for adversarial privacy loss (subtracted from total).

    Returns:
        Dict with individual losses and the aggregated ``total`` encoder loss.
    """
    l_recon = reconstruction_loss(x, outputs["x_recon"])
    l_task = task_loss(outputs["y_pred"], y)
    l_kl = kl_divergence(
        outputs["mu_s"], outputs["logvar_s"],
        outputs["mu_p"], outputs["logvar_p"],
    )
    l_mi = mutual_information_penalty(outputs["z_s"], outputs["z_p"])
    l_adv = adversarial_privacy_loss(outputs["s_pred"], s)

    total = (
        lambda_recon * l_recon
        + alpha * l_task
        + beta * l_kl
        + gamma * l_mi
        - delta * l_adv   # encoder wants discriminator to fail
    )

    return {
        "total": total,
        "recon": l_recon,
        "task": l_task,
        "kl": l_kl,
        "mi": l_mi,
        "adv": l_adv,
    }
