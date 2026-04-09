"""
Norm-based passive label inference attack.

Reference: Fu et al., "Label Inference Attacks Against VFL", USENIX Security 2022.

Idea: The gradient norm received by the passive party differs across classes.
      By clustering gradients by norm, the attacker can infer labels.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def norm_attack(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Norm-based attack: cluster embeddings by L2-norm, then map clusters
    to classes via majority vote.

    Args:
        embeddings: (N, D) — the Z_task received by attacker
        labels:     (N,) — ground-truth labels

    Returns:
        Attack accuracy (float).
    """
    norms = torch.norm(embeddings, dim=1).numpy().reshape(-1, 1)  # (N, 1)
    y_true = labels.numpy()

    kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(norms)

    # Map clusters to labels via majority vote
    cluster_to_label = {}
    for c in range(num_classes):
        mask = cluster_ids == c
        if mask.sum() > 0:
            vals, counts = np.unique(y_true[mask], return_counts=True)
            cluster_to_label[c] = vals[np.argmax(counts)]
        else:
            cluster_to_label[c] = c

    y_pred = np.array([cluster_to_label[c] for c in cluster_ids])
    return float(accuracy_score(y_true, y_pred))
