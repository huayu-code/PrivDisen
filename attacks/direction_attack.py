"""
Direction-based passive label inference attack.

Reference: Fu et al., "Label Inference Attacks Against VFL", USENIX Security 2022.

Idea: The gradient direction (cosine similarity) is more discriminative
      than raw norms. Cluster embeddings in direction space.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


def direction_attack(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Direction-based attack: normalize embeddings to unit sphere,
    then cluster and map to labels.

    Args:
        embeddings: (N, D)
        labels:     (N,)

    Returns:
        Attack accuracy (float).
    """
    # L2 normalize
    emb_norm = embeddings / (torch.norm(embeddings, dim=1, keepdim=True) + 1e-8)
    emb_np = emb_norm.numpy()
    y_true = labels.numpy()

    kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(emb_np)

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
