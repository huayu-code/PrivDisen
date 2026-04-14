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
from attacks.norm_attack import _best_cluster_mapping


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

    mapping = _best_cluster_mapping(cluster_ids, y_true, num_classes)
    y_pred = np.array([mapping.get(c, 0) for c in cluster_ids])
    return float(accuracy_score(y_true, y_pred))
