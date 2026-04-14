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


def _best_cluster_mapping(cluster_ids, y_true, num_classes):
    """
    Find the best mapping from cluster IDs to class labels.
    Uses Hungarian-style greedy assignment to avoid multiple clusters
    mapping to the same class.
    """
    # Build confusion matrix: conf[cluster][class] = count
    n_clusters = num_classes
    conf = np.zeros((n_clusters, num_classes), dtype=int)
    for c_id, y in zip(cluster_ids, y_true):
        if c_id < n_clusters and y < num_classes:
            conf[c_id][y] += 1

    # Greedy assignment: pick the (cluster, class) pair with highest count
    mapping = {}
    used_classes = set()
    # Sort all (cluster, class) pairs by count descending
    pairs = []
    for c in range(n_clusters):
        for k in range(num_classes):
            pairs.append((conf[c][k], c, k))
    pairs.sort(reverse=True)

    for count, c, k in pairs:
        if c not in mapping and k not in used_classes:
            mapping[c] = k
            used_classes.add(k)

    # Fill unmapped clusters
    unused = set(range(num_classes)) - used_classes
    for c in range(n_clusters):
        if c not in mapping:
            if unused:
                mapping[c] = unused.pop()
            else:
                mapping[c] = 0  # fallback

    return mapping


def norm_attack(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> float:
    """
    Norm-based attack: cluster embeddings by L2-norm, then map clusters
    to classes via best assignment.

    Args:
        embeddings: (N, D)
        labels:     (N,)

    Returns:
        Attack accuracy (float).
    """
    norms = torch.norm(embeddings, dim=1).numpy().reshape(-1, 1)
    y_true = labels.numpy()

    kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(norms)

    mapping = _best_cluster_mapping(cluster_ids, y_true, num_classes)
    y_pred = np.array([mapping.get(c, 0) for c in cluster_ids])
    return float(accuracy_score(y_true, y_pred))
