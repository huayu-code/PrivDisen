"""
Attack evaluation pipeline.
Runs specified attacks against trained models and collects results.
"""

from typing import Dict, List

import torch

from attacks import ATTACK_REGISTRY


def evaluate_attacks(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    attack_names: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Run multiple attacks and return results.

    Args:
        embeddings: (N, D) — Z_task embeddings
        labels:     (N,)
        num_classes: number of classes
        attack_names: list of attack names
        device:     device for model-based attacks

    Returns:
        Dict mapping attack_name -> accuracy (ASR).
    """
    results = {}
    for name in attack_names:
        if name not in ATTACK_REGISTRY:
            raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY.keys())}")

        fn = ATTACK_REGISTRY[name]

        # Passive attacks (norm, direction) take 3 args
        if name in ("norm", "direction"):
            asr = fn(embeddings, labels, num_classes)
        else:
            # Model-based attacks take extra kwargs
            asr = fn(embeddings, labels, num_classes, device=device)

        results[name] = asr

    return results
