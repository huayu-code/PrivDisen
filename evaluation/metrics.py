"""
Evaluation metrics for PrivDisen experiments.
"""

from typing import Dict


def compute_metrics(
    test_acc: float,
    attack_results: Dict[str, float],
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute privacy-utility metrics.

    Args:
        test_acc:       main task accuracy (MTA)
        attack_results: dict mapping attack_name -> attack_accuracy (ASR)
        num_classes:    for computing random-guess baseline

    Returns:
        Dict with MTA, ASR per attack, average ASR, PUT (privacy-utility trade-off).
    """
    random_guess = 1.0 / num_classes
    metrics = {"MTA": test_acc}

    asr_values = []
    for attack_name, asr in attack_results.items():
        metrics[f"ASR_{attack_name}"] = asr
        asr_values.append(asr)

    avg_asr = sum(asr_values) / len(asr_values) if asr_values else 0.0
    metrics["ASR_avg"] = avg_asr
    metrics["random_guess"] = random_guess

    # Privacy-Utility Trade-off: higher is better
    # PUT = MTA / max(ASR_avg, random_guess) to avoid division by near-zero
    metrics["PUT"] = test_acc / max(avg_asr, random_guess)

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Pretty-print metrics."""
    lines = []
    lines.append(f"  MTA (Main Task Accuracy): {metrics['MTA']:.4f}")
    for k, v in metrics.items():
        if k.startswith("ASR_"):
            lines.append(f"  {k}: {v:.4f}")
    lines.append(f"  PUT (Privacy-Utility Trade-off): {metrics['PUT']:.4f}")
    return "\n".join(lines)
