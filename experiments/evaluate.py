"""Evaluation script for PrivDisen.

Usage::

    python experiments/evaluate.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/best.pt

Prints test-set task accuracy and sensitive-attribute leakage accuracy.
A low sensitive-attribute accuracy (close to random chance, e.g. 0.5 for
binary) indicates that the shared representation z_s successfully hides the
sensitive attribute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data.datasets import get_mnist_loaders
from src.models import PrivDisen
from src.utils import accuracy, AverageMeter, load_checkpoint, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PrivDisen")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pt checkpoint file."
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_split(model: PrivDisen, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    task_acc = AverageMeter("task_acc")
    sens_acc = AverageMeter("sens_acc")

    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device)
        outputs = model(x)
        task_acc.update(accuracy(outputs["y_pred"], y), n=x.size(0))
        sens_acc.update(accuracy(outputs["s_pred"], s), n=x.size(0))

    return {"task_acc": task_acc.avg, "sens_acc": sens_acc.avg}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"]["device"])

    _, _, test_loader = get_mnist_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
    )

    model_cfg = cfg["model"]
    model = PrivDisen(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        shared_dim=model_cfg["shared_dim"],
        private_dim=model_cfg["private_dim"],
        num_classes=model_cfg["num_classes"],
        num_sensitive=model_cfg["num_sensitive"],
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    metrics = evaluate_split(model, test_loader, device)
    print("\n=== Test-set Results ===")
    print(f"  Task accuracy      : {metrics['task_acc']:.4f}")
    print(f"  Sensitive leakage  : {metrics['sens_acc']:.4f}  (lower ≈ better privacy)")
    chance = 1.0 / model_cfg["num_sensitive"]
    print(f"  Random-chance baseline: {chance:.4f}")


if __name__ == "__main__":
    main()
