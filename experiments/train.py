"""Training script for PrivDisen.

Usage::

    python experiments/train.py --config configs/default.yaml

The script:
1. Loads hyperparameters from the YAML config.
2. Builds the PrivDisen model.
3. Trains with an alternating optimisation:
   - **Encoder step**: update shared encoder, private encoder, task head,
     and decoder to minimise the combined objective.
   - **Discriminator step**: update the privacy discriminator to correctly
     predict the sensitive attribute from z_s.
4. Logs per-epoch metrics to ``tensorboard`` (optional) and console.
5. Saves the best checkpoint (by validation task accuracy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
from tqdm import tqdm

from data.datasets import get_mnist_loaders
from src.models import PrivDisen
from src.losses import adversarial_privacy_loss, compute_total_loss
from src.utils import AverageMeter, accuracy, load_config, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PrivDisen")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file."
    )
    return parser.parse_args()


def train_one_epoch(
    model: PrivDisen,
    loader,
    enc_optimizer: optim.Optimizer,
    disc_optimizer: optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> dict[str, float]:
    """Run one training epoch.

    Returns a dict of average loss components.
    """
    model.train()
    meters = {k: AverageMeter(k) for k in ["total", "recon", "task", "kl", "mi", "adv", "disc"]}

    lw = cfg["training"]

    for x, y, s in tqdm(loader, leave=False, desc="train"):
        x = x.to(device)
        y = y.to(device)
        s = s.to(device)

        # ---- Encoder step ------------------------------------------------
        enc_optimizer.zero_grad()
        outputs = model(x)
        losses = compute_total_loss(
            outputs, x, y, s,
            lambda_recon=lw["lambda_recon"],
            alpha=lw["alpha"],
            beta=lw["beta"],
            gamma=lw["gamma"],
            delta=lw["delta"],
        )
        losses["total"].backward()
        enc_optimizer.step()

        for k in ["total", "recon", "task", "kl", "mi", "adv"]:
            meters[k].update(losses[k].item(), n=x.size(0))

        # ---- Discriminator step ------------------------------------------
        disc_optimizer.zero_grad()
        # Re-run forward to get fresh z_s without encoder gradients
        with torch.no_grad():
            mu_s, _ = model.shared_encoder(x)
            z_s_det = mu_s  # use mean for stable discriminator training

        s_pred = model.discriminator(z_s_det)
        disc_loss = adversarial_privacy_loss(s_pred, s)
        disc_loss.backward()
        disc_optimizer.step()

        meters["disc"].update(disc_loss.item(), n=x.size(0))

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(
    model: PrivDisen,
    loader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate task accuracy and sensitive attribute leakage on a split."""
    model.eval()
    task_acc_meter = AverageMeter("task_acc")
    sens_acc_meter = AverageMeter("sens_acc")

    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device)
        outputs = model(x)
        task_acc_meter.update(accuracy(outputs["y_pred"], y), n=x.size(0))
        sens_acc_meter.update(accuracy(outputs["s_pred"], s), n=x.size(0))

    return {
        "task_acc": task_acc_meter.avg,
        "sens_acc": sens_acc_meter.avg,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["training"]["seed"])
    device = torch.device(cfg["training"]["device"])

    # Data
    train_loader, val_loader, _ = get_mnist_loaders(
        data_dir=cfg["data"]["data_dir"],
        batch_size=cfg["training"]["batch_size"],
        seed=cfg["training"]["seed"],
    )

    # Model
    model_cfg = cfg["model"]
    model = PrivDisen(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        shared_dim=model_cfg["shared_dim"],
        private_dim=model_cfg["private_dim"],
        num_classes=model_cfg["num_classes"],
        num_sensitive=model_cfg["num_sensitive"],
    ).to(device)

    # Optimizers – separate for encoder and discriminator
    enc_optimizer = optim.Adam(
        model.encoder_parameters,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    disc_optimizer = optim.Adam(
        model.discriminator_parameters,
        lr=cfg["training"]["lr"],
    )

    best_val_acc = 0.0
    checkpoint_dir = cfg["logging"]["checkpoint_dir"]
    epochs = cfg["training"]["epochs"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["logging"]["save_every"]

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, enc_optimizer, disc_optimizer, device, cfg
        )
        msg = (
            f"Epoch {epoch:03d}/{epochs} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        )

        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate(model, val_loader, device)
            msg += " | " + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())

            if val_metrics["task_acc"] > best_val_acc:
                best_val_acc = val_metrics["task_acc"]
                save_checkpoint(
                    {"epoch": epoch, "state_dict": model.state_dict(), **val_metrics},
                    checkpoint_dir,
                    filename="best.pt",
                )

        print(msg)

        if epoch % save_every == 0:
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict()},
                checkpoint_dir,
                filename=f"epoch_{epoch:03d}.pt",
            )

    print(f"\nTraining complete. Best validation task accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
