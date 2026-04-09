"""
Base VFL Trainer: vanilla VFL training without any defense.
Serves as a baseline and a building block for defense trainers.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bottom_model import build_bottom_model
from models.top_model import TopModel
from losses.task_loss import task_loss
from utils.logger import get_logger


class VFLTrainer:
    """
    Vanilla VFL trainer: N passive parties + 1 active party.

    Each passive party owns a BottomModel.
    The active party owns a TopModel.
    Forward: each passive party sends embeddings → TopModel predicts.
    Backward: TopModel sends gradients back to each passive party.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        bottom_model_type: str = "mlp",
        embedding_dim: int = 128,
        top_hidden_dim: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda:0",
        log_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_parties = len(feature_dims)
        self.embedding_dim = embedding_dim
        self.logger = get_logger("VFLTrainer", log_dir)

        # --- Build models ---
        self.bottom_models = nn.ModuleList([
            build_bottom_model(bottom_model_type, fd, embedding_dim)
            for fd in feature_dims
        ]).to(self.device)

        # Top model receives concatenated embeddings
        top_input_dim = embedding_dim * self.num_parties
        self.top_model = TopModel(
            top_input_dim, num_classes, top_hidden_dim
        ).to(self.device)

        # --- Optimizers ---
        self.bottom_optimizers = [
            optim.Adam(bm.parameters(), lr=lr, weight_decay=weight_decay)
            for bm in self.bottom_models
        ]
        self.top_optimizer = optim.Adam(
            self.top_model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _forward_bottom(
        self, party_features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through all bottom models.

        Returns:
            embeddings: list of (B, embedding_dim) per party
            z_concat:   (B, embedding_dim * num_parties) concatenated
        """
        embeddings = []
        for i, bm in enumerate(self.bottom_models):
            x_i = party_features[i].to(self.device)
            h_i = bm(x_i)
            embeddings.append(h_i)
        z_concat = torch.cat(embeddings, dim=-1)
        return embeddings, z_concat

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch. Returns dict of metrics."""
        self.bottom_models.train()
        self.top_model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for party_features, labels in train_loader:
            labels = labels.to(self.device)

            # Forward
            embeddings, z_concat = self._forward_bottom(party_features)
            logits = self.top_model(z_concat)
            loss = task_loss(logits, labels)

            # Backward
            self.top_optimizer.zero_grad()
            for opt in self.bottom_optimizers:
                opt.zero_grad()

            loss.backward()

            self.top_optimizer.step()
            for opt in self.bottom_optimizers:
                opt.step()

            # Metrics
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "train_loss": total_loss / total,
            "train_acc": correct / total,
        }

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set."""
        self.bottom_models.eval()
        self.top_model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for party_features, labels in test_loader:
            labels = labels.to(self.device)
            _, z_concat = self._forward_bottom(party_features)
            logits = self.top_model(z_concat)
            loss = task_loss(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "test_loss": total_loss / total,
            "test_acc": correct / total,
        }

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 100,
        eval_every: int = 5,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, list]:
        """Full training loop."""
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_acc"].append(train_metrics["train_acc"])

            if epoch % eval_every == 0 or epoch == epochs:
                test_metrics = self.evaluate(test_loader)
                history["test_loss"].append(test_metrics["test_loss"])
                history["test_acc"].append(test_metrics["test_acc"])

                self.logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} "
                    f"Train Acc: {train_metrics['train_acc']:.4f} | "
                    f"Test Loss: {test_metrics['test_loss']:.4f} "
                    f"Test Acc: {test_metrics['test_acc']:.4f}"
                )

                # Save best
                if checkpoint_dir and test_metrics["test_acc"] > best_acc:
                    best_acc = test_metrics["test_acc"]
                    self.save(os.path.join(checkpoint_dir, "vanilla_best.pt"))

        return history

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "bottom_models": self.bottom_models.state_dict(),
            "top_model": self.top_model.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.bottom_models.load_state_dict(ckpt["bottom_models"])
        self.top_model.load_state_dict(ckpt["top_model"])

    def get_embeddings(
        self, loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract all embeddings and labels (for attack evaluation)."""
        self.bottom_models.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for party_features, labels in loader:
                _, z_concat = self._forward_bottom(party_features)
                all_z.append(z_concat.cpu())
                all_y.append(labels)
        return torch.cat(all_z), torch.cat(all_y)
