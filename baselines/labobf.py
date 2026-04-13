"""
LabObf Baseline: Label Obfuscation via random soft-label mapping.

Reference:
    He et al., "Label Obfuscation for Vertical Federated Learning", 2024.

Core idea:
    The active party replaces ground-truth hard labels with randomly perturbed
    soft labels before computing the loss. A confusion matrix M maps true
    labels y to noisy labels y_tilde:
        y_tilde = (1-eps) * one_hot(y) + eps * uniform(K)
    This prevents the passive party from inferring labels via gradient analysis,
    while the top model still learns from the noisy supervision.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.bottom_model import build_bottom_model
from models.top_model import TopModel
from utils.logger import get_logger


def _soft_label(labels: torch.Tensor, num_classes: int, eps: float) -> torch.Tensor:
    """
    Convert hard labels to soft labels with label smoothing / obfuscation.
    y_soft = (1 - eps) * one_hot(y) + eps / K
    """
    one_hot = F.one_hot(labels, num_classes).float()
    return (1.0 - eps) * one_hot + eps / num_classes


class LabObfTrainer:
    """
    LabObf: train VFL with obfuscated (soft) labels.

    Architecture is identical to Vanilla VFL; the only difference is
    the loss function uses soft cross-entropy with noisy labels.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        bottom_model_type: str = "mlp",
        embedding_dim: int = 128,
        top_hidden_dim: int = 256,
        eps: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        log_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_parties = len(feature_dims)
        self.num_classes = num_classes
        self.eps = eps
        self.logger = get_logger("LabObf", log_dir)

        self.bottom_models = nn.ModuleList([
            build_bottom_model(bottom_model_type, fd, embedding_dim)
            for fd in feature_dims
        ]).to(self.device)

        top_input_dim = embedding_dim * self.num_parties
        self.top_model = TopModel(
            top_input_dim, num_classes, top_hidden_dim
        ).to(self.device)

        all_params = list(self.bottom_models.parameters()) + list(self.top_model.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    def _forward(self, party_features):
        embeddings = []
        for i, bm in enumerate(self.bottom_models):
            x_i = party_features[i].to(self.device)
            embeddings.append(bm(x_i))
        z_concat = torch.cat(embeddings, dim=-1)
        return embeddings, z_concat

    def _soft_ce_loss(self, logits, labels):
        """Cross-entropy with obfuscated soft labels."""
        soft = _soft_label(labels, self.num_classes, self.eps)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft * log_probs).sum(dim=-1).mean()

    def train_epoch(self, train_loader, epoch, max_epoch=None):
        self.bottom_models.train()
        self.top_model.train()
        total_loss, correct, total = 0.0, 0, 0

        for party_features, labels in train_loader:
            labels = labels.to(self.device)
            _, z = self._forward(party_features)
            logits = self.top_model(z)
            loss = self._soft_ce_loss(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return {"train_loss": total_loss / total, "train_acc": correct / total}

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.bottom_models.eval()
        self.top_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for party_features, labels in test_loader:
            labels = labels.to(self.device)
            _, z = self._forward(party_features)
            logits = self.top_model(z)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return {"test_loss": total_loss / total, "test_acc": correct / total}

    def fit(self, train_loader, test_loader, epochs=100, eval_every=5, checkpoint_dir=None):
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            m = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(m["train_loss"])
            history["train_acc"].append(m["train_acc"])
            if epoch % eval_every == 0 or epoch == epochs:
                t = self.evaluate(test_loader)
                history["test_loss"].append(t["test_loss"])
                history["test_acc"].append(t["test_acc"])
                self.logger.info(
                    f"Epoch {epoch}/{epochs} | Train: {m['train_acc']:.4f} | Test: {t['test_acc']:.4f}")
                if checkpoint_dir and t["test_acc"] > best_acc:
                    best_acc = t["test_acc"]
                    self.save(os.path.join(checkpoint_dir, "labobf_best.pt"))
        return history

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "bottom_models": self.bottom_models.state_dict(),
            "top_model": self.top_model.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.bottom_models.load_state_dict(ckpt["bottom_models"])
        self.top_model.load_state_dict(ckpt["top_model"])

    def get_embeddings(self, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.bottom_models.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for pf, labels in loader:
                _, z = self._forward(pf)
                all_z.append(z.cpu())
                all_y.append(labels)
        return torch.cat(all_z), torch.cat(all_y)
