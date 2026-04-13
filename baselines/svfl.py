"""
SVFL Baseline: Split VFL with two-classifier feature disentanglement.

Reference:
    Zhang et al., "Secure Feature Disentanglement for Vertical Federated Learning",
    Signal Processing, 2023.

Core idea:
    Each passive party trains TWO bottom models (encoders):
      - A task encoder:    h -> z_task   (sent to active party)
      - A private encoder: h -> z_priv   (kept locally)
    The task encoder is trained with a label classifier on z_priv that uses
    gradient reversal -- forcing z_task to NOT carry label information.
    Unlike PrivDisen, SVFL uses deterministic (hard) splitting, no variational
    inference, no MI loss.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.bottom_model import build_bottom_model
from models.top_model import TopModel
from models.adversarial import AdversarialLabelClassifier, compute_alpha
from losses.task_loss import task_loss
from utils.logger import get_logger


class SVFLTrainer:
    """
    SVFL trainer: deterministic two-encoder disentanglement.

    Each passive party has:
        BottomModel -> shared_h -> TaskEncoder -> z_task  (sent)
                                -> PrivEncoder -> z_priv  (local)
    Active party:
        concat(z_task_1, ..., z_task_N) -> TopModel -> prediction

    Adversarial classifier on z_task ensures label privacy.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        bottom_model_type: str = "mlp",
        embedding_dim: int = 128,
        task_dim: int = 128,
        private_dim: int = 64,
        top_hidden_dim: int = 256,
        alpha_schedule: str = "dann",
        alpha_max: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        log_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_parties = len(feature_dims)
        self.num_classes = num_classes
        self.alpha_schedule = alpha_schedule
        self.alpha_max = alpha_max
        self.logger = get_logger("SVFL", log_dir)

        # Bottom models (shared feature extractor per party)
        self.bottom_models = nn.ModuleList([
            build_bottom_model(bottom_model_type, fd, embedding_dim)
            for fd in feature_dims
        ]).to(self.device)

        # Task encoders (deterministic linear projection)
        self.task_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, task_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(self.num_parties)
        ]).to(self.device)

        # Private encoders (kept local)
        self.priv_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, private_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(self.num_parties)
        ]).to(self.device)

        # Top model
        top_input_dim = task_dim * self.num_parties
        self.top_model = TopModel(
            top_input_dim, num_classes, top_hidden_dim
        ).to(self.device)

        # Adversarial label classifier on z_task (with GRL)
        self.alc = AdversarialLabelClassifier(
            top_input_dim, num_classes
        ).to(self.device)

        # Optimizers
        main_params = (
            list(self.bottom_models.parameters())
            + list(self.task_encoders.parameters())
            + list(self.priv_encoders.parameters())
            + list(self.top_model.parameters())
        )
        self.main_optimizer = optim.Adam(main_params, lr=lr, weight_decay=weight_decay)
        self.alc_optimizer = optim.Adam(self.alc.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, train_loader: DataLoader, epoch: int, max_epoch: int) -> Dict[str, float]:
        self.bottom_models.train()
        self.task_encoders.train()
        self.priv_encoders.train()
        self.top_model.train()
        self.alc.train()

        alpha = compute_alpha(epoch, max_epoch, self.alpha_schedule, self.alpha_max)
        self.alc.set_lambda(alpha)

        total_loss = 0.0
        correct = 0
        total = 0

        for party_features, labels in train_loader:
            labels = labels.to(self.device)
            bs = labels.size(0)

            all_z_task = []
            for i in range(self.num_parties):
                x_i = party_features[i].to(self.device)
                h_i = self.bottom_models[i](x_i)
                z_task_i = self.task_encoders[i](h_i)
                all_z_task.append(z_task_i)

            z_task_cat = torch.cat(all_z_task, dim=-1)
            logits = self.top_model(z_task_cat)

            l_task = task_loss(logits, labels)
            adv_logits = self.alc(z_task_cat)
            l_adv = task_loss(adv_logits, labels)

            loss = l_task + alpha * l_adv

            self.main_optimizer.zero_grad()
            self.alc_optimizer.zero_grad()
            loss.backward()
            self.main_optimizer.step()
            self.alc_optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * bs
            correct += (preds == labels).sum().item()
            total += bs

        return {"train_loss": total_loss / total, "train_acc": correct / total}

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        self.bottom_models.eval()
        self.task_encoders.eval()
        self.top_model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for party_features, labels in test_loader:
            labels = labels.to(self.device)
            all_z = []
            for i in range(self.num_parties):
                x_i = party_features[i].to(self.device)
                h_i = self.bottom_models[i](x_i)
                all_z.append(self.task_encoders[i](h_i))
            z_cat = torch.cat(all_z, dim=-1)
            logits = self.top_model(z_cat)
            loss = task_loss(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        return {"test_loss": total_loss / total, "test_acc": correct / total}

    def fit(self, train_loader, test_loader, epochs=100, eval_every=5, checkpoint_dir=None):
        history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            m = self.train_epoch(train_loader, epoch, epochs)
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
                    self.save(os.path.join(checkpoint_dir, "svfl_best.pt"))
        return history

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "bottom_models": self.bottom_models.state_dict(),
            "task_encoders": self.task_encoders.state_dict(),
            "top_model": self.top_model.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.bottom_models.load_state_dict(ckpt["bottom_models"])
        self.task_encoders.load_state_dict(ckpt["task_encoders"])
        self.top_model.load_state_dict(ckpt["top_model"])

    def get_embeddings(self, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.bottom_models.eval()
        self.task_encoders.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for party_features, labels in loader:
                zs = []
                for i in range(self.num_parties):
                    x_i = party_features[i].to(self.device)
                    h_i = self.bottom_models[i](x_i)
                    zs.append(self.task_encoders[i](h_i))
                all_z.append(torch.cat(zs, dim=-1).cpu())
                all_y.append(labels)
        return torch.cat(all_z), torch.cat(all_y)
