"""
MID Baseline: Mutual Information regularization Defense.

Reference:
    Zou et al., "Mutual Information Regularization for Vertical Federated
    Learning", 2023.

Core idea:
    Add a mutual information penalty term to the VFL training objective that
    minimizes I(Z; Y) where Z is the embedding sent from passive to active
    party, and Y is the label.

    Since I(Z;Y) is intractable, we use a variational upper bound:
        I(Z;Y) <= E_Y[ KL( p(Z|Y) || p(Z) ) ]
    Approximated by pushing the conditional embedding distribution toward
    a class-agnostic prior via a KL penalty (similar to VIB but applied
    to the original embedding, not a disentangled subspace).
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.bottom_model import build_bottom_model
from models.top_model import TopModel
from losses.task_loss import task_loss
from utils.logger import get_logger


class _StochasticEncoder(nn.Module):
    """
    Wraps a deterministic embedding into a stochastic one:
        h -> (mu, logvar) -> z ~ N(mu, exp(logvar))
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.logvar_layer = nn.Linear(input_dim, output_dim)

    def forward(self, h):
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z, mu, logvar


def _kl_to_prior(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)), averaged over batch."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class MIDTrainer:
    """
    MID: VFL + MI regularization on embeddings.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        bottom_model_type: str = "mlp",
        embedding_dim: int = 128,
        top_hidden_dim: int = 256,
        mi_weight: float = 0.01,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        log_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_parties = len(feature_dims)
        self.num_classes = num_classes
        self.mi_weight = mi_weight
        self.logger = get_logger("MID", log_dir)

        # Bottom models
        self.bottom_models = nn.ModuleList([
            build_bottom_model(bottom_model_type, fd, embedding_dim)
            for fd in feature_dims
        ]).to(self.device)

        # Stochastic encoders (MI bottleneck)
        self.stoch_encoders = nn.ModuleList([
            _StochasticEncoder(embedding_dim, embedding_dim)
            for _ in range(self.num_parties)
        ]).to(self.device)

        # Top model
        top_input_dim = embedding_dim * self.num_parties
        self.top_model = TopModel(
            top_input_dim, num_classes, top_hidden_dim
        ).to(self.device)

        all_params = (
            list(self.bottom_models.parameters())
            + list(self.stoch_encoders.parameters())
            + list(self.top_model.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, train_loader, epoch, max_epoch=None):
        self.bottom_models.train()
        self.stoch_encoders.train()
        self.top_model.train()

        total_loss, correct, total = 0.0, 0, 0

        for party_features, labels in train_loader:
            labels = labels.to(self.device)
            all_z = []
            kl_sum = torch.tensor(0.0, device=self.device)

            for i in range(self.num_parties):
                x_i = party_features[i].to(self.device)
                h_i = self.bottom_models[i](x_i)
                z_i, mu_i, logvar_i = self.stoch_encoders[i](h_i)
                all_z.append(z_i)
                kl_sum = kl_sum + _kl_to_prior(mu_i, logvar_i)

            kl_avg = kl_sum / self.num_parties
            z_cat = torch.cat(all_z, dim=-1)
            logits = self.top_model(z_cat)

            l_task = task_loss(logits, labels)
            loss = l_task + self.mi_weight * kl_avg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {"train_loss": total_loss / total, "train_acc": correct / total}

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.bottom_models.eval()
        self.stoch_encoders.eval()
        self.top_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for pf, labels in test_loader:
            labels = labels.to(self.device)
            all_z = []
            for i in range(self.num_parties):
                x_i = pf[i].to(self.device)
                h_i = self.bottom_models[i](x_i)
                z_i, _, _ = self.stoch_encoders[i](h_i)
                all_z.append(z_i)
            logits = self.top_model(torch.cat(all_z, dim=-1))
            loss = task_loss(logits, labels)
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
                    self.save(os.path.join(checkpoint_dir, "mid_best.pt"))
        return history

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "bottom_models": self.bottom_models.state_dict(),
            "stoch_encoders": self.stoch_encoders.state_dict(),
            "top_model": self.top_model.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.bottom_models.load_state_dict(ckpt["bottom_models"])
        self.stoch_encoders.load_state_dict(ckpt["stoch_encoders"])
        self.top_model.load_state_dict(ckpt["top_model"])

    def get_embeddings(self, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.bottom_models.eval()
        self.stoch_encoders.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for pf, labels in loader:
                zs = []
                for i in range(self.num_parties):
                    x_i = pf[i].to(self.device)
                    h_i = self.bottom_models[i](x_i)
                    z_i, _, _ = self.stoch_encoders[i](h_i)
                    zs.append(z_i)
                all_z.append(torch.cat(zs, dim=-1).cpu())
                all_y.append(labels)
        return torch.cat(all_z), torch.cat(all_y)
