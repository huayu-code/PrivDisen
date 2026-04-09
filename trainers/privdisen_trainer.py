"""
PrivDisen Trainer: the core training loop for our method.

Extends VFL training with:
  - Variational Disentanglement Module (VDM)
  - Adversarial Label Classifier (ALC) with Gradient Reversal
  - MI constraint (KL divergence)
  - HSIC independence loss
  - Reconstruction loss
  - Optional gradient purification
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.bottom_model import build_bottom_model
from models.top_model import TopModel
from models.vdm import VariationalDisentangleModule
from models.adversarial import AdversarialLabelClassifier, compute_alpha
from models.reconstruction import ReconstructionDecoder
from models.gradient_purifier import GradientPurifier
from losses.task_loss import task_loss
from losses.mi_loss import mi_loss
from losses.hsic_loss import hsic_loss
from losses.reconstruction_loss import reconstruction_loss
from utils.logger import get_logger


class PrivDisenTrainer:
    """
    PrivDisen trainer with full defense pipeline.

    Architecture per passive party:
        X_i → BottomModel_i → h_i → VDM_i → (Z_task_i, Z_private_i)

    Active party:
        concat(Z_task_1, ..., Z_task_N) → TopModel → ŷ

    Losses:
        L = L_task + α·L_adv + β·L_MI + γ·L_recon + δ·L_indep
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        # Model architecture
        bottom_model_type: str = "mlp",
        embedding_dim: int = 128,
        task_dim: int = 128,
        private_dim: int = 64,
        vdm_hidden_dim: int = 256,
        top_hidden_dim: int = 256,
        alc_hidden_dims: Optional[List[int]] = None,
        # Loss weights
        alpha_schedule: str = "dann",
        alpha_max: float = 1.0,
        beta: float = 0.01,
        gamma: float = 1.0,
        delta: float = 0.1,
        # Training
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        # Optional
        use_gradient_purifier: bool = False,
        device: str = "cuda:0",
        log_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_parties = len(feature_dims)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.task_dim = task_dim
        self.private_dim = private_dim
        self.logger = get_logger("PrivDisen", log_dir)

        # Loss weight config
        self.alpha_schedule = alpha_schedule
        self.alpha_max = alpha_max
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        if alc_hidden_dims is None:
            alc_hidden_dims = [128, 64]

        # ==================== Build Models ====================

        # Bottom models (one per passive party)
        self.bottom_models = nn.ModuleList([
            build_bottom_model(bottom_model_type, fd, embedding_dim)
            for fd in feature_dims
        ]).to(self.device)

        # VDM (one per passive party)
        self.vdms = nn.ModuleList([
            VariationalDisentangleModule(
                embedding_dim, task_dim, private_dim, vdm_hidden_dim
            )
            for _ in range(self.num_parties)
        ]).to(self.device)

        # Reconstruction decoders (one per passive party)
        self.recon_decoders = nn.ModuleList([
            ReconstructionDecoder(task_dim, private_dim, embedding_dim)
            for _ in range(self.num_parties)
        ]).to(self.device)

        # Top model (active party)
        top_input_dim = task_dim * self.num_parties
        self.top_model = TopModel(
            top_input_dim, num_classes, top_hidden_dim
        ).to(self.device)

        # Adversarial Label Classifier (shared across parties)
        self.alc = AdversarialLabelClassifier(
            task_dim * self.num_parties, num_classes, alc_hidden_dims
        ).to(self.device)

        # Optional: Gradient Purifier
        self.use_gradient_purifier = use_gradient_purifier
        if use_gradient_purifier:
            self.grad_purifier = GradientPurifier(
                task_dim * self.num_parties, num_classes
            ).to(self.device)

        # ==================== Optimizers ====================

        # Main optimizer: bottom models + VDMs + recon decoders + top model
        main_params = (
            list(self.bottom_models.parameters())
            + list(self.vdms.parameters())
            + list(self.recon_decoders.parameters())
            + list(self.top_model.parameters())
        )
        self.main_optimizer = optim.Adam(
            main_params, lr=lr, weight_decay=weight_decay
        )

        # ALC optimizer (separate — the GRL handles adversarial training)
        self.alc_optimizer = optim.Adam(
            self.alc.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_epoch(
        self, train_loader: DataLoader, epoch: int, max_epoch: int
    ) -> Dict[str, float]:
        """Train one epoch with full PrivDisen pipeline."""
        self.bottom_models.train()
        self.vdms.train()
        self.recon_decoders.train()
        self.top_model.train()
        self.alc.train()

        # Compute alpha for this epoch
        alpha = compute_alpha(epoch, max_epoch, self.alpha_schedule, self.alpha_max)
        self.alc.set_lambda(alpha)

        metrics = {
            "train_loss": 0.0, "l_task": 0.0, "l_adv": 0.0,
            "l_mi": 0.0, "l_recon": 0.0, "l_indep": 0.0,
            "train_acc": 0.0, "alpha": alpha,
        }
        total = 0

        for party_features, labels in train_loader:
            labels = labels.to(self.device)
            bs = labels.size(0)

            # ========== Forward ==========

            all_z_task = []
            all_z_private = []
            all_h = []
            all_dist_params = []

            for i in range(self.num_parties):
                x_i = party_features[i].to(self.device)
                h_i = self.bottom_models[i](x_i)               # (B, emb_dim)
                z_task_i, z_priv_i, dp_i = self.vdms[i](h_i)   # VDM

                all_h.append(h_i)
                all_z_task.append(z_task_i)
                all_z_private.append(z_priv_i)
                all_dist_params.append(dp_i)

            z_task_concat = torch.cat(all_z_task, dim=-1)   # (B, task_dim * N)

            # Top model prediction
            logits = self.top_model(z_task_concat)

            # ========== Losses ==========

            # L_task: main classification
            l_task = task_loss(logits, labels)

            # L_adv: adversarial label classifier (GRL handles gradient reversal)
            adv_logits = self.alc(z_task_concat)
            l_adv = task_loss(adv_logits, labels)

            # L_MI: mutual information constraint (averaged over parties)
            l_mi = torch.tensor(0.0, device=self.device)
            for dp in all_dist_params:
                l_mi = l_mi + mi_loss(dp["task_mu"], dp["task_logvar"])
            l_mi = l_mi / self.num_parties

            # L_recon: reconstruction loss (averaged over parties)
            l_recon = torch.tensor(0.0, device=self.device)
            for i in range(self.num_parties):
                h_recon = self.recon_decoders[i](all_z_task[i], all_z_private[i])
                l_recon = l_recon + reconstruction_loss(all_h[i], h_recon)
            l_recon = l_recon / self.num_parties

            # L_indep: HSIC independence (averaged over parties)
            l_indep = torch.tensor(0.0, device=self.device)
            for i in range(self.num_parties):
                l_indep = l_indep + hsic_loss(all_z_task[i], all_z_private[i])
            l_indep = l_indep / self.num_parties

            # Total loss
            loss = (
                l_task
                + alpha * l_adv
                + self.beta * l_mi
                + self.gamma * l_recon
                + self.delta * l_indep
            )

            # ========== Backward ==========

            self.main_optimizer.zero_grad()
            self.alc_optimizer.zero_grad()

            loss.backward()

            self.main_optimizer.step()
            self.alc_optimizer.step()

            # Optional: gradient purifier update
            if self.use_gradient_purifier:
                with torch.no_grad():
                    # Get the gradient that would be sent back
                    if z_task_concat.grad is not None:
                        self.grad_purifier.update_centroids(
                            z_task_concat.grad, labels
                        )

            # ========== Metrics ==========
            preds = logits.argmax(dim=1)
            metrics["train_loss"] += loss.item() * bs
            metrics["l_task"] += l_task.item() * bs
            metrics["l_adv"] += l_adv.item() * bs
            metrics["l_mi"] += l_mi.item() * bs
            metrics["l_recon"] += l_recon.item() * bs
            metrics["l_indep"] += l_indep.item() * bs
            metrics["train_acc"] += (preds == labels).sum().item()
            total += bs

        for k in metrics:
            if k != "alpha":
                metrics[k] /= total

        return metrics

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set (main task accuracy)."""
        self.bottom_models.eval()
        self.vdms.eval()
        self.top_model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        for party_features, labels in test_loader:
            labels = labels.to(self.device)

            all_z_task = []
            for i in range(self.num_parties):
                x_i = party_features[i].to(self.device)
                h_i = self.bottom_models[i](x_i)
                z_task_i, _, _ = self.vdms[i](h_i)
                all_z_task.append(z_task_i)

            z_task_concat = torch.cat(all_z_task, dim=-1)
            logits = self.top_model(z_task_concat)
            loss = task_loss(logits, labels)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
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
        history = {
            "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
            "l_task": [], "l_adv": [], "l_mi": [], "l_recon": [], "l_indep": [],
            "alpha": [],
        }
        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch, epochs)
            for k, v in train_metrics.items():
                history[k].append(v)

            if epoch % eval_every == 0 or epoch == epochs:
                test_metrics = self.evaluate(test_loader)
                history["test_loss"].append(test_metrics["test_loss"])
                history["test_acc"].append(test_metrics["test_acc"])

                self.logger.info(
                    f"Epoch {epoch}/{epochs} | α={train_metrics['alpha']:.3f} | "
                    f"Loss: {train_metrics['train_loss']:.4f} "
                    f"(task={train_metrics['l_task']:.3f} "
                    f"adv={train_metrics['l_adv']:.3f} "
                    f"mi={train_metrics['l_mi']:.4f} "
                    f"recon={train_metrics['l_recon']:.4f} "
                    f"indep={train_metrics['l_indep']:.4f}) | "
                    f"Train Acc: {train_metrics['train_acc']:.4f} | "
                    f"Test Acc: {test_metrics['test_acc']:.4f}"
                )

                if checkpoint_dir and test_metrics["test_acc"] > best_acc:
                    best_acc = test_metrics["test_acc"]
                    self.save(os.path.join(checkpoint_dir, "privdisen_best.pt"))

        return history

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "bottom_models": self.bottom_models.state_dict(),
            "vdms": self.vdms.state_dict(),
            "recon_decoders": self.recon_decoders.state_dict(),
            "top_model": self.top_model.state_dict(),
            "alc": self.alc.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.bottom_models.load_state_dict(ckpt["bottom_models"])
        self.vdms.load_state_dict(ckpt["vdms"])
        self.recon_decoders.load_state_dict(ckpt["recon_decoders"])
        self.top_model.load_state_dict(ckpt["top_model"])
        self.alc.load_state_dict(ckpt["alc"])

    def get_embeddings(
        self, loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract Z_task (concatenated), Z_private, and labels.
        For attack evaluation and visualization.
        """
        self.bottom_models.eval()
        self.vdms.eval()
        all_z_task, all_z_priv, all_y = [], [], []

        with torch.no_grad():
            for party_features, labels in loader:
                z_tasks, z_privs = [], []
                for i in range(self.num_parties):
                    x_i = party_features[i].to(self.device)
                    h_i = self.bottom_models[i](x_i)
                    z_t, z_p, _ = self.vdms[i](h_i)
                    z_tasks.append(z_t)
                    z_privs.append(z_p)
                all_z_task.append(torch.cat(z_tasks, dim=-1).cpu())
                all_z_priv.append(torch.cat(z_privs, dim=-1).cpu())
                all_y.append(labels)

        return torch.cat(all_z_task), torch.cat(all_z_priv), torch.cat(all_y)
