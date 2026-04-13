"""
KDk Baseline: Knowledge Distillation + k-Anonymity for VFL label protection.

Reference:
    Arazzi et al., "Knowledge Distillation and k-Anonymity for Privacy-Preserving
    VFL", Neurocomputing, 2025.

Core idea:
    1. Train a teacher VFL model normally (Vanilla VFL).
    2. Train a student model using soft labels from the teacher (knowledge
       distillation), so the gradients sent to passive parties carry less
       direct label information.
    3. k-Anonymity: group samples into k-sized buckets and average gradients
       within each bucket before sending back to passive parties.

We implement steps 2+3 (teacher is assumed pre-trained or trained jointly).
For simplicity, this implementation trains teacher and student end-to-end:
    - Teacher branch: uses hard labels (CE loss)
    - Student branch: uses soft targets from teacher (KD loss)
    - Gradients to passive parties come from the student branch only.
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
from losses.task_loss import task_loss
from utils.logger import get_logger


def _kd_loss(student_logits, teacher_logits, temperature=4.0):
    """Knowledge distillation loss: KL(student_soft || teacher_soft)."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def _k_anonymize_grad(grad: torch.Tensor, k: int) -> torch.Tensor:
    """Average gradients within groups of size k (k-anonymity on gradients)."""
    B = grad.size(0)
    if k <= 1 or B <= k:
        return grad
    # Pad to multiple of k
    n_groups = (B + k - 1) // k
    pad_size = n_groups * k - B
    if pad_size > 0:
        grad = torch.cat([grad, grad[:pad_size]], dim=0)
    # Reshape -> average within groups -> repeat
    grouped = grad.view(n_groups, k, -1).mean(dim=1, keepdim=True)
    anonymized = grouped.expand(n_groups, k, -1).reshape(-1, grad.size(-1))
    return anonymized[:B]


class _GradKAnonymizeHook:
    """Hook that replaces backward gradients with k-anonymized version."""
    def __init__(self, k: int):
        self.k = k

    def __call__(self, grad):
        return _k_anonymize_grad(grad, self.k)


class KDkTrainer:
    """
    KDk: Knowledge Distillation + k-Anonymity defense.
    """

    def __init__(
        self,
        feature_dims: List[int],
        num_classes: int,
        bottom_model_type: str = "mlp",
        embedding_dim: int = 128,
        top_hidden_dim: int = 256,
        temperature: float = 4.0,
        kd_alpha: float = 0.7,
        k_anon: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        log_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.num_parties = len(feature_dims)
        self.num_classes = num_classes
        self.temperature = temperature
        self.kd_alpha = kd_alpha
        self.k_anon = k_anon
        self.logger = get_logger("KDk", log_dir)

        # Shared bottom models
        self.bottom_models = nn.ModuleList([
            build_bottom_model(bottom_model_type, fd, embedding_dim)
            for fd in feature_dims
        ]).to(self.device)

        top_input_dim = embedding_dim * self.num_parties

        # Teacher top model (trained with hard labels)
        self.teacher_top = TopModel(top_input_dim, num_classes, top_hidden_dim).to(self.device)
        # Student top model (trained with soft labels from teacher)
        self.student_top = TopModel(top_input_dim, num_classes, top_hidden_dim).to(self.device)

        all_params = (
            list(self.bottom_models.parameters())
            + list(self.teacher_top.parameters())
            + list(self.student_top.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    def _forward(self, party_features):
        embeddings = []
        for i, bm in enumerate(self.bottom_models):
            x_i = party_features[i].to(self.device)
            embeddings.append(bm(x_i))
        z = torch.cat(embeddings, dim=-1)
        return z

    def train_epoch(self, train_loader, epoch, max_epoch=None):
        self.bottom_models.train()
        self.teacher_top.train()
        self.student_top.train()

        total_loss, correct, total = 0.0, 0, 0

        for party_features, labels in train_loader:
            labels = labels.to(self.device)
            z = self._forward(party_features)

            # Register k-anonymity hook on embedding gradients
            if z.requires_grad and self.k_anon > 1:
                z.register_hook(_GradKAnonymizeHook(self.k_anon))

            # Teacher (hard label)
            with torch.no_grad():
                teacher_logits = self.teacher_top(z.detach())

            # Student (soft label from teacher + partial hard label)
            student_logits = self.student_top(z)
            l_hard = task_loss(student_logits, labels)
            l_kd = _kd_loss(student_logits, teacher_logits.detach(), self.temperature)
            loss_student = (1 - self.kd_alpha) * l_hard + self.kd_alpha * l_kd

            # Also update teacher with hard labels
            teacher_logits_train = self.teacher_top(z.detach())
            loss_teacher = task_loss(teacher_logits_train, labels)

            loss = loss_student + loss_teacher

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = student_logits.argmax(1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {"train_loss": total_loss / total, "train_acc": correct / total}

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.bottom_models.eval()
        self.student_top.eval()
        total_loss, correct, total = 0.0, 0, 0
        for pf, labels in test_loader:
            labels = labels.to(self.device)
            z = self._forward(pf)
            logits = self.student_top(z)
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
                    self.save(os.path.join(checkpoint_dir, "kdk_best.pt"))
        return history

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "bottom_models": self.bottom_models.state_dict(),
            "teacher_top": self.teacher_top.state_dict(),
            "student_top": self.student_top.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.bottom_models.load_state_dict(ckpt["bottom_models"])
        self.teacher_top.load_state_dict(ckpt["teacher_top"])
        self.student_top.load_state_dict(ckpt["student_top"])

    def get_embeddings(self, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.bottom_models.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for pf, labels in loader:
                z = self._forward(pf)
                all_z.append(z.cpu())
                all_y.append(labels)
        return torch.cat(all_z), torch.cat(all_y)
