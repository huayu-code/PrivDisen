"""
Embedding Extension Attack.

Reference: He et al., "LabObf", 2024.

Idea: The attacker manipulates their bottom model's embeddings to
      amplify label-correlated signals, then trains an attack classifier.
      This is a stronger version of model completion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np


class ExtendedAttackModel(nn.Module):
    """Attack model with feature augmentation."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # Augment the embedding
        self.augmentor = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x_aug = self.augmentor(x) + x  # residual
        return self.classifier(x_aug)


def embedding_extension_attack(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    aux_ratio: float = 0.1,
    epochs: int = 80,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """
    Embedding extension attack (stronger variant of model completion).

    The attacker uses an augmentor network to amplify label signals
    before classification.

    Args:
        embeddings: (N, D)
        labels:     (N,)
        num_classes: number of classes
        aux_ratio:  fraction of auxiliary labeled data
        epochs:     attack training epochs
        lr:         learning rate
        device:     device

    Returns:
        Attack accuracy on non-auxiliary samples.
    """
    device = torch.device(device)
    N = embeddings.shape[0]
    n_aux = max(int(N * aux_ratio), num_classes)

    perm = torch.randperm(N)
    aux_idx = perm[:n_aux]
    eval_idx = perm[n_aux:]

    X_aux = embeddings[aux_idx].to(device)
    y_aux = labels[aux_idx].to(device)
    X_eval = embeddings[eval_idx].to(device)
    y_eval = labels[eval_idx].numpy()

    attack_model = ExtendedAttackModel(embeddings.shape[1], num_classes).to(device)
    optimizer = optim.Adam(attack_model.parameters(), lr=lr)

    attack_model.train()
    for _ in range(epochs):
        logits = attack_model(X_aux)
        loss = nn.functional.cross_entropy(logits, y_aux)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    attack_model.eval()
    with torch.no_grad():
        preds = attack_model(X_eval).argmax(dim=1).cpu().numpy()

    return float(accuracy_score(y_eval, preds))
