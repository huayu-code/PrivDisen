"""
Model Completion Attack.

Reference: Fu et al., "Label Inference Attacks Against VFL", USENIX Security 2022.

Idea: The attacker trains a separate classifier on top of the embeddings
      (Z_task) using a small set of auxiliary labeled samples.
      If the embeddings carry label information, this classifier succeeds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np


class AttackModel(nn.Module):
    """Simple MLP attack model."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def model_completion_attack(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    aux_ratio: float = 0.1,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
) -> float:
    """
    Model completion attack.

    Uses `aux_ratio` of samples as auxiliary labeled data to train
    an attack model, then evaluates on the rest.

    Args:
        embeddings: (N, D)
        labels:     (N,)
        num_classes: number of classes
        aux_ratio:  fraction of samples used as auxiliary labeled data
        epochs:     attack model training epochs
        lr:         learning rate
        device:     device

    Returns:
        Attack accuracy on non-auxiliary samples.
    """
    device = torch.device(device)
    N = embeddings.shape[0]
    n_aux = max(int(N * aux_ratio), num_classes)  # at least 1 per class

    # Random split
    perm = torch.randperm(N)
    aux_idx = perm[:n_aux]
    eval_idx = perm[n_aux:]

    X_aux = embeddings[aux_idx].to(device)
    y_aux = labels[aux_idx].to(device)
    X_eval = embeddings[eval_idx].to(device)
    y_eval = labels[eval_idx].numpy()

    # Train attack model
    attack_model = AttackModel(embeddings.shape[1], num_classes).to(device)
    optimizer = optim.Adam(attack_model.parameters(), lr=lr)

    attack_model.train()
    for _ in range(epochs):
        logits = attack_model(X_aux)
        loss = nn.functional.cross_entropy(logits, y_aux)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    attack_model.eval()
    with torch.no_grad():
        preds = attack_model(X_eval).argmax(dim=1).cpu().numpy()

    return float(accuracy_score(y_eval, preds))
