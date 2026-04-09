"""
Visualization utilities: t-SNE, Pareto curves, training curves.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE",
    save_path: Optional[str] = None,
    max_samples: int = 3000,
):
    """t-SNE visualization of embeddings colored by label."""
    if len(embeddings) > max_samples:
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels,
                          cmap="tab10", s=5, alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[str] = None,
):
    """Plot training loss, accuracy, and component losses."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history.get("train_loss", []), label="Train Loss")
    if "l_task" in history:
        axes[0].plot(history["l_task"], label="L_task", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.get("train_acc", []), label="Train Acc")
    if history.get("test_acc"):
        # test_acc may be recorded less frequently
        test_epochs = np.linspace(0, len(history["train_acc"]) - 1,
                                  len(history["test_acc"]))
        axes[1].plot(test_epochs, history["test_acc"], label="Test Acc", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    # Component losses (PrivDisen specific)
    component_keys = ["l_mi", "l_recon", "l_indep", "l_adv"]
    for k in component_keys:
        if k in history:
            axes[2].plot(history[k], label=k)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Component Losses")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
):
    """
    Plot Privacy-Utility Pareto frontier.

    Args:
        results: dict of method_name -> {"MTA": float, "ASR_avg": float}
    """
    plt.figure(figsize=(8, 6))

    for method, metrics in results.items():
        mta = metrics["MTA"]
        asr = metrics["ASR_avg"]
        plt.scatter(asr, mta, s=100, zorder=5)
        plt.annotate(method, (asr, mta), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    plt.xlabel("Attack Success Rate (ASR) ↓", fontsize=12)
    plt.ylabel("Main Task Accuracy (MTA) ↑", fontsize=12)
    plt.title("Privacy-Utility Trade-off", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
