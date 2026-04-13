"""
VFL feature partitioning across N passive parties.

For image data:  split by channels or spatial regions.
For tabular data: split columns evenly.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import platform


# ======================================================================
# Partitioning strategies
# ======================================================================

def partition_features_tabular(
    X: np.ndarray, num_parties: int
) -> List[np.ndarray]:
    """
    Split tabular features column-wise into *num_parties* roughly equal parts.

    Args:
        X: shape (N, D)
        num_parties: number of passive parties

    Returns:
        List of arrays, each with shape (N, D_i), sum(D_i) == D.
    """
    D = X.shape[1]
    splits = np.array_split(np.arange(D), num_parties)
    return [X[:, idx] for idx in splits]


def partition_features_image(
    X: np.ndarray, num_parties: int
) -> List[np.ndarray]:
    """
    Split image features for VFL.

    Strategy:
      - If num_parties <= num_channels: split by channel groups.
      - Otherwise: split spatially (vertical strips).

    Args:
        X: shape (N, C, H, W)
        num_parties: number of passive parties

    Returns:
        List of arrays. Each is flattened to (N, D_i).
    """
    N, C, H, W = X.shape

    if num_parties <= C:
        # Split channels
        ch_splits = np.array_split(np.arange(C), num_parties)
        parts = []
        for ch_idx in ch_splits:
            part = X[:, ch_idx, :, :]  # (N, C_i, H, W)
            parts.append(part.reshape(N, -1))
        return parts
    else:
        # Split spatial (vertical strips)
        w_splits = np.array_split(np.arange(W), num_parties)
        parts = []
        for w_idx in w_splits:
            part = X[:, :, :, w_idx]  # (N, C, H, W_i)
            parts.append(part.reshape(N, -1))
        return parts


def partition_features(
    X: np.ndarray, num_parties: int, is_image: bool = False
) -> List[np.ndarray]:
    """Dispatch to the right partition function."""
    if is_image:
        return partition_features_image(X, num_parties)
    else:
        return partition_features_tabular(X, num_parties)


# ======================================================================
# VFL Dataset wrapper
# ======================================================================

class VFLDataset(Dataset):
    """
    A dataset that holds feature partitions for each passive party + labels.

    party_features: list of np.ndarray, one per passive party
    labels: np.ndarray
    """

    def __init__(self, party_features: List[np.ndarray], labels: np.ndarray):
        super().__init__()
        self.party_features = [
            torch.tensor(f, dtype=torch.float32) for f in party_features
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_parties = len(party_features)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = [self.party_features[i][idx] for i in range(self.num_parties)]
        label = self.labels[idx]
        return features, label


def vfl_collate_fn(batch):
    """
    Custom collate: batch is list of (List[Tensor], Tensor).
    Returns (List[Tensor], Tensor) where each party's features are batched.
    """
    features_list, labels = zip(*batch)
    num_parties = len(features_list[0])
    batched_features = [
        torch.stack([f[i] for f in features_list]) for i in range(num_parties)
    ]
    batched_labels = torch.stack(labels)
    return batched_features, batched_labels


def build_vfl_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_parties: int,
    is_image: bool,
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, List[int]]:
    """
    Build train/test dataloaders for VFL.

    Returns:
        train_loader, test_loader, feature_dims (list of int per party)
    """
    # Windows 不支持 DataLoader 的 fork 多进程，需要 num_workers=0
    if platform.system() == "Windows" and num_workers > 0:
        print("[提示] Windows 系统检测到，自动设置 num_workers=0 以避免多进程错误")
        num_workers = 0

    # pin_memory 仅在 CUDA 可用时生效
    use_pin_memory = torch.cuda.is_available()

    train_parts = partition_features(X_train, num_parties, is_image)
    test_parts = partition_features(X_test, num_parties, is_image)

    feature_dims = [p.shape[1] for p in train_parts]

    train_ds = VFLDataset(train_parts, y_train)
    test_ds = VFLDataset(test_parts, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vfl_collate_fn,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vfl_collate_fn,
        pin_memory=use_pin_memory,
    )

    return train_loader, test_loader, feature_dims
