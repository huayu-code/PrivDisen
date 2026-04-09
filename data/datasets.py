"""Dataset utilities for PrivDisen.

This module provides ``DatasetWrapper`` – a lightweight adapter that augments
any existing PyTorch dataset with a synthetic *sensitive attribute* label so
that the full PrivDisen training pipeline (task + privacy) can be exercised
with standard vision benchmarks such as MNIST.

For real privacy-preservation experiments, replace the ``sensitive_fn``
argument with one that reads ground-truth sensitive attributes from your data.
"""

from __future__ import annotations

from typing import Callable, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DatasetWrapper(Dataset):
    """Wraps a base dataset and adds a synthetic sensitive attribute.

    Args:
        base_dataset: Any PyTorch Dataset that returns ``(x, y)`` pairs.
        sensitive_fn: Callable that takes the task label ``y`` (int) and
            returns a sensitive attribute label (int).  Defaults to
            ``y % 2`` (even/odd parity), which creates a binary sensitive
            attribute that is correlated with the task label.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        sensitive_fn: Optional[Callable[[int], int]] = None,
    ) -> None:
        self.base = base_dataset
        self.sensitive_fn = sensitive_fn or (lambda y: y % 2)

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x, y = self.base[idx]
        s = self.sensitive_fn(int(y))
        return x, y, s


def get_mnist_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return MNIST train / validation / test ``DataLoader`` objects.

    The dataset is wrapped with ``DatasetWrapper`` so each batch yields
    ``(x, y, s)`` triples (image, digit label, sensitive attribute).

    The MNIST test split is fixed (10 000 samples) and is loaded directly
    from the dataset; ``train_split`` and ``val_split`` control how the
    60 000 training samples are divided between train and validation sets.
    Any remainder after applying both fractions is discarded.

    Args:
        data_dir: Directory where MNIST will be downloaded / cached.
        batch_size: Number of samples per batch.
        train_split: Fraction of the 60 k training samples used for training.
        val_split: Fraction of the 60 k training samples used for validation.
        seed: Random seed for the split.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``.
    """
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])

    full_train = DatasetWrapper(
        torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    )
    test_set = DatasetWrapper(
        torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    )

    n_total = len(full_train)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_rest = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, _ = random_split(
        full_train, [n_train, n_val, n_rest], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
