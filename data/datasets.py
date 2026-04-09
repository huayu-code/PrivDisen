"""
Dataset loaders for PrivDisen experiments.

Supported: CIFAR-10, CIFAR-100, MNIST, Adult, Bank, Criteo (sampled).
All loaders return (X_train, y_train, X_test, y_test) as numpy arrays
so that VFL partitioning can be applied uniformly.

NOTE: Image dataset URLs are patched to use China-mainland mirrors by default.
      Set environment variable PRIVDISEN_NO_MIRROR=1 to use original URLs.
"""

import os
from typing import Tuple

import numpy as np
import torch
from torchvision import datasets, transforms


# ======================================================================
# China-mainland mirror URLs for torchvision datasets
# ======================================================================
# torchvision 默认从多伦多大学/Yann LeCun 等国外服务器下载，国内很慢。
# 这里 monkey-patch 各数据集类的 url / mirrors 属性，使用国内镜像。

_USE_MIRROR = os.environ.get("PRIVDISEN_NO_MIRROR", "0") != "1"

if _USE_MIRROR:
    # --- CIFAR-10 / CIFAR-100 ---
    # 官方源: https://www.cs.toronto.edu/~kriz/
    # 替换为: 阿里云 / 清华等国内可达的地址
    _CIFAR10_MIRROR = "https://pai-datasets.oss-cn-beijing.aliyuncs.com/cifar/"
    _CIFAR100_MIRROR = "https://pai-datasets.oss-cn-beijing.aliyuncs.com/cifar/"

    # Patch CIFAR-10
    if hasattr(datasets.CIFAR10, "url"):
        datasets.CIFAR10.url = _CIFAR10_MIRROR + "cifar-10-python.tar.gz"
    if hasattr(datasets.CIFAR10, "mirrors"):
        datasets.CIFAR10.mirrors = [_CIFAR10_MIRROR]

    # Patch CIFAR-100
    if hasattr(datasets.CIFAR100, "url"):
        datasets.CIFAR100.url = _CIFAR100_MIRROR + "cifar-100-python.tar.gz"
    if hasattr(datasets.CIFAR100, "mirrors"):
        datasets.CIFAR100.mirrors = [_CIFAR100_MIRROR]

    # --- MNIST ---
    # 官方源: http://yann.lecun.com/exdb/mnist/ (已不可用)
    # torchvision 现在默认走 GitHub release 或 AWS，国内也慢
    # 替换为可用镜像
    _MNIST_MIRRORS = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]
    if hasattr(datasets.MNIST, "mirrors"):
        datasets.MNIST.mirrors = _MNIST_MIRRORS
    # 如果 torchvision 版本用 resources 列表（新版本）
    if hasattr(datasets.MNIST, "resources"):
        pass  # resources 包含 (filename, md5)，URL 由 mirrors 拼接，已通过上面修改


# ======================================================================
# Image datasets
# ======================================================================

def _load_image_dataset(cls, data_dir: str, flatten: bool = False):
    """Generic loader for torchvision image datasets."""
    transform = transforms.ToTensor()

    print(f"[数据集] 正在加载 {cls.__name__}...")
    if _USE_MIRROR:
        mirror_url = getattr(cls, "url", None) or (
            getattr(cls, "mirrors", [""])[0] if hasattr(cls, "mirrors") else ""
        )
        print(f"[数据集] 使用镜像源: {mirror_url[:60]}...")
    else:
        print(f"[数据集] 使用官方源（可设 PRIVDISEN_NO_MIRROR=0 切换镜像）")

    train_ds = cls(root=data_dir, train=True, download=True, transform=transform)
    test_ds = cls(root=data_dir, train=False, download=True, transform=transform)

    X_train = torch.stack([x for x, _ in train_ds]).numpy()
    y_train = np.array([y for _, y in train_ds])
    X_test = torch.stack([x for x, _ in test_ds]).numpy()
    y_test = np.array([y for _, y in test_ds])

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)

    print(f"[数据集] {cls.__name__} 加载完成: "
          f"train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_cifar10(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    return _load_image_dataset(datasets.CIFAR10, data_dir)


def load_cifar100(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    return _load_image_dataset(datasets.CIFAR100, data_dir)


def load_mnist(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    return _load_image_dataset(datasets.MNIST, data_dir)


# ======================================================================
# Tabular datasets
# ======================================================================

def _download_uci(url: str, save_path: str) -> None:
    """Download a file if it does not exist."""
    if os.path.exists(save_path):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import urllib.request
    print(f"Downloading {url} -> {save_path}")
    urllib.request.urlretrieve(url, save_path)


def load_adult(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    """UCI Adult (Census Income) dataset."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    train_path = os.path.join(data_dir, "adult", "adult.data")
    test_path = os.path.join(data_dir, "adult", "adult.test")

    _download_uci(train_url, train_path)
    _download_uci(test_url, test_path)

    import pandas as pd

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income",
    ]

    df_train = pd.read_csv(train_path, header=None, names=columns,
                           na_values=" ?", skipinitialspace=True)
    df_test = pd.read_csv(test_path, header=None, names=columns,
                          na_values=" ?", skipinitialspace=True, skiprows=1)

    df = pd.concat([df_train, df_test], ignore_index=True).dropna()

    # Encode categoricals
    le_dict = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col == "income":
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Label
    df["income"] = df["income"].apply(
        lambda x: 1 if x.strip().rstrip(".") == ">50K" else 0
    )

    y = df["income"].values
    X = df.drop("income", axis=1).values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_train = len(df_train.dropna())
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test


def load_bank(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    """UCI Bank Marketing dataset."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import pandas as pd

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    zip_path = os.path.join(data_dir, "bank", "bank.zip")
    csv_path = os.path.join(data_dir, "bank", "bank-full.csv")

    if not os.path.exists(csv_path):
        _download_uci(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.join(data_dir, "bank"))

    df = pd.read_csv(csv_path, sep=";")

    # Encode categoricals
    for col in df.select_dtypes(include=["object"]).columns:
        if col == "y":
            continue
        df[col] = LabelEncoder().fit_transform(df[col])

    df["y"] = (df["y"] == "yes").astype(int)
    y = df["y"].values
    X = df.drop("y", axis=1).values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 80/20 split
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test


# ======================================================================
# Dispatcher
# ======================================================================

DATASET_REGISTRY = {
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "mnist": load_mnist,
    "adult": load_adult,
    "bank": load_bank,
}


def load_dataset(name: str, data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    """Load dataset by name. Returns (X_train, y_train, X_test, y_test)."""
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](data_dir)


def get_num_classes(name: str) -> int:
    """Return number of classes for a dataset."""
    return {
        "cifar10": 10, "cifar100": 100, "mnist": 10,
        "adult": 2, "bank": 2, "criteo": 2,
    }[name.lower()]


def is_image_dataset(name: str) -> bool:
    return name.lower() in {"cifar10", "cifar100", "mnist"}
