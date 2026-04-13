"""
Dataset loaders for PrivDisen experiments.

Supported: CIFAR-10, CIFAR-100, MNIST, Adult, Bank.
All loaders return (X_train, y_train, X_test, y_test) as numpy arrays
so that VFL partitioning can be applied uniformly.

数据集下载策略：
  1. 如果本地已有文件，直接加载（不重复下载）
  2. 否则依次尝试多个国内镜像源下载（优先清华/南大/中科大等教育网镜像）
  3. 如果全部失败，打印手动下载指引

手动下载方式（任选其一）：
  - 浏览器打开 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 下载
  - 或从 ModelScope: https://www.modelscope.cn/datasets/cutedataset/cifar-10
  - 或从 OpenDataLab: https://opendatalab.com/CIFAR-10
  下载后将 cifar-10-python.tar.gz 放到 data/raw/ 目录下即可
"""

import os
import sys
import ssl
import tarfile
import hashlib
import time
from typing import List, Optional, Tuple
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
import torch
from torchvision import datasets, transforms


# ======================================================================
# Download utilities with multi-mirror fallback
# ======================================================================

# 连接超时和读取超时（秒），国内网络波动大，需要更宽裕
_CONNECT_TIMEOUT = 60
_MAX_RETRIES = 2


def _md5_check(filepath: str, expected_md5: Optional[str]) -> bool:
    """Check MD5 of a file. Skip if expected_md5 is None."""
    if expected_md5 is None:
        return True
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def _make_ssl_context():
    """创建宽松的 SSL 上下文（部分国内镜像证书有问题）。"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _download_with_progress(url: str, dest: str, timeout: int = _CONNECT_TIMEOUT) -> None:
    """Download a URL to dest with a simple progress indicator and retry."""
    print(f"  下载中: {url}")
    print(f"  保存到: {dest}")

    ctx = _make_ssl_context()

    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = Request(url, headers={
                "User-Agent": "PrivDisen/1.0 (Python)",
                "Accept-Encoding": "identity",  # 避免压缩编码导致长度不匹配
            })
            with urlopen(req, context=ctx, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                start_time = time.time()
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)  # 256KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            elapsed = time.time() - start_time
                            speed = downloaded / max(elapsed, 0.01) / 1024 / 1024
                            bar_len = 40
                            filled = int(bar_len * downloaded / total)
                            bar = "=" * filled + "-" * (bar_len - filled)
                            sys.stdout.write(
                                f"\r  [{bar}] {pct:.1f}% "
                                f"({downloaded/1024/1024:.1f}/{total/1024/1024:.1f}MB) "
                                f"{speed:.1f}MB/s"
                            )
                            sys.stdout.flush()
                print()  # newline after progress bar
                return  # 下载成功，直接返回
        except (URLError, HTTPError, OSError, TimeoutError) as e:
            last_err = e
            if os.path.exists(dest):
                os.remove(dest)
            if attempt < _MAX_RETRIES:
                wait = attempt * 3
                print(f"  ⚠️ 第 {attempt} 次尝试失败: {e}，{wait}s 后重试...")
                time.sleep(wait)

    # 所有重试都失败
    raise last_err  # type: ignore[misc]


def _download_from_mirrors(
    mirrors: List[str],
    dest: str,
    md5: Optional[str] = None,
) -> bool:
    """
    Try downloading from a list of mirror URLs.
    Returns True on success, False if all mirrors fail.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    for i, url in enumerate(mirrors):
        try:
            print(f"[镜像 {i+1}/{len(mirrors)}] 尝试下载...")
            _download_with_progress(url, dest)
            if _md5_check(dest, md5):
                print(f"  ✅ 下载成功，MD5 校验通过")
                return True
            else:
                print(f"  ⚠️ MD5 校验失败，尝试下一个镜像...")
                os.remove(dest)
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            continue

    return False


# ======================================================================
# Mirror URLs for each dataset
# 镜像选择原则：优先国内教育网 / 阿里云 CDN，最后才是海外源
# ======================================================================

CIFAR10_MIRRORS = [
    # ① 官方源直连（torchvision 也用这个，但走了 CDN 后国内部分地区可达）
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    # ② HuggingFace 国内镜像
    "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/cifar-10-python.tar.gz",
]
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"

CIFAR100_MIRRORS = [
    "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    "https://hf-mirror.com/datasets/uoft-cs/cifar100/resolve/main/cifar-100-python.tar.gz",
]
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
CIFAR100_FILENAME = "cifar-100-python.tar.gz"

# MNIST 镜像（torchvision 默认的 yann.lecun.com 在国内几乎不可达）
MNIST_MIRRORS = [
    # Tencent AI Lab 镜像（国内稳定）
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    # 备用：GH archive
    "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/",
]

# UCI 数据集镜像
UCI_MIRRORS = [
    # ① UCI 官方
    "https://archive.ics.uci.edu/ml/machine-learning-databases",
    # ② UCI 镜像（Gitee/国内镜像站）— 如果官方不可达
    "https://archive.ics.uci.edu/static/public",
]


# ======================================================================
# Image datasets
# ======================================================================

def _ensure_cifar_downloaded(
    data_dir: str,
    filename: str,
    extracted_dir: str,
    mirrors: List[str],
    md5: Optional[str],
) -> None:
    """Ensure CIFAR dataset is downloaded and extracted."""
    extracted_path = os.path.join(data_dir, extracted_dir)
    tar_path = os.path.join(data_dir, filename)

    # Already extracted?
    if os.path.isdir(extracted_path):
        print(f"[数据集] 已存在: {extracted_path}")
        return

    # Tar file exists but not extracted?
    if os.path.isfile(tar_path):
        if _md5_check(tar_path, md5):
            print(f"[数据集] 发现已下载的文件: {tar_path}，正在解压...")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(path=data_dir)
            print(f"  ✅ 解压完成")
            return
        else:
            print(f"[数据集] 文件 {tar_path} MD5 不匹配，重新下载...")
            os.remove(tar_path)

    # Need to download
    os.makedirs(data_dir, exist_ok=True)
    print(f"[数据集] 开始下载 {filename}...")
    success = _download_from_mirrors(mirrors, tar_path, md5)

    if success:
        print(f"[数据集] 正在解压 {filename}...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=data_dir)
        print(f"  ✅ 解压完成: {extracted_path}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ 所有镜像均下载失败！请手动下载数据集：")
        print(f"")
        print(f"方法 1: 浏览器直接下载")
        print(f"  https://www.cs.toronto.edu/~kriz/{filename}")
        print(f"")
        print(f"方法 2: 从 ModelScope 下载")
        print(f"  https://www.modelscope.cn/datasets/cutedataset/cifar-10")
        print(f"")
        print(f"方法 3: 从 OpenDataLab 下载")
        print(f"  https://opendatalab.com/CIFAR-10")
        print(f"")
        print(f"下载后将 {filename} 放到 {data_dir}/ 目录下，然后重新运行即可。")
        print(f"{'='*60}")
        raise RuntimeError(f"无法下载 {filename}，请参考上方提示手动下载。")


def _load_image_dataset(cls, data_dir: str, flatten: bool = False):
    """Generic loader for torchvision image datasets (download=False, assume already present)."""
    transform = transforms.ToTensor()

    print(f"[数据集] 正在加载 {cls.__name__}...")

    # Load with download=False (we handle download ourselves)
    train_ds = cls(root=data_dir, train=True, download=False, transform=transform)
    test_ds = cls(root=data_dir, train=False, download=False, transform=transform)

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
    _ensure_cifar_downloaded(
        data_dir, CIFAR10_FILENAME, "cifar-10-batches-py",
        CIFAR10_MIRRORS, CIFAR10_MD5,
    )
    return _load_image_dataset(datasets.CIFAR10, data_dir)


def load_cifar100(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    _ensure_cifar_downloaded(
        data_dir, CIFAR100_FILENAME, "cifar-100-python",
        CIFAR100_MIRRORS, CIFAR100_MD5,
    )
    return _load_image_dataset(datasets.CIFAR100, data_dir)


def _ensure_mnist_downloaded(data_dir: str) -> None:
    """
    确保 MNIST 数据已下载。

    torchvision 默认从 yann.lecun.com 下载，该站点在国内基本不可达。
    我们先尝试用国内可用的镜像下载 4 个 gz 文件到 torchvision 期望的目录结构，
    然后再用 torchvision 加载（download=False 或 download=True 作为 fallback）。
    """
    mnist_dir = os.path.join(data_dir, "MNIST", "raw")
    os.makedirs(mnist_dir, exist_ok=True)

    # torchvision MNIST 需要的 4 个文件
    mnist_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    # 检查是否已经全部存在
    all_exist = all(
        os.path.isfile(os.path.join(mnist_dir, f)) for f in mnist_files
    )
    if all_exist:
        print(f"[数据集] MNIST 原始文件已存在: {mnist_dir}")
        return

    # 国内可用的 MNIST 镜像
    mirror_bases = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ]

    print("[数据集] 开始下载 MNIST...")
    for fname in mnist_files:
        fpath = os.path.join(mnist_dir, fname)
        if os.path.isfile(fpath):
            continue
        mirrors = [base + fname for base in mirror_bases]
        success = _download_from_mirrors(mirrors, fpath, md5=None)
        if not success:
            print(f"  ⚠️ 镜像下载 {fname} 失败，尝试 torchvision 自带下载...")
            # fallback: 让 torchvision 自己试一次
            try:
                datasets.MNIST(root=data_dir, train=True, download=True)
                return
            except Exception as e:
                raise RuntimeError(
                    f"MNIST 下载失败（{fname}）。请手动下载 MNIST 数据集：\n"
                    f"  从 https://ossci-datasets.s3.amazonaws.com/mnist/ 下载所有 .gz 文件\n"
                    f"  放到 {mnist_dir}/ 目录下\n"
                    f"原始错误: {e}"
                )


def load_mnist(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    """MNIST: 先用国内镜像下载，再用 torchvision 加载。"""
    _ensure_mnist_downloaded(data_dir)

    transform = transforms.ToTensor()
    print(f"[数据集] 正在加载 MNIST...")
    # 先尝试 download=False（文件已由我们下载好），失败则 fallback download=True
    try:
        train_ds = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    except Exception:
        train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    X_train = torch.stack([x for x, _ in train_ds]).numpy()
    y_train = np.array([y for _, y in train_ds])
    X_test = torch.stack([x for x, _ in test_ds]).numpy()
    y_test = np.array([y for _, y in test_ds])

    print(f"[数据集] MNIST 加载完成: train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test


# ======================================================================
# Tabular datasets
# ======================================================================

def _download_uci(url: str, save_path: str) -> None:
    """Download a file with retry and timeout. Skip if already exists."""
    if os.path.exists(save_path):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 尝试多个 URL（支持传入单个 str 或 list）
    urls = [url] if isinstance(url, str) else url

    for u in urls:
        try:
            print(f"  下载中: {u} -> {save_path}")
            _download_with_progress(u, save_path)
            return
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            continue

    raise RuntimeError(
        f"UCI 数据下载失败。请手动下载并放到 {save_path}\n"
        f"尝试过的 URL: {urls}"
    )


def load_adult(data_dir: str = "data/raw") -> Tuple[np.ndarray, ...]:
    """UCI Adult (Census Income) dataset."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # 每个文件提供多个候选 URL
    train_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
    ]
    test_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    ]
    train_path = os.path.join(data_dir, "adult", "adult.data")
    test_path = os.path.join(data_dir, "adult", "adult.test")

    _download_uci(train_urls, train_path)
    _download_uci(test_urls, test_path)

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

    bank_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
    ]
    zip_path = os.path.join(data_dir, "bank", "bank.zip")
    csv_path = os.path.join(data_dir, "bank", "bank-full.csv")

    if not os.path.exists(csv_path):
        _download_uci(bank_urls, zip_path)
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
