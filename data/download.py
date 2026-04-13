"""
数据集下载脚本。

下载策略（按优先级）：
  1. 检查本地是否已有文件（tar.gz / 解压目录）
  2. ModelScope SDK（阿里云 OSS，国内最快）
  3. 多镜像 fallback（datasets.py 中的国内镜像）
  4. torchvision 官方源
  5. 全部失败则打印手动下载指引

用法：
    python data/download.py --dataset cifar10
    python data/download.py --dataset all
"""

import argparse
import os
import sys
import tarfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _try_import_snapshot_download():
    """兼容不同版本 modelscope 的 snapshot_download import。"""
    try:
        # modelscope >= 1.14 推荐用法
        from modelscope import snapshot_download
        return snapshot_download
    except ImportError:
        pass
    try:
        # modelscope < 1.14 的旧路径
        from modelscope.hub.snapshot_download import snapshot_download
        return snapshot_download
    except ImportError:
        pass
    return None


def download_cifar10_modelscope(data_dir: str) -> bool:
    """用 ModelScope SDK 下载 CIFAR-10（走阿里云 OSS，国内快）"""
    extracted = os.path.join(data_dir, "cifar-10-batches-py")
    if os.path.isdir(extracted):
        print(f"[✅] CIFAR-10 已存在: {extracted}")
        return True

    snapshot_download = _try_import_snapshot_download()
    if snapshot_download is None:
        print("[提示] modelscope 未安装，跳过 ModelScope 下载")
        return False

    try:
        print("[下载] 使用 ModelScope SDK 下载 CIFAR-10（阿里云 OSS）...")
        cache_dir = os.path.join(data_dir, "_modelscope_cache")
        local_path = snapshot_download(
            "cutedataset/cifar-10",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        print(f"[✅] 下载完成: {local_path}")

        # 找到 tar.gz 文件并解压到 data_dir
        tar_found = False
        for root, dirs, files in os.walk(local_path):
            for f in files:
                if f.endswith(".tar.gz") and "cifar-10" in f.lower():
                    tar_path = os.path.join(root, f)
                    print(f"[解压] {tar_path}")
                    with tarfile.open(tar_path, "r:gz") as tf:
                        tf.extractall(path=data_dir)
                    tar_found = True
                    break
            if tar_found:
                break

        if not tar_found:
            print("[提示] 未找到 tar.gz，数据可能已是解压格式")
            for root, dirs, files in os.walk(local_path):
                if "data_batch_1" in files:
                    if not os.path.exists(extracted):
                        shutil.copytree(root, extracted)
                    print(f"[✅] 数据已就绪: {extracted}")
                    return True

        if os.path.isdir(extracted):
            print(f"[✅] 解压完成: {extracted}")
            return True

        return False

    except Exception as e:
        print(f"[⚠️] ModelScope 下载失败: {e}")
        return False


def download_cifar100_modelscope(data_dir: str) -> bool:
    """用 ModelScope SDK 下载 CIFAR-100"""
    extracted = os.path.join(data_dir, "cifar-100-python")
    if os.path.isdir(extracted):
        print(f"[✅] CIFAR-100 已存在: {extracted}")
        return True

    snapshot_download = _try_import_snapshot_download()
    if snapshot_download is None:
        print("[提示] modelscope 未安装")
        return False

    try:
        print("[下载] 使用 ModelScope SDK 下载 CIFAR-100...")
        cache_dir = os.path.join(data_dir, "_modelscope_cache")
        local_path = snapshot_download(
            "maveriq/cifar-100",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        print(f"[✅] 下载完成: {local_path}")

        for root, dirs, files in os.walk(local_path):
            for f in files:
                if f.endswith(".tar.gz") and "cifar-100" in f.lower():
                    tar_path = os.path.join(root, f)
                    print(f"[解压] {tar_path}")
                    with tarfile.open(tar_path, "r:gz") as tf:
                        tf.extractall(path=data_dir)
                    if os.path.isdir(extracted):
                        print(f"[✅] 解压完成: {extracted}")
                        return True

        return False
    except Exception as e:
        print(f"[⚠️] 下载失败: {e}")
        return False


def download_with_mirrors(dataset_name: str, data_dir: str) -> bool:
    """使用 datasets.py 中的多镜像下载逻辑（国内镜像优先）"""
    print(f"[下载] 使用国内镜像下载 {dataset_name}...")
    try:
        from data.datasets import (
            _download_from_mirrors, _ensure_cifar_downloaded, _ensure_mnist_downloaded,
            CIFAR10_MIRRORS, CIFAR10_MD5, CIFAR10_FILENAME,
            CIFAR100_MIRRORS, CIFAR100_MD5, CIFAR100_FILENAME,
        )
        if dataset_name == "cifar10":
            _ensure_cifar_downloaded(
                data_dir, CIFAR10_FILENAME, "cifar-10-batches-py",
                CIFAR10_MIRRORS, CIFAR10_MD5,
            )
            return True
        elif dataset_name == "cifar100":
            _ensure_cifar_downloaded(
                data_dir, CIFAR100_FILENAME, "cifar-100-python",
                CIFAR100_MIRRORS, CIFAR100_MD5,
            )
            return True
        elif dataset_name == "mnist":
            _ensure_mnist_downloaded(data_dir)
            return True
    except Exception as e:
        print(f"[❌] 镜像下载失败: {e}")
    return False


def download_with_torchvision(dataset_name: str, data_dir: str) -> bool:
    """Fallback: 直接用 torchvision 下载"""
    print(f"[下载] 使用 torchvision 下载 {dataset_name}（官方源，可能较慢）...")
    try:
        from torchvision import datasets
        if dataset_name == "cifar10":
            datasets.CIFAR10(root=data_dir, train=True, download=True)
        elif dataset_name == "cifar100":
            datasets.CIFAR100(root=data_dir, train=True, download=True)
        elif dataset_name == "mnist":
            datasets.MNIST(root=data_dir, train=True, download=True)
        print(f"[✅] {dataset_name} 下载完成")
        return True
    except Exception as e:
        print(f"[❌] torchvision 下载失败: {e}")
        return False


def print_manual_guide(dataset_name: str, data_dir: str):
    """打印手动下载指引"""
    print()
    print("=" * 60)
    print(f"❌ {dataset_name} 自动下载失败，请手动下载：")
    print()

    if dataset_name == "cifar10":
        print("方法 1: 用 ModelScope SDK（推荐，国内最快）")
        print("  pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("  然后重新运行本脚本")
        print()
        print("方法 2: 浏览器手动下载")
        print("  打开 https://www.modelscope.cn/datasets/cutedataset/cifar-10/files")
        print("  下载 cifar-10-python.tar.gz")
        print(f"  放到 {data_dir}/ 目录下，重新运行本脚本")
        print()
        print("方法 3: 直接下载 tar.gz")
        print("  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        print(f"  下载后放到 {data_dir}/ 目录下")
    elif dataset_name == "cifar100":
        print("方法 1: 浏览器下载")
        print("  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")
        print(f"  下载后放到 {data_dir}/ 目录下")
    elif dataset_name == "mnist":
        print("MNIST 文件需要 4 个 .gz 文件：")
        print("  从 https://ossci-datasets.s3.amazonaws.com/mnist/ 下载：")
        print("    train-images-idx3-ubyte.gz")
        print("    train-labels-idx1-ubyte.gz")
        print("    t10k-images-idx3-ubyte.gz")
        print("    t10k-labels-idx1-ubyte.gz")
        print(f"  放到 {data_dir}/MNIST/raw/ 目录下")

    print("=" * 60)


def try_extract_local(data_dir: str, filename: str, extracted_dir: str) -> bool:
    """检查本地是否已有 tar.gz 文件，有则直接解压"""
    extracted = os.path.join(data_dir, extracted_dir)
    if os.path.isdir(extracted):
        print(f"[✅] 已存在: {extracted}")
        return True

    tar_path = os.path.join(data_dir, filename)
    if os.path.isfile(tar_path):
        print(f"[发现] 本地文件: {tar_path}，正在解压...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=data_dir)
        if os.path.isdir(extracted):
            print(f"[✅] 解压完成: {extracted}")
            return True

    return False


def download_dataset(name: str, data_dir: str):
    """下载指定数据集"""
    os.makedirs(data_dir, exist_ok=True)

    if name == "cifar10":
        # 1. 检查本地文件
        if try_extract_local(data_dir, "cifar-10-python.tar.gz", "cifar-10-batches-py"):
            return
        # 2. ModelScope（阿里云 OSS，国内最快）
        if download_cifar10_modelscope(data_dir):
            return
        # 3. 国内镜像 fallback（datasets.py 中的镜像列表）
        if download_with_mirrors("cifar10", data_dir):
            return
        # 4. torchvision（官方源）
        if download_with_torchvision("cifar10", data_dir):
            return
        # 5. 手动指引
        print_manual_guide("cifar10", data_dir)

    elif name == "cifar100":
        if try_extract_local(data_dir, "cifar-100-python.tar.gz", "cifar-100-python"):
            return
        if download_cifar100_modelscope(data_dir):
            return
        if download_with_mirrors("cifar100", data_dir):
            return
        if download_with_torchvision("cifar100", data_dir):
            return
        print_manual_guide("cifar100", data_dir)

    elif name == "mnist":
        # MNIST: 先用国内镜像，再 torchvision fallback
        if download_with_mirrors("mnist", data_dir):
            return
        if download_with_torchvision("mnist", data_dir):
            return
        print_manual_guide("mnist", data_dir)

    elif name in ("adult", "bank"):
        print(f"[下载] {name}（表格数据集，文件小，直接下载）...")
        from data.datasets import load_dataset
        X_train, y_train, X_test, y_test = load_dataset(name, data_dir)
        print(f"[✅] {name} 完成: train={X_train.shape}, test={X_test.shape}")

    else:
        print(f"[跳过] 未知数据集: {name}")


def main():
    parser = argparse.ArgumentParser(description="PrivDisen 数据集下载工具")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "mnist", "adult", "bank", "all"],
                        help="要下载的数据集")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="数据存储目录")
    args = parser.parse_args()

    print("=" * 50)
    print(" PrivDisen 数据集下载工具")
    print(" 优先使用 ModelScope（阿里云 OSS，国内快速）")
    print("=" * 50)
    print()

    if args.dataset == "all":
        for ds in ["cifar10", "cifar100", "mnist", "adult", "bank"]:
            download_dataset(ds, args.data_dir)
            print()
    else:
        download_dataset(args.dataset, args.data_dir)

    print("\n完成！")


if __name__ == "__main__":
    main()
