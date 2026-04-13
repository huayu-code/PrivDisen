"""
数据集下载脚本。

使用 HuggingFace Hub + 国内镜像 (hf-mirror.com) 下载数据集，
支持断点续传、多线程，国内速度快。

用法：
    python data/download.py --dataset cifar10
    python data/download.py --dataset cifar100
    python data/download.py --dataset all
"""

import argparse
import os
import sys
import tarfile

# 设置 HuggingFace 国内镜像（必须在 import huggingface_hub 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================================================================
# HuggingFace Hub 下载
# ======================================================================

DATASET_INFO = {
    "cifar10": {
        "repo_id": "uoft-cs/cifar10",
        "filename": "cifar-10-python.tar.gz",
        "extracted_dir": "cifar-10-batches-py",
    },
    "cifar100": {
        "repo_id": "uoft-cs/cifar100",
        "filename": "cifar-100-python.tar.gz",
        "extracted_dir": "cifar-100-python",
    },
}


def download_with_hf_hub(dataset_name: str, data_dir: str = "data/raw"):
    """使用 huggingface_hub 从国内镜像下载数据集。"""

    if dataset_name not in DATASET_INFO:
        print(f"[跳过] {dataset_name} 不需要手动下载（会自动下载或为表格数据集）")
        return

    info = DATASET_INFO[dataset_name]
    extracted_path = os.path.join(data_dir, info["extracted_dir"])

    # 已经解压过了
    if os.path.isdir(extracted_path):
        print(f"[✅] {dataset_name} 已存在: {extracted_path}")
        return

    os.makedirs(data_dir, exist_ok=True)

    print(f"[下载] {dataset_name}")
    print(f"  镜像源: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"  仓库: {info['repo_id']}")
    print(f"  文件: {info['filename']}")
    print()

    try:
        from huggingface_hub import hf_hub_download

        # hf_hub_download 支持断点续传、多线程、进度条
        downloaded_path = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            repo_type="dataset",
            local_dir=data_dir,
            local_dir_use_symlinks=False,  # Windows 兼容
        )

        print(f"\n[✅] 下载完成: {downloaded_path}")

        # 解压
        tar_path = os.path.join(data_dir, info["filename"])
        if not os.path.exists(tar_path) and os.path.exists(downloaded_path):
            tar_path = downloaded_path

        if os.path.exists(tar_path) and tarfile.is_tarfile(tar_path):
            print(f"[解压] {tar_path} -> {data_dir}/")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(path=data_dir)
            print(f"[✅] 解压完成: {extracted_path}")
        else:
            print(f"[⚠️] 未找到 tar 文件，请检查 {data_dir}/ 目录")

    except ImportError:
        print("[❌] 未安装 huggingface_hub，请先执行:")
        print("     pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)
    except Exception as e:
        print(f"[❌] 下载失败: {e}")
        print()
        print("请尝试以下解决方案:")
        print("  1. 手动在浏览器下载:")
        print(f"     https://hf-mirror.com/datasets/{info['repo_id']}/resolve/main/{info['filename']}")
        print(f"  2. 下载后放到 {data_dir}/ 目录下")
        print(f"  3. 重新运行本脚本")
        sys.exit(1)


def download_tabular(dataset_name: str, data_dir: str = "data/raw"):
    """下载表格数据集（文件小，直接从 UCI 下载）。"""
    from data.datasets import load_dataset
    print(f"[下载] {dataset_name}（表格数据集，文件较小）...")
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, data_dir)
    print(f"[✅] {dataset_name} 加载完成: train={X_train.shape}, test={X_test.shape}")


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
    print(f" 镜像: {os.environ.get('HF_ENDPOINT', '(默认)')}")
    print("=" * 50)
    print()

    if args.dataset == "all":
        datasets_to_download = ["cifar10", "cifar100", "mnist", "adult", "bank"]
    else:
        datasets_to_download = [args.dataset]

    for ds in datasets_to_download:
        if ds in DATASET_INFO:
            download_with_hf_hub(ds, args.data_dir)
        elif ds in ("adult", "bank"):
            download_tabular(ds, args.data_dir)
        elif ds == "mnist":
            print(f"[跳过] MNIST 会在首次训练时自动下载")
        print()

    print("全部完成！")


if __name__ == "__main__":
    main()
