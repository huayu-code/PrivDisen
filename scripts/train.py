"""
PrivDisen Training Pipeline — 跨平台 Python 脚本。

等价于原来的 scripts/train.sh，但在 Windows / Linux / macOS 上均可运行。

用法：
    python scripts/train.py
    python scripts/train.py --device cuda:0 --epochs 50
    python scripts/train.py --datasets cifar10 adult --methods privdisen
"""

import argparse
import subprocess
import sys
import os

# Windows console UTF-8 support
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def run_cmd(args_list):
    """运行子进程，实时打印输出，失败时退出。"""
    print(f"\n>>> {' '.join(args_list)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(args_list, env=env)
    if result.returncode != 0:
        print(f"[Error] return code: {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="PrivDisen Training Pipeline")
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备 (cuda:0 / cpu / auto)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "adult", "bank"],
                        help="要训练的数据集列表")
    parser.add_argument("--methods", nargs="+", default=["vanilla", "privdisen"],
                        help="要训练的方法列表")
    parser.add_argument("--num_parties", type=int, default=2)
    args = parser.parse_args()

    # 自动检测设备
    if args.device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    print("=" * 50)
    print(" PrivDisen Training Pipeline")
    print(f" Device: {device}")
    print(f" Epochs: {args.epochs}")
    print(f" Datasets: {args.datasets}")
    print(f" Methods: {args.methods}")
    print("=" * 50)

    for dataset in args.datasets:
        for method in args.methods:
            run_cmd([
                sys.executable, "experiments/run_main.py",
                "--config", "configs/default.yaml",
                "--method", method,
                "--dataset", dataset,
                "--epochs", str(args.epochs),
                "--device", device,
                "--num_parties", str(args.num_parties),
            ])

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
