"""
PrivDisen Evaluation Pipeline — 跨平台 Python 脚本。

等价于原来的 scripts/eval.sh，但在 Windows / Linux / macOS 上均可运行。

用法：
    python scripts/eval.py
    python scripts/eval.py --device cuda:0 --epochs 50
"""

import argparse
import subprocess
import sys


def run_cmd(args_list):
    """运行子进程，实时打印输出，失败时退出。"""
    print(f"\n>>> {' '.join(args_list)}")
    result = subprocess.run(args_list, text=True)
    if result.returncode != 0:
        print(f"[错误] 命令返回码: {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="PrivDisen Evaluation Pipeline")
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备 (cuda:0 / cpu / auto)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="cifar10")
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
    print(" PrivDisen Evaluation Pipeline")
    print(f" Device: {device}")
    print("=" * 50)

    # 多方实验
    print("\n>>> Multi-party experiment")
    run_cmd([
        sys.executable, "experiments/run_multi_party.py",
        "--config", "configs/default.yaml",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--device", device,
    ])

    # 消融实验
    print("\n>>> Ablation study")
    run_cmd([
        sys.executable, "experiments/run_ablation.py",
        "--config", "configs/default.yaml",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--device", device,
    ])

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
