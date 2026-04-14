"""
Quick smoke test: CIFAR-10, 6 methods, 10 epochs, 2 parties.
Verifies all code paths work before running full experiments.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --device cuda:0
"""

import os
import subprocess
import sys

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

METHODS = ["vanilla", "svfl", "labobf", "kdk", "mid", "privdisen"]


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = detect_device() if args.device == "auto" else args.device

    print("=" * 55)
    print(" Quick Test: CIFAR-10, 6 methods, 2 parties")
    print(f" Device: {device}, Epochs: {args.epochs}")
    print("=" * 55)

    results = {}
    for method in METHODS:
        cmd = [
            sys.executable, "experiments/run_main.py",
            "--config", "configs/default.yaml",
            "--method", method,
            "--dataset", "cifar10",
            "--epochs", str(args.epochs),
            "--device", device,
            "--num_parties", "2",
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        print(f"\n>>> [{method}] running...")
        rc = subprocess.run(cmd, env=env).returncode
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        results[method] = status
        print(f"    [{status}] {method}")

    print("\n" + "=" * 55)
    print(" Quick Test Results:")
    all_ok = True
    for method, status in results.items():
        flag = "pass" if status == "OK" else "FAIL"
        print(f"  [{flag}] {method}")
        if status != "OK":
            all_ok = False
    print("=" * 55)

    if all_ok:
        print("\nAll methods passed! You can now run full experiments:")
        print("  python scripts/run_all_experiments.py --experiment all --epochs 200")
    else:
        print("\nSome methods failed. Check the errors above.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
