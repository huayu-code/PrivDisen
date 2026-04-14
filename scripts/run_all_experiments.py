"""
PrivDisen Automated Experiment Runner.

Runs all experiments required for the paper, automatically records metrics
to CSV/JSON, and saves checkpoints + figures.

Usage:
    python scripts/run_all_experiments.py --experiment all
    python scripts/run_all_experiments.py --experiment main --epochs 100
    python scripts/run_all_experiments.py --experiment pareto --device cuda:0

Experiments:
    main         - Table 1: 6 methods x 3 datasets x 3 attacks
    multi_party  - Table 2: PrivDisen with 2/3/4/5 parties
    ablation     - Table 3: Remove each loss component
    pareto       - Figure:  Sweep beta for MTA vs ASR curve
    all          - Run all of the above sequentially
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

# Windows UTF-8
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
METHODS = ["vanilla", "svfl", "labobf", "kdk", "mid", "privdisen"]
DATASETS = ["cifar10", "mnist", "adult", "bank"]
PARTY_COUNTS = [2, 3, 4, 5]
BETA_SWEEP = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
RECORD_DIR = "results/experiment_records"


def _ensure_dirs():
    for d in [RECORD_DIR, "results/logs", "results/checkpoints", "results/figures"]:
        os.makedirs(d, exist_ok=True)


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _detect_device(user_device):
    if user_device != "auto":
        return user_device
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def run_python(args_list, desc=""):
    """Run a python subprocess and return (returncode, elapsed_seconds)."""
    cmd = [sys.executable] + args_list
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"\n{'='*60}")
    print(f"[RUN] {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"[{status}] {desc} ({elapsed:.1f}s)")
    return result.returncode, elapsed


def append_csv(filepath, row_dict):
    """Append a dict as a row to a CSV file (create header if new)."""
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# --------------------------------------------------------------------------
# Experiment runners
# --------------------------------------------------------------------------

def run_main_experiment(device, epochs):
    """Exp1: Main comparison - 6 methods x 3 datasets."""
    print("\n" + "#"*60)
    print("# Experiment 1: Main Comparison (Table 1)")
    print("#"*60)

    csv_path = os.path.join(RECORD_DIR, f"exp1_main_{_timestamp()}.csv")
    total, success, failed = 0, 0, 0

    for dataset in DATASETS:
        for method in METHODS:
            total += 1
            desc = f"main: {method} on {dataset}"
            rc, elapsed = run_python([
                "experiments/run_main.py",
                "--config", "configs/default.yaml",
                "--method", method,
                "--dataset", dataset,
                "--epochs", str(epochs),
                "--device", device,
                "--num_parties", "2",
            ], desc=desc)

            row = {
                "timestamp": _timestamp(),
                "experiment": "main",
                "method": method,
                "dataset": dataset,
                "num_parties": 2,
                "epochs": epochs,
                "device": device,
                "status": "ok" if rc == 0 else "fail",
                "elapsed_sec": f"{elapsed:.1f}",
            }
            append_csv(csv_path, row)

            if rc == 0:
                success += 1
            else:
                failed += 1

    print(f"\n[Exp1 Summary] Total={total}, Success={success}, Failed={failed}")
    print(f"  Records saved to: {csv_path}")
    return csv_path


def run_multi_party_experiment(device, epochs):
    """Exp2: Multi-party scaling."""
    print("\n" + "#"*60)
    print("# Experiment 2: Multi-Party Scaling (Table 2)")
    print("#"*60)

    csv_path = os.path.join(RECORD_DIR, f"exp2_multi_party_{_timestamp()}.csv")

    rc, elapsed = run_python([
        "experiments/run_multi_party.py",
        "--config", "configs/default.yaml",
        "--dataset", "cifar10",
        "--epochs", str(epochs),
        "--device", device,
    ], desc="multi_party: cifar10, parties=2,3,4,5")

    row = {
        "timestamp": _timestamp(),
        "experiment": "multi_party",
        "dataset": "cifar10",
        "party_counts": "2,3,4,5",
        "epochs": epochs,
        "status": "ok" if rc == 0 else "fail",
        "elapsed_sec": f"{elapsed:.1f}",
    }
    append_csv(csv_path, row)
    print(f"  Records saved to: {csv_path}")
    return csv_path


def run_ablation_experiment(device, epochs):
    """Exp3: Ablation study."""
    print("\n" + "#"*60)
    print("# Experiment 3: Ablation Study (Table 3)")
    print("#"*60)

    csv_path = os.path.join(RECORD_DIR, f"exp3_ablation_{_timestamp()}.csv")

    rc, elapsed = run_python([
        "experiments/run_ablation.py",
        "--config", "configs/default.yaml",
        "--dataset", "cifar10",
        "--epochs", str(epochs),
        "--device", device,
    ], desc="ablation: cifar10, 6 variants")

    row = {
        "timestamp": _timestamp(),
        "experiment": "ablation",
        "dataset": "cifar10",
        "epochs": epochs,
        "status": "ok" if rc == 0 else "fail",
        "elapsed_sec": f"{elapsed:.1f}",
    }
    append_csv(csv_path, row)
    print(f"  Records saved to: {csv_path}")
    return csv_path


def run_pareto_experiment(device, epochs):
    """Exp4: Pareto curve - sweep beta."""
    print("\n" + "#"*60)
    print("# Experiment 4: Pareto Curve (beta sweep)")
    print("#"*60)

    csv_path = os.path.join(RECORD_DIR, f"exp4_pareto_{_timestamp()}.csv")

    for beta in BETA_SWEEP:
        desc = f"pareto: privdisen, beta={beta}"
        rc, elapsed = run_python([
            "experiments/run_main.py",
            "--config", "configs/default.yaml",
            "--method", "privdisen",
            "--dataset", "cifar10",
            "--beta", str(beta),
            "--epochs", str(epochs),
            "--device", device,
            "--num_parties", "2",
        ], desc=desc)

        row = {
            "timestamp": _timestamp(),
            "experiment": "pareto",
            "method": "privdisen",
            "dataset": "cifar10",
            "beta": beta,
            "epochs": epochs,
            "status": "ok" if rc == 0 else "fail",
            "elapsed_sec": f"{elapsed:.1f}",
        }
        append_csv(csv_path, row)

    print(f"  Records saved to: {csv_path}")
    return csv_path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PrivDisen Automated Experiment Runner")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["main", "multi_party", "ablation", "pareto", "all"],
                        help="Which experiment to run")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = _detect_device(args.device)
    _ensure_dirs()

    print("=" * 60)
    print(" PrivDisen Automated Experiment Runner")
    print(f" Experiment: {args.experiment}")
    print(f" Device:     {device}")
    print(f" Epochs:     {args.epochs}")
    print(f" Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t_total = time.time()
    records = []

    if args.experiment in ("main", "all"):
        records.append(run_main_experiment(device, args.epochs))

    if args.experiment in ("multi_party", "all"):
        records.append(run_multi_party_experiment(device, args.epochs))

    if args.experiment in ("ablation", "all"):
        records.append(run_ablation_experiment(device, args.epochs))

    if args.experiment in ("pareto", "all"):
        records.append(run_pareto_experiment(device, args.epochs))

    total_elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print(f" ALL DONE! Total time: {total_elapsed/60:.1f} minutes")
    print(f" Records saved to: {RECORD_DIR}/")
    for r in records:
        print(f"   - {r}")
    print("=" * 60)

    # Save a summary JSON
    summary = {
        "experiment": args.experiment,
        "device": device,
        "epochs": args.epochs,
        "total_time_sec": round(total_elapsed, 1),
        "record_files": records,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = os.path.join(RECORD_DIR, f"summary_{_timestamp()}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
