"""
Auto-download datasets.
"""

import argparse

from data.datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset to download")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    args = parser.parse_args()

    print(f"Downloading {args.dataset}...")
    X_train, y_train, X_test, y_test = load_dataset(args.dataset, args.data_dir)
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
