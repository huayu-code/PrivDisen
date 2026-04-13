"""
Multi-party scaling experiment (Table 2).
Tests how PrivDisen scales with 2, 3, 4, 5 passive parties.

Usage:
    python experiments/run_multi_party.py --config configs/default.yaml --dataset cifar10
"""

import sys
import os

# Windows console UTF-8 support
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json

from utils import load_config, set_seed, get_logger
from data import load_dataset, get_num_classes, is_image_dataset, build_vfl_dataloaders
from trainers import VFLTrainer, PrivDisenTrainer
from evaluation import evaluate_attacks, compute_metrics, format_metrics


def main():
    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    logger = get_logger("multi_party", cfg.get("log_dir", "results/logs"))

    dataset = cfg.get("dataset", "cifar10")
    device = cfg.get("device", "cuda:0")
    party_counts = [2, 3, 4, 5]
    methods = ["vanilla", "privdisen"]

    num_classes = get_num_classes(dataset)
    is_img = is_image_dataset(dataset)
    data_dir = cfg.get("data_dir", "data/raw")
    X_train, y_train, X_test, y_test = load_dataset(dataset, data_dir)

    all_results = {}

    for num_parties in party_counts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with {num_parties} passive parties")
        logger.info(f"{'='*60}")

        train_loader, test_loader, feature_dims = build_vfl_dataloaders(
            X_train, y_train, X_test, y_test,
            num_parties=num_parties,
            is_image=is_img,
            batch_size=cfg.get("batch_size", 256),
            num_workers=cfg.get("num_workers", 4),
        )

        for method in methods:
            logger.info(f"\n--- {method}, {num_parties} parties ---")

            if method == "vanilla":
                trainer = VFLTrainer(
                    feature_dims=feature_dims,
                    num_classes=num_classes,
                    bottom_model_type="cnn" if is_img else "mlp",
                    embedding_dim=128,
                    lr=cfg.get("lr", 1e-3),
                    device=device,
                )
            elif method == "privdisen":
                trainer = PrivDisenTrainer(
                    feature_dims=feature_dims,
                    num_classes=num_classes,
                    bottom_model_type="cnn" if is_img else "mlp",
                    embedding_dim=cfg.get("task_dim", 128),
                    task_dim=cfg.get("task_dim", 128),
                    private_dim=cfg.get("private_dim", 64),
                    beta=cfg.get("beta", 0.01),
                    gamma=cfg.get("gamma", 1.0),
                    delta=cfg.get("delta", 0.1),
                    lr=cfg.get("lr", 1e-3),
                    device=device,
                )

            trainer.fit(
                train_loader, test_loader,
                epochs=cfg.get("epochs", 100),
                eval_every=cfg.get("eval_every", 5),
            )

            test_metrics = trainer.evaluate(test_loader)

            # Get embeddings
            if method == "privdisen":
                emb, _, labels = trainer.get_embeddings(test_loader)
            else:
                emb, labels = trainer.get_embeddings(test_loader)

            attack_names = cfg.get("attacks", ["norm", "direction", "model_completion"])
            attack_results = evaluate_attacks(emb, labels, num_classes, attack_names, device)
            metrics = compute_metrics(test_metrics["test_acc"], attack_results, num_classes)

            key = f"{method}_{num_parties}p"
            all_results[key] = metrics
            logger.info(f"\n{format_metrics(metrics)}")

    # Save all results
    result_path = os.path.join(
        cfg.get("log_dir", "results/logs"),
        f"multi_party_{dataset}_results.json",
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {result_path}")


if __name__ == "__main__":
    main()
