"""
Ablation study (Table 3).
Tests the impact of each loss component.

Usage:
    python experiments/run_ablation.py --config configs/default.yaml --dataset cifar10
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
from trainers import PrivDisenTrainer
from evaluation import evaluate_attacks, compute_metrics, format_metrics


ABLATION_CONFIGS = {
    "full":              {"beta": 0.01, "gamma": 1.0,  "delta": 0.1, "alpha_max": 1.0},
    "w/o L_adv":         {"beta": 0.01, "gamma": 1.0,  "delta": 0.1, "alpha_max": 0.0},
    "w/o L_MI":          {"beta": 0.0,  "gamma": 1.0,  "delta": 0.1, "alpha_max": 1.0},
    "w/o L_recon":       {"beta": 0.01, "gamma": 0.0,  "delta": 0.1, "alpha_max": 1.0},
    "w/o L_indep":       {"beta": 0.01, "gamma": 1.0,  "delta": 0.0, "alpha_max": 1.0},
    "only task+adv":     {"beta": 0.0,  "gamma": 0.0,  "delta": 0.0, "alpha_max": 1.0},
}


def main():
    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    logger = get_logger("ablation", cfg.get("log_dir", "results/logs"))

    dataset = cfg.get("dataset", "cifar10")
    device = cfg.get("device", "cuda:0")
    num_parties = cfg.get("num_parties", 2)

    num_classes = get_num_classes(dataset)
    is_img = is_image_dataset(dataset)
    X_train, y_train, X_test, y_test = load_dataset(dataset, cfg.get("data_dir", "data/raw"))

    train_loader, test_loader, feature_dims = build_vfl_dataloaders(
        X_train, y_train, X_test, y_test,
        num_parties=num_parties, is_image=is_img,
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 4),
    )

    all_results = {}

    for variant_name, variant_cfg in ABLATION_CONFIGS.items():
        logger.info(f"\n--- Ablation: {variant_name} ---")
        set_seed(cfg.get("seed", 42))  # reset seed for fairness

        trainer = PrivDisenTrainer(
            feature_dims=feature_dims,
            num_classes=num_classes,
            bottom_model_type="cnn" if is_img else "mlp",
            embedding_dim=cfg.get("task_dim", 128),
            task_dim=cfg.get("task_dim", 128),
            private_dim=cfg.get("private_dim", 64),
            alpha_max=variant_cfg["alpha_max"],
            beta=variant_cfg["beta"],
            gamma=variant_cfg["gamma"],
            delta=variant_cfg["delta"],
            lr=cfg.get("lr", 1e-3),
            device=device,
        )

        trainer.fit(
            train_loader, test_loader,
            epochs=cfg.get("epochs", 100),
            eval_every=cfg.get("eval_every", 5),
        )

        test_metrics = trainer.evaluate(test_loader)
        emb, _, labels = trainer.get_embeddings(test_loader)

        attack_names = cfg.get("attacks", ["norm", "direction", "model_completion"])
        attack_results = evaluate_attacks(emb, labels, num_classes, attack_names, device)
        metrics = compute_metrics(test_metrics["test_acc"], attack_results, num_classes)

        all_results[variant_name] = metrics
        logger.info(f"\n{format_metrics(metrics)}")

    # Save
    result_path = os.path.join(
        cfg.get("log_dir", "results/logs"),
        f"ablation_{dataset}_results.json",
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAblation results saved to {result_path}")


if __name__ == "__main__":
    main()
