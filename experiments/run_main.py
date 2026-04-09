"""
Main experiment: train a defense method and evaluate under attacks.

Usage:
    python experiments/run_main.py --config configs/default.yaml --method privdisen
    python experiments/run_main.py --method vanilla --dataset cifar10 --num_parties 2
"""

import sys
import os

# Ensure project root is on sys.path regardless of where the script is invoked
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json

from utils import load_config, set_seed, get_logger
from data import load_dataset, get_num_classes, is_image_dataset, build_vfl_dataloaders
from trainers import VFLTrainer, PrivDisenTrainer
from evaluation import evaluate_attacks, compute_metrics, format_metrics
from evaluation.visualization import plot_training_curves, plot_tsne


def main():
    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    logger = get_logger("main", cfg.get("log_dir", "results/logs"))

    dataset = cfg.get("dataset", "cifar10")
    method = cfg.get("method", "privdisen")
    num_parties = cfg.get("num_parties", 2)
    device = cfg.get("device", "cuda:0")

    logger.info(f"=== Experiment: method={method}, dataset={dataset}, "
                f"parties={num_parties}, device={device} ===")

    # ---- Data ----
    num_classes = get_num_classes(dataset)
    is_img = is_image_dataset(dataset)
    data_dir = cfg.get("data_dir", "data/raw")
    X_train, y_train, X_test, y_test = load_dataset(dataset, data_dir)

    if is_img:
        # flatten for VFL partition
        pass  # partition_features_image handles 4D arrays

    train_loader, test_loader, feature_dims = build_vfl_dataloaders(
        X_train, y_train, X_test, y_test,
        num_parties=num_parties,
        is_image=is_img,
        batch_size=cfg.get("batch_size", 256),
        num_workers=cfg.get("num_workers", 4),
    )

    logger.info(f"Feature dims per party: {feature_dims}")

    # ---- Trainer ----
    embedding_dim = cfg.get("task_dim", 128) if method != "vanilla" else 128

    if method == "vanilla":
        trainer = VFLTrainer(
            feature_dims=feature_dims,
            num_classes=num_classes,
            bottom_model_type=cfg.get("bottom_model", "mlp") if not is_img else "cnn",
            embedding_dim=embedding_dim,
            top_hidden_dim=cfg.get("top_hidden_dim", 256),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
            device=device,
            log_dir=cfg.get("log_dir"),
        )
    elif method == "privdisen":
        trainer = PrivDisenTrainer(
            feature_dims=feature_dims,
            num_classes=num_classes,
            bottom_model_type="cnn" if is_img else "mlp",
            embedding_dim=embedding_dim,
            task_dim=cfg.get("task_dim", 128),
            private_dim=cfg.get("private_dim", 64),
            vdm_hidden_dim=cfg.get("vdm_hidden_dim", 256),
            top_hidden_dim=cfg.get("top_hidden_dim", 256),
            alc_hidden_dims=cfg.get("alc_hidden_dims", [128, 64]),
            alpha_schedule=cfg.get("alpha_schedule", "dann"),
            alpha_max=cfg.get("alpha_max", 1.0),
            beta=cfg.get("beta", 0.01),
            gamma=cfg.get("gamma", 1.0),
            delta=cfg.get("delta", 0.1),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
            use_gradient_purifier=cfg.get("use_gradient_purifier", False),
            device=device,
            log_dir=cfg.get("log_dir"),
        )
    else:
        raise ValueError(f"Method '{method}' not yet implemented. "
                         f"Use 'vanilla' or 'privdisen'.")

    # ---- Train ----
    if not cfg.get("eval_only", False):
        epochs = cfg.get("epochs", 100)
        history = trainer.fit(
            train_loader, test_loader,
            epochs=epochs,
            eval_every=cfg.get("eval_every", 5),
            checkpoint_dir=cfg.get("checkpoint_dir", "results/checkpoints"),
        )

        # Plot training curves
        fig_dir = cfg.get("figure_dir", "results/figures")
        plot_training_curves(
            history,
            save_path=os.path.join(fig_dir, f"{method}_{dataset}_training.png"),
        )
    else:
        ckpt = cfg.get("checkpoint")
        if ckpt:
            trainer.load(ckpt)
            logger.info(f"Loaded checkpoint: {ckpt}")

    # ---- Evaluate ----
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Accuracy: {test_metrics['test_acc']:.4f}")

    # ---- Attack evaluation ----
    attack_names = cfg.get("attacks", ["norm", "direction", "model_completion"])

    if method == "privdisen":
        z_task, z_priv, labels = trainer.get_embeddings(test_loader)
        embeddings = z_task
    else:
        embeddings, labels = trainer.get_embeddings(test_loader)

    attack_results = evaluate_attacks(
        embeddings, labels, num_classes, attack_names, device=device,
    )

    metrics = compute_metrics(test_metrics["test_acc"], attack_results, num_classes)
    logger.info(f"\n=== Results ===\n{format_metrics(metrics)}")

    # ---- Save results ----
    result_path = os.path.join(
        cfg.get("log_dir", "results/logs"),
        f"{method}_{dataset}_{num_parties}p_results.json",
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {result_path}")

    # ---- t-SNE visualization ----
    fig_dir = cfg.get("figure_dir", "results/figures")
    plot_tsne(
        embeddings.numpy(), labels.numpy(),
        title=f"{method} Z_task ({dataset})",
        save_path=os.path.join(fig_dir, f"{method}_{dataset}_tsne_ztask.png"),
    )

    if method == "privdisen":
        plot_tsne(
            z_priv.numpy(), labels.numpy(),
            title=f"{method} Z_private ({dataset})",
            save_path=os.path.join(fig_dir, f"{method}_{dataset}_tsne_zpriv.png"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
