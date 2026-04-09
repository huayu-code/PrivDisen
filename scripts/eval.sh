#!/bin/bash
# ================================================
# PrivDisen Evaluation Scripts
# ================================================
set -e

DEVICE="cuda:0"

echo "=============================="
echo " PrivDisen Evaluation Pipeline"
echo "=============================="

# --- Multi-party experiment ---
echo ">>> Multi-party experiment (CIFAR-10)"
python experiments/run_multi_party.py \
    --config configs/default.yaml \
    --dataset cifar10 \
    --epochs 100 \
    --device ${DEVICE}

# --- Ablation study ---
echo ""
echo ">>> Ablation study (CIFAR-10)"
python experiments/run_ablation.py \
    --config configs/default.yaml \
    --dataset cifar10 \
    --epochs 100 \
    --device ${DEVICE}

echo ""
echo "=== Evaluation complete ==="
