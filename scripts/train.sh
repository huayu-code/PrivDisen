#!/bin/bash
# ================================================
# PrivDisen Training Scripts
# ================================================
set -e

DEVICE="cuda:0"
EPOCHS=100

echo "=============================="
echo " PrivDisen Training Pipeline"
echo "=============================="

# --- Main experiments ---
for DATASET in cifar10 adult bank; do
    for METHOD in vanilla privdisen; do
        echo ""
        echo ">>> Training: method=${METHOD}, dataset=${DATASET}"
        python experiments/run_main.py \
            --config configs/default.yaml \
            --method ${METHOD} \
            --dataset ${DATASET} \
            --epochs ${EPOCHS} \
            --device ${DEVICE} \
            --num_parties 2
    done
done

echo ""
echo "=== Training complete ==="
