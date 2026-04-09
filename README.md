<<<<<<< HEAD
# PrivDisen
=======
# PrivDisen

**Privacy-Preserving Label Protection via Variational Disentangled Representation in Vertical Federated Learning**

<p align="center">
  <img src="assets/architecture.png" width="700" alt="PrivDisen Architecture">
</p>

> PrivDisen decomposes intermediate embeddings in Vertical Federated Learning (VFL) into a **task-relevant** subspace and a **label-sensitive** subspace via variational inference, transmitting only the former to the active party while provably bounding label leakage through mutual-information constraints.

---

## Highlights

- **Variational Disentanglement Module (VDM):** Replaces the coarse two-classifier split of prior work (SVFL) with a probabilistic, continuously controllable separation.
- **Provable Privacy Guarantee:** An explicit mutual-information bound, derived from the Variational Information Bottleneck (VIB) and Fano's inequality, upper-bounds any attacker's label inference accuracy.
- **Tunable Privacy–Utility Trade-off:** A single scalar $\beta$ traces the full Pareto frontier between model accuracy and attack success rate.
- **Comprehensive Evaluation:** Tested against four families of label-inference attacks across six datasets and up to five participating parties.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Experiments](#experiments)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Architecture Overview

```
Passive Party                        Active Party
┌──────────────────────┐             ┌──────────────────┐
│  X ──► Bottom Model  │             │   Top Model      │
│         │             │             │      │           │
│    ┌────▼────┐        │   Z_task   │   ┌──▼──┐       │
│    │   VDM   │────────│──────────►─│───│Fuse │──► ŷ  │
│    │ ┌─────┐ │        │            │   └─────┘       │
│    │ │Z_priv│ │        │            │      │           │
│    │ └──┬──┘ │        │    ∇'      │   L_task(ŷ, y)  │
│    └────┘    │        │◄───────────│   Grad Purif.   │
│  (kept local)│        │            └──────────────────┘
└──────────────────────┘

Adversarial Label Classifier (ALC)
   ← Gradient Reversal Layer ←── Z_task
```

**Core Modules:**

| Module | Purpose |
|--------|---------|
| **VDM** (Variational Disentanglement Module) | Splits embeddings into Z_task and Z_private via reparameterized Gaussian |
| **ALC** (Adversarial Label Classifier) | Gradient-reversal–based adversary ensuring Z_task carries no label signal |
| **MI Loss** | KL divergence term upper-bounding $I(Z_{\text{task}}; Y)$ |
| **HSIC Loss** | Independence constraint between Z_task and Z_private |
| **Recon Loss** | Ensures lossless information decomposition |
| **Gradient Purifier** *(optional)* | Projects back-propagated gradients to remove label-correlated components |

---

## Installation

### Prerequisites

- Python 3.9
- CUDA 11.x+ (GPU required)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/PrivDisen.git
cd PrivDisen

# Create a virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
# See https://pytorch.org/get-started/locally/ for your specific CUDA version
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Install the project itself (editable mode — makes all modules importable)
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

---

## Project Structure

```
PrivDisen/
├── configs/                     # YAML configuration files
│   ├── default.yaml             # Default hyperparameters
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   ├── adult.yaml
│   └── bank.yaml
│
├── data/                        # Data loading & VFL partitioning
│   ├── datasets.py              # Dataset loaders
│   ├── vfl_partition.py         # N-party feature partitioning
│   └── download.py              # Auto-download script
│
├── models/                      # Model definitions
│   ├── bottom_model.py          # Passive-party encoder (CNN / MLP)
│   ├── top_model.py             # Active-party classifier
│   ├── vdm.py                   # ★ Variational Disentanglement Module
│   ├── adversarial.py           # ★ ALC + Gradient Reversal Layer
│   ├── gradient_purifier.py     # Gradient purification (optional)
│   └── reconstruction.py        # Reconstruction decoder
│
├── losses/                      # Loss functions
│   ├── task_loss.py             # Cross-entropy
│   ├── mi_loss.py               # KL-based MI upper bound
│   ├── hsic_loss.py             # HSIC independence criterion
│   └── reconstruction_loss.py   # MSE reconstruction
│
├── attacks/                     # Label inference attacks
│   ├── norm_attack.py           # Norm-based passive attack
│   ├── direction_attack.py      # Direction-based passive attack
│   ├── model_completion.py      # Model completion attack
│   └── embedding_extension.py   # Embedding extension attack
│
├── trainers/                    # Training orchestration
│   ├── vfl_trainer.py           # Base VFL trainer
│   ├── privdisen_trainer.py     # ★ PrivDisen trainer
│   └── baseline_trainers.py     # Baseline method trainers
│
├── baselines/                   # Baseline implementations
│   ├── vanilla_vfl.py
│   ├── dp_vfl.py
│   ├── svfl.py
│   ├── labobf.py
│   ├── kdk.py
│   ├── ladsg.py
│   └── mid.py
│
├── evaluation/                  # Evaluation & visualization
│   ├── metrics.py               # MTA, ASR, Privacy-Utility Trade-off
│   ├── attack_eval.py           # Attack evaluation pipeline
│   └── visualization.py         # t-SNE, Pareto curves, loss plots
│
├── experiments/                 # Experiment entry points
│   ├── run_main.py              # Main comparison (Table 1)
│   ├── run_multi_party.py       # Multi-party scaling (Table 2)
│   ├── run_ablation.py          # Ablation study (Table 3)
│   ├── run_pareto.py            # Pareto frontier (Figure 1)
│   ├── run_sensitivity.py       # Hyperparameter sensitivity (Figure 2)
│   └── run_visualization.py     # t-SNE visualization (Figure 4)
│
├── utils/                       # Utilities
│   ├── logger.py
│   ├── seed.py
│   └── config.py
│
├── scripts/                     # Shell scripts
│   ├── train.sh
│   └── eval.sh
│
├── results/                     # Output (git-ignored except .gitkeep)
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
│
├── assets/                      # Images for README
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Download Data

```bash
python data/download.py --dataset cifar10
```

### 2. Train PrivDisen (2-party, CIFAR-10)

```bash
python experiments/run_main.py \
    --config configs/cifar10.yaml \
    --method privdisen \
    --num_parties 2 \
    --beta 0.01 \
    --device cuda:0 \
    --seed 42
```

### 3. Evaluate Under Attacks

```bash
python experiments/run_main.py \
    --config configs/cifar10.yaml \
    --method privdisen \
    --eval_only \
    --checkpoint results/checkpoints/privdisen_cifar10_best.pt \
    --attacks norm direction model_completion \
    --device cuda:0
```

### 4. Run All Experiments

```bash
# Main comparison across all datasets and methods
bash scripts/train.sh

# Evaluate all
bash scripts/eval.sh
```

---

## Datasets

| Dataset | Type | Samples | Features | Classes | VFL Partition |
|---------|------|---------|----------|---------|---------------|
| CIFAR-10 | Image | 60K | 3×32×32 | 10 | Channel / spatial split |
| CIFAR-100 | Image | 60K | 3×32×32 | 100 | Channel / spatial split |
| MNIST | Image | 70K | 1×28×28 | 10 | Left-half / right-half |
| Adult | Tabular | 48K | 14 | 2 | Column-wise split |
| Bank | Tabular | 45K | 16 | 2 | Column-wise split |
| Criteo | Tabular | 100K | 39 | 2 | Column-wise split |

Datasets are auto-downloaded on first use. Tabular datasets are sourced from the UCI Machine Learning Repository.

---

## Experiments

### Experiment Overview

| Experiment | Script | Description |
|-----------|--------|-------------|
| **Main Comparison** | `run_main.py` | PrivDisen vs. 7 baselines across 6 datasets × 4 attacks |
| **Multi-Party** | `run_multi_party.py` | Scaling to 2, 3, 4, 5 passive parties |
| **Ablation** | `run_ablation.py` | Impact of each loss component |
| **Pareto Frontier** | `run_pareto.py` | Privacy–utility trade-off curves |
| **Sensitivity** | `run_sensitivity.py` | Hyperparameter β analysis |
| **Visualization** | `run_visualization.py` | t-SNE of Z_task vs Z_private |

### Key Metrics

| Metric | Symbol | Direction | Meaning |
|--------|--------|-----------|---------|
| Main Task Accuracy | MTA | ↑ | Model performance on the primary task |
| Attack Success Rate | ASR | ↓ | Attacker's label inference accuracy |
| Privacy-Utility Trade-off | PUT | ↑ | MTA / ASR ratio |

---

## Configuration

All hyperparameters are managed via YAML files in `configs/`. Key parameters:

```yaml
# Training
epochs: 100
batch_size: 256
lr: 0.001
optimizer: adam

# VFL
num_parties: 2           # Number of passive parties
task_dim: 128             # Dimension of Z_task
private_dim: 64           # Dimension of Z_private

# PrivDisen Loss Weights
alpha_schedule: "dann"    # Adversarial strength: gradual increase
beta: 0.01                # MI constraint strength (privacy knob)
gamma: 1.0                # Reconstruction weight
delta: 0.1                # HSIC independence weight

# Device
device: "cuda:0"
seed: 42
```

Override any parameter from command line:

```bash
python experiments/run_main.py --config configs/cifar10.yaml --beta 0.1 --num_parties 3
```

---

## Results

> Results will be populated after running experiments.

### Expected Outcomes

- **MTA**: Within 1–3% of vanilla VFL (no protection baseline)
- **ASR**: Reduced to near random-guess level (1/K for K classes)
- **Multi-party**: Consistent protection as the number of parties increases from 2 to 5
- **Pareto**: PrivDisen dominates existing methods (higher MTA at equivalent ASR)

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{privdisen2026,
  title     = {PrivDisen: Privacy-Preserving Label Protection via Variational Disentangled Representation in Vertical Federated Learning},
  author    = {Tan, Yang},
  year      = {2026},
  note      = {Preprint}
}
```

---

## Related Work

- **SVFL** – Zhang et al., *Signal Processing* 2023 — Feature disentanglement via two classifiers
- **LabObf** – He et al., 2024 — Label obfuscation through random soft-label mapping
- **KDk** – Arazzi et al., *Neurocomputing* 2025 — Knowledge distillation + k-anonymity
- **LADSG** – Yan et al., 2025 — Label-anonymized distillation + gradient substitution
- **VMask** – Tan et al., *FCS* 2025 — Layer masking via secret sharing
- **MID** – Zou et al., 2023 — Mutual information regularization defense

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds upon the [VFLAIR](https://github.com/FLAIR-THU/VFLAIR) benchmark for VFL attack and defense evaluation.
>>>>>>> c92b093 (init: project scaffold)
