# PrivDisen: Privacy-Preserving Disentangled Representation Learning

> A paper study and implementation of privacy-aware disentanglement for federated and centralized learning.

## Overview

**PrivDisen** studies the problem of *privacy-preserving disentangled representation learning*. The core idea is to decompose a data sample's latent representation into two orthogonal components:

1. **Shared (task-relevant) representation** – features that are useful for the downstream task and safe to share.
2. **Private (sensitive) representation** – features that encode sensitive or identity-specific attributes that must be protected.

By explicitly separating these components, PrivDisen allows models to:
- Perform accurate predictions using only the shared representation.
- Prevent leakage of sensitive attributes from the shared features.
- Support privacy-aware federated learning where only safe representations are transmitted.

### Key Concepts

| Concept | Description |
|---|---|
| Disentanglement | Decompose latent space into independent semantic components |
| Privacy protection | Apply differential privacy / adversarial suppression to sensitive features |
| Mutual information minimization | Minimise MI between shared and private representations |
| Reconstruction | Decode from both components to preserve information |

## Architecture

```
Input x
   │
   ▼
┌──────────────┐
│   Encoder E  │
└──────┬───────┘
       │  z
   ┌───┴───┐
   ▼       ▼
 E_s(·)  E_p(·)      ← Shared / Private branch encoders
   │       │
  z_s     z_p        ← Shared / Private latent vectors
   │       │
   └───┬───┘
       │
   ┌───▼────┐
   │Decoder │  → x̂   (reconstruction)
   └────────┘
   
  z_s ──► Task Head  → ŷ   (prediction)
  z_s ─── Privacy Discriminator (adversarial suppression of sensitive info)
```

## Paper Notes

### Related Works

- **β-VAE** (Higgins et al., 2017): Disentangled VAE representations via a β-weighted KL term.
- **Factor-VAE** (Kim & Mnih, 2018): Total correlation penalisation for disentanglement.
- **TCVAE** (Chen et al., 2018): Decomposition of ELBO into MI, TC, and dimension-wise KL.
- **PrivateMixer / InfoDisent**: Federated privacy via disentangled representation mixing.
- **PPFL** (Li et al., 2021): Privacy-preserving federated learning with representation separation.

### Loss Function

The total training objective combines:

```
L = L_recon + α·L_task + β·L_KL + γ·L_mi + δ·L_adv
```

where:
- `L_recon`: Reconstruction loss (MSE or cross-entropy)
- `L_task`: Supervised task loss on shared representation
- `L_KL`: KL divergence for the VAE prior
- `L_mi`: Mutual information penalty between z_s and z_p
- `L_adv`: Adversarial loss to suppress sensitive info in z_s

## Project Structure

```
PrivDisen/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── default.yaml          # Default hyperparameters
├── data/
│   └── datasets.py           # Dataset loaders
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py        # Shared + private encoders
│   │   ├── decoder.py        # Reconstruction decoder
│   │   ├── discriminator.py  # Privacy discriminator
│   │   └── privdisen.py      # Full model
│   ├── losses/
│   │   ├── __init__.py
│   │   └── losses.py         # All loss functions
│   └── utils/
│       ├── __init__.py
│       └── utils.py          # Helpers (metrics, logging, etc.)
├── experiments/
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
└── tests/
    └── test_models.py        # Unit tests
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python experiments/train.py --config configs/default.yaml
```

### Evaluation

```bash
python experiments/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.pt
```

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `latent_dim` | 128 | Total latent dimension |
| `shared_dim` | 64 | Shared (task) representation dimension |
| `private_dim` | 64 | Private (sensitive) representation dimension |
| `alpha` | 1.0 | Task loss weight |
| `beta` | 0.1 | KL divergence weight |
| `gamma` | 0.5 | Mutual information penalty weight |
| `delta` | 0.5 | Adversarial privacy loss weight |
| `lr` | 1e-3 | Learning rate |
| `batch_size` | 64 | Batch size |

## License

MIT
