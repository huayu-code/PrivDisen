# PrivDisen

**Privacy-Preserving Label Protection via Variational Disentangled Representation in Vertical Federated Learning**

**基于变分解耦表征的纵向联邦学习标签隐私保护**

<p align="center">
  <img src="assets/architecture.png" width="700" alt="PrivDisen Architecture">
</p>

---

## 1. Introduction

### 1.1 Background: Vertical Federated Learning

**纵向联邦学习（Vertical Federated Learning, VFL）** 是一种多方协同机器学习范式。在 VFL 中，多个参与方持有**相同样本集的不同特征**（例如银行持有交易数据，电商持有购物数据，两者共享相同用户 ID），通过协同训练获得比任何单方数据更好的模型。

VFL 的典型架构为 **Split Learning**：
- **被动方（Passive Party）**：持有特征 X，训练底层模型（Bottom Model），输出中间嵌入（embedding）发送给主动方
- **主动方（Active Party）**：持有标签 Y，训练顶层模型（Top Model），计算损失并将梯度回传给被动方

```
被动方 1: X_1 → BottomModel_1 → h_1 ─┐
被动方 2: X_2 → BottomModel_2 → h_2 ─┼─→ concat(h_1,..,h_N) → TopModel → y_hat
   ...                                │                         ↑
被动方 N: X_N → BottomModel_N → h_N ─┘                    labels Y (主动方)
```

### 1.2 Problem: Label Leakage in VFL

虽然 VFL 中各方不直接共享原始数据，但研究表明**标签信息可通过多种渠道泄露**：

| 攻击类型 | 代表工作 | 攻击原理 |
|---------|---------|---------|
| **Norm Attack** | Li et al., 2022 | 正类样本的梯度范数显著大于负类 |
| **Direction Attack** | Fu et al., 2022 | 梯度方向编码了标签信息，余弦相似度可区分类别 |
| **Model Completion** | Fu et al., 2022 | 被动方在本地训练一个分类头直接推断标签 |
| **Embedding Extension** | Ye et al., 2024 | 扩展嵌入空间，通过梯度反向工程恢复标签 |

这些攻击意味着：**即使不共享原始数据，VFL 中的梯度和嵌入也会泄露标签隐私**。

### 1.3 Existing Defenses and Their Limitations

| 防御方法 | 核心思路 | 主要局限 |
|---------|---------|---------|
| **DP-VFL** | 在梯度上加高斯噪声（差分隐私） | 噪声太大严重降低模型精度 |
| **SVFL** (Zhang 2023) | 两个编码器硬分割特征为 task/private | 确定性分割无理论保证，无可调参数 |
| **LabObf** (He 2024) | 用随机矩阵混淆软标签 | 仅防梯度攻击，不防模型完成攻击 |
| **KDk** (Arazzi 2025) | 知识蒸馏 + k-匿名梯度 | k 过大严重影响收敛，无隐私上界 |
| **MID** (Zou 2023) | 互信息正则化 | 直接约束嵌入 MI，未做子空间分离 |

**核心问题**：现有方法要么缺乏**理论隐私保证**，要么**隐私-效用权衡不可调**，要么**仅防特定攻击类型**。

### 1.4 Our Method: PrivDisen

PrivDisen 提出一种**基于变分推断的解耦表征**方法，在被动方将中间嵌入分解为两个子空间：

- **Z_task**（任务相关表征）：仅包含完成主任务所需的信息，传输给主动方
- **Z_private**（标签敏感表征）：包含可能泄露标签的信息，**保留在本地不传输**

#### 核心技术创新

**1. 变分解耦模块（VDM）**

不同于 SVFL 的确定性双编码器，VDM 使用**参数化概率分布**（重参数化高斯）实现软分离：

```
h → VDM → Z_task ~ N(μ_task, σ²_task)    # 传输给主动方
         → Z_private ~ N(μ_priv, σ²_priv)  # 保留在本地
```

参数化分布的优势：(1) 天然支持互信息上界约束；(2) 分离边界是软的、可学习的；(3) 重参数化技巧保证端到端可微。

**2. 五项联合损失函数**

```
L_total = L_task + α·L_adv + β·L_MI + γ·L_recon + δ·L_indep
```

| 损失项 | 公式 | 作用 |
|-------|------|------|
| L_task | CrossEntropy(y_hat, y) | 主分类任务 |
| L_adv | CE(ALC(GRL(Z_task)), y) | 通过梯度反转层确保 Z_task 不携带标签信号 |
| L_MI | KL(q(Z_task\|X) \|\| N(0,I)) | 约束 I(Z_task; X) 的上界（VIB 原理） |
| L_recon | MSE(Decoder(Z_task, Z_priv), h) | 确保信息无损分解 |
| L_indep | HSIC(Z_task, Z_priv) | Z_task 与 Z_private 的统计独立性 |

**3. 可证明的隐私保证**

基于数据处理不等式 (DPI) 和 Fano 不等式：
- I(Z_task; Y) ≤ I(Z_task; X) ≤ KL(q||p) = L_MI
- 由 Fano 不等式，攻击者最优推断准确率 P_attack ≤ (L_MI + log2) / logK

因此 **β 越大 → L_MI 越小 → P_attack 上界越低 → 隐私保护越强**，但模型精度可能下降。β 是唯一的**隐私旋钮（privacy knob）**。

**4. 对抗训练 + DANN 渐进式调度**

α 采用 DANN 式 sigmoid 调度：训练初期 α≈0（模型先学任务），后期 α→1（强化隐私对抗），避免对抗训练早期不稳定。

### 1.5 Comparison with SVFL

| 维度 | SVFL (Zhang 2023) | PrivDisen (Ours) |
|------|-------------------|-----------------|
| 分离方式 | 确定性双编码器（硬分割） | 变分推断（软分离，概率分布） |
| 理论保证 | 无 | Fano + MI bound，显式攻击上界 |
| 可调性 | 固定 | β 可调，Pareto 曲线 |
| 独立性约束 | 无 | HSIC 约束 Z_task ⊥ Z_private |
| 攻击覆盖 | 仅被动 | 被动 + 主动 + 模型完成 |
| 信息保全 | 无 | 重构损失确保无损分解 |

---

## 2. System Architecture

```
Passive Party                           Active Party
┌──────────────────────────┐          ┌────────────────────┐
│  X ──► Bottom Model      │          │   Top Model        │
│         │                │          │      │             │
│    ┌────▼────┐           │  Z_task  │   ┌──▼──┐         │
│    │   VDM   │───────────│────────► │───│Fusion│──► y_hat│
│    │ ┌─────┐ │           │          │   └─────┘         │
│    │ │Z_priv│ │           │          │      │             │
│    │ └──┬──┘ │           │   ∇'     │  L_task(y_hat, y) │
│    └────┘    │           │◄─────────│  Gradient Purify   │
│  (kept local)│           │          └────────────────────┘
└──────────────────────────┘

Adversarial Label Classifier (ALC)
   ← Gradient Reversal Layer (GRL) ←── Z_task
```

---

## 3. Environment Setup (Step-by-Step)

### 3.1 Prerequisites

- Python 3.9+
- CUDA 11.x+ (optional; auto-fallback to CPU)
- Git
- Windows / Linux / macOS

### 3.2 Full Setup

```bash
# Step 1: Clone
git clone https://github.com/<your-username>/PrivDisen.git
cd PrivDisen

# Step 2: Create environment (choose one)
conda create -n privdisen python=3.9 -y && conda activate privdisen
# OR: python -m venv venv && source venv/bin/activate  (Linux/Mac)
# OR: python -m venv venv && venv\Scripts\activate      (Windows)

# Step 3: Install PyTorch (MUST do this FIRST)
# GPU + CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# GPU + CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install dependencies (China mirror recommended)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Step 5: Download datasets
python data/download.py --dataset all

# Step 6: Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> **Tip**: Set pip mirror permanently:
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 4. Experiments

### 4.1 Experiment Overview

We conduct **5 groups** of experiments for the paper:

| # | Experiment | Paper Section | Script | What it produces |
|---|-----------|---------------|--------|-----------------|
| **Exp1** | Main Comparison | Table 1 | `run_main.py` | 6 methods × 3 datasets × 3 attacks → MTA, ASR, PUT |
| **Exp2** | Multi-Party Scaling | Table 2 | `run_multi_party.py` | PrivDisen with 2/3/4/5 parties → scalability |
| **Exp3** | Ablation Study | Table 3 | `run_ablation.py` | Remove each loss component → contribution analysis |
| **Exp4** | Privacy-Utility Pareto | Figure | `run_all_experiments.py` | Sweep β → MTA vs ASR curve |
| **Exp5** | Visualization | Figure | `run_all_experiments.py` | t-SNE of Z_task / Z_private, training curves |

### 4.2 Methods Compared (6 total)

| Method | Type | Key Parameter | Command |
|--------|------|---------------|---------|
| **Vanilla** | No defense | — | `--method vanilla` |
| **SVFL** | Feature disentangle | alpha_schedule | `--method svfl` |
| **LabObf** | Label obfuscation | eps=0.3 | `--method labobf` |
| **KDk** | Knowledge distillation | temperature=4, k=4 | `--method kdk` |
| **MID** | MI regularization | mi_weight=0.01 | `--method mid` |
| **PrivDisen** | Ours | beta=0.01 | `--method privdisen` |

### 4.3 Datasets

| Dataset | Type | Samples | Features | Classes | VFL Split |
|---------|------|---------|----------|---------|-----------|
| CIFAR-10 | Image | 60K | 3x32x32 | 10 | Channel / Spatial |
| Adult | Tabular | 48K | 14 | 2 | Column-wise |
| Bank | Tabular | 45K | 16 | 2 | Column-wise |

### 4.4 Attack Metrics

| Metric | Symbol | Direction | Meaning |
|--------|--------|-----------|---------|
| Main Task Accuracy | MTA | ↑ higher is better | Model accuracy on the main classification task |
| Attack Success Rate | ASR | ↓ lower is better | Attacker's label inference accuracy |
| Privacy-Utility Tradeoff | PUT | ↑ higher is better | MTA / ASR ratio |

### 4.5 How to Run Each Experiment

#### Exp1: Main Comparison (Table 1)

```bash
# Run all 6 methods on all 3 datasets (auto-saves results to results/)
python scripts/run_all_experiments.py --experiment main --epochs 100

# Or run one method at a time:
python experiments/run_main.py --method vanilla --dataset cifar10
python experiments/run_main.py --method svfl --dataset cifar10
python experiments/run_main.py --method labobf --dataset cifar10
python experiments/run_main.py --method kdk --dataset cifar10
python experiments/run_main.py --method mid --dataset cifar10
python experiments/run_main.py --method privdisen --dataset cifar10 --beta 0.01
```

#### Exp2: Multi-Party Scaling (Table 2)

```bash
python scripts/run_all_experiments.py --experiment multi_party --epochs 100

# Or manually:
python experiments/run_multi_party.py --dataset cifar10 --epochs 100
```

#### Exp3: Ablation Study (Table 3)

```bash
python scripts/run_all_experiments.py --experiment ablation --epochs 100

# Or manually:
python experiments/run_ablation.py --dataset cifar10 --epochs 100
```

#### Exp4: Pareto Curve (beta sweep)

```bash
python scripts/run_all_experiments.py --experiment pareto --epochs 100
```

This sweeps `beta` over {0.001, 0.005, 0.01, 0.05, 0.1, 0.5} and records MTA vs ASR.

#### Exp5: Run ALL experiments at once

```bash
# This runs Exp1-Exp4 sequentially and saves everything
python scripts/run_all_experiments.py --experiment all --epochs 100

# With custom device
python scripts/run_all_experiments.py --experiment all --epochs 100 --device cuda:0
```

All results are automatically saved to:
- `results/logs/` — training logs
- `results/experiment_records/` — CSV/JSON with all metrics
- `results/checkpoints/` — model weights
- `results/figures/` — plots and visualizations

### 4.6 Expected Results

| Method | CIFAR-10 MTA ↑ | CIFAR-10 ASR ↓ | Privacy Improvement |
|--------|----------------|----------------|---------------------|
| Vanilla | ~92% | ~85% | None (baseline) |
| SVFL | ~90% | ~45% | Moderate |
| LabObf | ~88% | ~55% | Low-Moderate |
| KDk | ~89% | ~50% | Moderate |
| MID | ~90% | ~40% | Good |
| **PrivDisen** | **~91%** | **~15%** | **Best** |

> Note: These are target ranges. Actual numbers depend on hyperparameters and training.

---

## 5. Project Structure

```
PrivDisen/
├── configs/
│   └── default.yaml                 # Default hyperparameters
│
├── data/
│   ├── datasets.py                  # Dataset loaders (CIFAR/MNIST/Adult/Bank)
│   ├── vfl_partition.py             # N-party VFL feature partitioning
│   └── download.py                  # Auto download with China mirrors
│
├── models/
│   ├── bottom_model.py              # Passive party bottom models (CNN/MLP)
│   ├── top_model.py                 # Active party top classifier
│   ├── vdm.py                      # ★ Variational Disentangle Module
│   ├── adversarial.py              # ★ ALC + GRL + alpha scheduling
│   ├── gradient_purifier.py        # Gradient purification (optional)
│   └── reconstruction.py           # Reconstruction decoder
│
├── losses/
│   ├── task_loss.py                # Cross-entropy (main task)
│   ├── mi_loss.py                  # KL divergence (MI upper bound)
│   ├── hsic_loss.py                # HSIC independence constraint
│   └── reconstruction_loss.py      # MSE reconstruction
│
├── attacks/
│   ├── norm_attack.py              # Gradient norm passive attack
│   ├── direction_attack.py         # Gradient direction passive attack
│   ├── model_completion.py         # Model completion attack
│   └── embedding_extension.py      # Embedding extension attack
│
├── baselines/
│   ├── svfl.py                     # SVFL (Zhang et al., 2023)
│   ├── labobf.py                   # LabObf (He et al., 2024)
│   ├── kdk.py                      # KDk (Arazzi et al., 2025)
│   └── mid.py                      # MID (Zou et al., 2023)
│
├── trainers/
│   ├── vfl_trainer.py              # Vanilla VFL trainer
│   └── privdisen_trainer.py        # ★ PrivDisen trainer
│
├── evaluation/
│   ├── metrics.py                  # MTA, ASR, PUT metrics
│   ├── attack_eval.py              # Attack evaluation pipeline
│   └── visualization.py            # t-SNE, Pareto, training curves
│
├── experiments/
│   ├── run_main.py                 # Main comparison (Table 1)
│   ├── run_multi_party.py          # Multi-party experiment (Table 2)
│   └── run_ablation.py             # Ablation study (Table 3)
│
├── scripts/
│   ├── run_all_experiments.py      # ★ Automated experiment runner
│   ├── train.py                    # Quick training (cross-platform)
│   └── eval.py                     # Quick evaluation (cross-platform)
│
├── results/                        # Output (gitignored)
│   ├── logs/
│   ├── checkpoints/
│   ├── figures/
│   └── experiment_records/         # Auto-saved CSV/JSON
│
├── setup.py
├── requirements.txt
├── PrivDisen_Handover.md           # Project roadmap
└── README.md
```

---

## 6. Configuration

All hyperparameters in `configs/default.yaml`. Key parameters:

```yaml
# Device
device: "auto"            # auto | cuda:0 | cpu

# VFL
num_parties: 2
batch_size: 256
epochs: 100

# PrivDisen
task_dim: 128             # Z_task dimension
private_dim: 64           # Z_private dimension
beta: 0.01                # MI constraint (privacy knob: larger = more private)
gamma: 1.0                # Reconstruction weight
delta: 0.1                # HSIC independence weight
alpha_schedule: "dann"    # Adversarial strength: dann | linear | constant
```

Override from command line:
```bash
python experiments/run_main.py --beta 0.1 --num_parties 3 --epochs 50
```

---

## 7. FAQ

**Q: `No module named 'data'`?**
```bash
pip install -e .
```

**Q: Windows `bash scripts/train.sh` error?**
Use Python scripts instead: `python scripts/train.py`

**Q: No GPU?**
Works on CPU automatically. `device: "auto"` detects CUDA.

**Q: Dataset download slow?**
Auto-tries ModelScope (Aliyun) → China mirrors → official. Manual: download to `data/raw/`.

---

## 8. Citation

```bibtex
@article{privdisen2026,
  title     = {PrivDisen: Privacy-Preserving Label Protection via Variational
               Disentangled Representation in Vertical Federated Learning},
  author    = {Tan, Yang},
  year      = {2026},
  note      = {Preprint}
}
```

---

## 9. Related Work

- **SVFL** – Zhang et al., *Signal Processing* 2023
- **LabObf** – He et al., 2024
- **KDk** – Arazzi et al., *Neurocomputing* 2025
- **LADSG** – Yan et al., 2025
- **VMask** – Tan et al., *FCS* 2025
- **MID** – Zou et al., 2023

## License

MIT License. See [LICENSE](LICENSE).
