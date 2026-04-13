# PrivDisen

**基于变分解耦表征的纵向联邦学习标签隐私保护**

*PrivDisen: Privacy-Preserving Label Protection via Variational Disentangled Representation in Vertical Federated Learning*

<p align="center">
  <img src="assets/architecture.png" width="700" alt="PrivDisen 系统架构">
</p>

> PrivDisen 通过变分推断将纵向联邦学习（VFL）中的中间嵌入分解为**任务相关子空间**和**标签敏感子空间**，仅传输前者给主动方，并通过互信息约束提供可证明的隐私保证。

---

## 核心亮点

- **变分解耦模块 (VDM)**：用参数化概率分布实现软分离，替代 SVFL 的两分类器硬分割
- **可证明隐私保证**：基于变分信息瓶颈（VIB）和 Fano 不等式，给出攻击者标签推断准确率的显式上界
- **可调隐私-效用权衡**：单一参数 $\beta$ 即可在隐私保护和模型精度之间灵活调节
- **全面评估**：4 类标签推断攻击 × 6 个数据集 × 最多 5 个参与方

---

## 目录

- [系统架构](#系统架构)
- [环境安装与快速开始](#环境安装与快速开始)
- [项目结构](#项目结构)
- [数据集](#数据集)
- [实验](#实验)
- [配置说明](#配置说明)
- [实验结果](#实验结果)
- [引用](#引用)
- [相关工作](#相关工作)
- [许可证](#许可证)

---

## 系统架构

```
被动方 (Passive Party)                 主动方 (Active Party)
┌──────────────────────┐             ┌──────────────────┐
│  X ──► Bottom Model  │             │   Top Model      │
│         │             │             │      │           │
│    ┌────▼────┐        │   Z_task   │   ┌──▼──┐       │
│    │   VDM   │────────│──────────►─│───│融合层│──► ŷ  │
│    │ ┌─────┐ │        │            │   └─────┘       │
│    │ │Z_priv│ │        │            │      │           │
│    │ └──┬──┘ │        │    ∇'      │   L_task(ŷ, y)  │
│    └────┘    │        │◄───────────│   梯度净化       │
│  (保留在本地) │        │            └──────────────────┘
└──────────────────────┘

对抗标签分类器 (ALC)
   ← 梯度反转层 (GRL) ←── Z_task
```

**核心模块：**

| 模块 | 作用 |
|------|------|
| **VDM** (变分解耦模块) | 将嵌入分解为 Z_task 和 Z_private，基于重参数化高斯分布 |
| **ALC** (对抗标签分类器) | 通过梯度反转确保 Z_task 不携带标签信号 |
| **MI Loss** | KL 散度项，约束 $I(Z_{\text{task}}; Y)$ 的上界 |
| **HSIC Loss** | Z_task 与 Z_private 之间的独立性约束 |
| **Recon Loss** | 确保信息无损分解（重构原始嵌入） |
| **梯度净化** *(可选)* | 投影去除反向传播梯度中的标签相关分量 |

---

## 环境安装与快速开始

### 前置要求

- Python 3.9+
- CUDA 11.x+（可选，有 GPU 则用 GPU，没有则自动回退到 CPU）
- Git

### 完整流程（从零到跑通实验）

#### Step 1：克隆仓库

```bash
git clone https://github.com/<你的用户名>/PrivDisen.git
cd PrivDisen
```

#### Step 2：创建虚拟环境

```bash
# Linux / macOS
python3.9 -m venv venv
source venv/bin/activate

# Windows（PowerShell）
python -m venv venv
venv\Scripts\activate

# Windows（CMD）
python -m venv venv
venv\Scripts\activate.bat

# conda（全平台通用）
conda create -n privdisen python=3.9 -y
conda activate privdisen
```

#### Step 3：安装 PyTorch

根据你的系统和 CUDA 版本选择（详见 [PyTorch 官网](https://pytorch.org/get-started/locally/)）：

```bash
# 有 NVIDIA GPU + CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 有 NVIDIA GPU + CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 无 GPU / 仅 CPU（也完全可以跑，只是慢一些）
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

#### Step 4：安装项目依赖

```bash
# 国内用户推荐使用清华镜像（快很多）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装项目本身（必须执行，否则会报 No module named 'data' 错误）
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **💡 可选：永久设置清华镜像**（设一次以后不用每次加 `-i`）
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

#### Step 5：下载数据集

```bash
# 下载 CIFAR-10（自动检测国内镜像）
python data/download.py --dataset cifar10

# 下载全部数据集
python data/download.py --dataset all
```

> **⚠️ 如果自动下载仍然失败**，可以手动下载：
>
> 1. 打开 [ModelScope CIFAR-10 页面](https://www.modelscope.cn/datasets/cutedataset/cifar-10/files)
> 2. 下载 `cifar-10-python.tar.gz`
> 3. 放到 `data/raw/` 目录下
> 4. 重新运行 `python data/download.py --dataset cifar10`，会自动检测并解压

#### Step 6：验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### Step 7：开始训练

> **注意**：`device` 默认为 `auto`（自动检测 GPU/CPU），无需手动指定。

```bash
# 训练 PrivDisen（2 方，CIFAR-10）
python experiments/run_main.py --config configs/default.yaml --method privdisen --num_parties 2 --beta 0.01

# 训练 Vanilla VFL（无保护基线，用于对比）
python experiments/run_main.py --config configs/default.yaml --method vanilla --dataset cifar10

# 指定 GPU
python experiments/run_main.py --config configs/default.yaml --method privdisen --device cuda:0
```

#### Step 8：评估攻击效果

```bash
python experiments/run_main.py --config configs/default.yaml --method privdisen --eval_only --checkpoint results/checkpoints/privdisen_best.pt --attacks norm direction model_completion
```

#### Step 9：运行全部实验

> **跨平台**：使用 Python 脚本代替 bash，Windows / Linux / macOS 均可直接运行。

```bash
# 主对比实验
python scripts/train.py

# 多方实验 + 消融实验
python scripts/eval.py

# 自定义参数
python scripts/train.py --device cuda:0 --epochs 50 --datasets cifar10 adult
python scripts/eval.py --device cpu --epochs 20
```

### ⚠️ 常见问题

**Q: 运行时报错 `No module named 'data'` 或 `No module named 'models'`**

A: 这是因为没有执行 `pip install -e .`。这一步将项目注册为可编辑 Python 包，使得 `data`、`models`、`losses` 等内部模块可以在任意位置正确导入。

```bash
pip install -e .
```

如果不想安装，也可以在运行时指定 `PYTHONPATH`：

```bash
# Linux / macOS
PYTHONPATH=. python experiments/run_main.py --method privdisen

# Windows PowerShell
$env:PYTHONPATH="."; python experiments/run_main.py --method privdisen

# Windows CMD
set PYTHONPATH=. && python experiments/run_main.py --method privdisen
```

**Q: Windows 上运行 `bash scripts/train.sh` 报错？**

A: `.sh` 脚本是 Linux/macOS 的 Shell 脚本，Windows 无法直接运行。请改用 Python 脚本：

```bash
# 代替 bash scripts/train.sh
python scripts/train.py

# 代替 bash scripts/eval.sh
python scripts/eval.py
```

**Q: 没有 GPU 能跑吗？**

A: 可以。`device` 默认为 `auto`，会自动检测：有 CUDA 用 GPU，否则用 CPU。CPU 模式下训练较慢，但完全可以正确运行。如需手动指定：

```bash
python experiments/run_main.py --device cpu --method privdisen
```

**Q: 数据集下载很慢怎么办？**

A: 下载脚本会自动尝试 ModelScope（阿里云 OSS）→ 国内镜像 → 官方源。如果仍然失败，参考 Step 5 中的手动下载方式，浏览器打开链接下载后放到 `data/raw/` 目录即可。

**Q: 可用的 pip 镜像源有哪些？**

| 镜像源 | 地址 |
|--------|------|
| 清华 | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple` |

---

## 项目结构

```
PrivDisen/
├── configs/                     # YAML 配置文件
│   └── default.yaml             # 默认超参数
│
├── data/                        # 数据加载与 VFL 划分
│   ├── datasets.py              # 数据集加载器（CIFAR-10/100, MNIST, Adult, Bank）
│   ├── vfl_partition.py         # N 方特征划分（支持图像和表格）
│   └── download.py              # 数据自动下载脚本
│
├── models/                      # 模型定义
│   ├── bottom_model.py          # 被动方底层模型（CNN / MLP）
│   ├── top_model.py             # 主动方顶层分类器
│   ├── vdm.py                   # ★ 变分解耦模块 (VDM)
│   ├── adversarial.py           # ★ 对抗标签分类器 (ALC) + 梯度反转层 (GRL)
│   ├── gradient_purifier.py     # 梯度净化模块（可选）
│   └── reconstruction.py        # 重构解码器
│
├── losses/                      # 损失函数
│   ├── task_loss.py             # 交叉熵（主任务）
│   ├── mi_loss.py               # KL 散度（互信息上界）
│   ├── hsic_loss.py             # HSIC 独立性约束
│   └── reconstruction_loss.py   # MSE 重构损失
│
├── attacks/                     # 标签推断攻击
│   ├── norm_attack.py           # 基于梯度范数的被动攻击
│   ├── direction_attack.py      # 基于梯度方向的被动攻击
│   ├── model_completion.py      # 模型完成攻击
│   └── embedding_extension.py   # 嵌入扩展攻击
│
├── trainers/                    # 训练器
│   ├── vfl_trainer.py           # Vanilla VFL 训练器（基线）
│   └── privdisen_trainer.py     # ★ PrivDisen 训练器（完整流程）
│
├── baselines/                   # 基线方法实现
│
├── evaluation/                  # 评估与可视化
│   ├── metrics.py               # MTA, ASR, PUT 等指标
│   ├── attack_eval.py           # 攻击评估流水线
│   └── visualization.py         # t-SNE、Pareto 曲线、训练曲线
│
├── experiments/                 # 实验入口
│   ├── run_main.py              # 主对比实验（表1）
│   ├── run_multi_party.py       # 多方扩展实验（表2）
│   └── run_ablation.py          # 消融实验（表3）
│
├── utils/                       # 工具函数
│   ├── logger.py                # 日志
│   ├── seed.py                  # 随机种子
│   └── config.py                # 配置管理（YAML + CLI）
│
├── scripts/                     # 运行脚本
│   ├── train.py                 # 一键训练（跨平台）
│   ├── eval.py                  # 一键评估（跨平台）
│   ├── train.sh                 # 一键训练（仅 Linux/macOS）
│   └── eval.sh                  # 一键评估（仅 Linux/macOS）
│
├── results/                     # 输出目录（已 gitignore）
│   ├── logs/
│   ├── checkpoints/
│   └── figures/
│
├── setup.py                     # 项目安装配置
├── requirements.txt             # 依赖清单
├── .gitignore
├── LICENSE
└── README.md
```

---

## 快速开始

### 1. 下载数据

```bash
python data/download.py --dataset cifar10
```

### 2. 训练 PrivDisen（2 方，CIFAR-10）

```bash
python experiments/run_main.py --config configs/default.yaml --method privdisen --num_parties 2 --beta 0.01
```

### 3. 评估攻击效果

```bash
python experiments/run_main.py --config configs/default.yaml --method privdisen --eval_only --checkpoint results/checkpoints/privdisen_best.pt --attacks norm direction model_completion
```

### 4. 运行全部实验

```bash
# 主对比实验
python scripts/train.py

# 多方实验 + 消融实验
python scripts/eval.py
```

---

## 数据集

| 数据集 | 类型 | 样本量 | 特征维度 | 类别数 | VFL 划分方式 |
|--------|------|--------|---------|--------|-------------|
| CIFAR-10 | 图像 | 60K | 3×32×32 | 10 | 按通道 / 按空间区域 |
| CIFAR-100 | 图像 | 60K | 3×32×32 | 100 | 按通道 / 按空间区域 |
| MNIST | 图像 | 70K | 1×28×28 | 10 | 左半 / 右半 |
| Adult | 表格 | 48K | 14 | 2 | 按特征列 |
| Bank | 表格 | 45K | 16 | 2 | 按特征列 |
| Criteo | 表格 | 100K | 39 | 2 | 按特征列 |

数据集在首次使用时自动下载。表格数据集来源于 UCI 机器学习仓库。

---

## 实验

### 实验总览

| 实验 | 脚本 | 说明 |
|------|------|------|
| **主对比实验** | `run_main.py` | PrivDisen vs. 7 个基线 × 6 数据集 × 4 种攻击 |
| **多方实验** | `run_multi_party.py` | 2/3/4/5 个被动方的扩展性实验 |
| **消融实验** | `run_ablation.py` | 各损失分量的贡献分析 |

### 核心指标

| 指标 | 符号 | 方向 | 含义 |
|------|------|------|------|
| 主任务准确率 | MTA | ↑ | 模型在主任务上的精度 |
| 攻击成功率 | ASR | ↓ | 攻击者推断标签的准确率 |
| 隐私-效用权衡 | PUT | ↑ | MTA / ASR 比值 |

---

## 配置说明

所有超参数通过 `configs/` 目录下的 YAML 文件管理。关键参数：

```yaml
# 训练
epochs: 100
batch_size: 256
lr: 0.001
optimizer: adam

# VFL 设置
num_parties: 2           # 被动方数量
task_dim: 128             # Z_task 维度
private_dim: 64           # Z_private 维度

# PrivDisen 损失权重
alpha_schedule: "dann"    # 对抗强度：渐进增大
beta: 0.01                # MI 约束强度（隐私旋钮）
gamma: 1.0                # 重构损失权重
delta: 0.1                # HSIC 独立性权重

# 设备
device: "cuda:0"
seed: 42
```

命令行覆盖任意参数：

```bash
python experiments/run_main.py --config configs/default.yaml --beta 0.1 --num_parties 3
```

---

## 实验结果

> 运行实验后自动填充。

### 预期效果

- **MTA**：与无保护的 Vanilla VFL 相比，精度损失控制在 1-3% 以内
- **ASR**：降低至接近随机猜测水平（K 分类时为 1/K）
- **多方场景**：2-5 方场景下保护效果一致
- **Pareto 曲线**：PrivDisen 在相同 ASR 下取得更高 MTA，优于已有方法

---

## 引用

如果本工作对你有帮助，请引用：

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

## 相关工作

- **SVFL** – Zhang et al., *Signal Processing* 2023 — 基于两分类器的特征解纠缠
- **LabObf** – He et al., 2024 — 随机软标签映射的标签混淆
- **KDk** – Arazzi et al., *Neurocomputing* 2025 — 知识蒸馏 + k-匿名
- **LADSG** – Yan et al., 2025 — 标签匿名化蒸馏 + 梯度替代
- **VMask** – Tan et al., *FCS* 2025 — 基于秘密共享的层掩码
- **MID** – Zou et al., 2023 — 互信息正则化防御

---

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。

---

## 致谢

本工作参考了 [VFLAIR](https://github.com/FLAIR-THU/VFLAIR) 纵向联邦学习攻防评估基准。
