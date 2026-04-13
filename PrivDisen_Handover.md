# PrivDisen Project Handover & Roadmap

## 一、项目背景

### 研究方向
谭杨（tanyang）正在做**纵向联邦学习（Vertical Federated Learning, VFL）的标签保护**方向。

### 用户已有的工作
1. **第一篇论文（已完成）**：基于 GAN 的 soft label 转换保护标签信息
2. **第二篇论文（本项目）**：需要一个非 GAN 路线的新方法

### 本项目的核心思路：PrivDisen
**PrivDisen**（Privacy-preserving Disentangled Representation）是第二篇论文的方法。核心思想是：

> 在 VFL 的被动方引入**变分解耦模块（VDM）**，将中间嵌入分解为"任务相关表征 Z_task"和"标签敏感表征 Z_private"，只传输 Z_task 给主动方，Z_private 保留在本地。通过互信息约束提供可证明的隐私保证。

### 与已有工作的差异
唯一的直接竞品是 **SVFL**（Zhang et al., Signal Processing, 2023），它也用了特征解耦+对抗训练，但有以下局限性，是 PrivDisen 的突破点：

| SVFL 的局限 | PrivDisen 的改进 |
|------------|----------------|
| 两个分类器硬分割 | **变分推断软分离**（参数化概率分布） |
| 无理论隐私保证 | **Fano 不等式 + MI bound**，可证明攻击上界 |
| 固定保护强度 | **β 参数可调**，提供 Pareto 曲线 |
| 只防被动攻击 | 实验覆盖被动+主动+模型完成攻击 |
| 无解耦质量保证 | **HSIC 独立性约束** |

---

## 二、已完成的工作

### 2.1 研究调研（3 份报告）

| 文件 | 内容 |
|------|------|
| `VFL_Label_Protection_Research_Analysis.md` | 18 篇核心论文综述，攻击与防御全景分析 |
| `VFL_Novel_Research_Ideas_Paper2.md` | 5 个非 GAN 方向建议，已占坑方向汇总 |
| `PrivDisen_Research_Plan.md` | 表征解耦方向的完整研究方案（架构+代码+理论+实验） |
| `PrivDisen_Implementation_Plan.md` | 实施方案修订版（六个局限性评估+实验表格设计+时间线） |

### 2.2 代码实现（已全部完成 ✅）

项目路径：`~/paper/PrivDisen/`（已推送到 GitHub）

```
✅ utils/          — seed, logger, config（YAML + CLI, auto device, UTF-8 safe）
✅ data/           — 6 个数据集加载器 + N 方 VFL 特征划分 + 国内镜像下载
✅ models/         — ★ VDM + ★ ALC+GRL + Bottom/Top Model + ReconDecoder + GradPurifier
✅ losses/         — task(CE) + mi(KL) + hsic + reconstruction
✅ attacks/        — norm + direction + model_completion + embedding_extension
✅ trainers/       — VFLTrainer(vanilla) + ★ PrivDisenTrainer(完整流程)
✅ baselines/      — ★ SVFL + LabObf + KDk + MID (4 个基线方法全部实现)
✅ evaluation/     — metrics + attack_eval + visualization(t-SNE/Pareto/曲线)
✅ experiments/    — run_main + run_multi_party + run_ablation
✅ scripts/        — train.py + eval.py（跨平台 Python 脚本）+ train.sh + eval.sh
✅ README.md       — 中文版，含完整安装运行流程 + Windows 兼容
✅ requirements.txt + setup.py + .gitignore + LICENSE
```

### 2.3 Windows 兼容性修复（已完成 ✅）

- ✅ `.sh` 脚本 → 跨平台 `.py` 脚本
- ✅ YAML/文件读写全部 `encoding="utf-8"`
- ✅ `num_workers=0` Windows 默认值
- ✅ `device="auto"` 自动检测（含实际 CUDA tensor 验证）
- ✅ 所有 Python 入口文件 `sys.stdout.reconfigure(encoding="utf-8")`
- ✅ 子进程 `PYTHONIOENCODING=utf-8` 传递

### 2.4 数据下载修复（已完成 ✅）

- ✅ CIFAR: 官方源 + hf-mirror.com 多镜像，每个 URL 自动重试 2 次
- ✅ MNIST: 不再依赖 yann.lecun.com，改用 `ossci-datasets.s3.amazonaws.com`
- ✅ UCI: 带超时、重试、进度条的下载器
- ✅ `download.py`: ModelScope SDK import 兼容新旧版本

---

## 三、接下来的工作计划

### Phase 1: 实验验证（当前阶段）

| # | 任务 | 优先级 | 状态 |
|---|------|--------|------|
| 1 | 编写自动化实验脚本 `scripts/run_all_experiments.py`，自动记录训练指标并保存到 CSV/JSON | P0 | ✅ 已完成 |
| 2 | 在 GPU 上跑通 Vanilla VFL + PrivDisen（CIFAR-10, 2方），验证代码正确性 | P0 | 待执行 |
| 3 | 调试 VDM 变分推断稳定性（β KL annealing、梯度裁剪） | P0 | 待执行 |
| 4 | 跑完 6 个方法 × 3 个数据集 的主对比实验（Table 1） | P0 | 待执行 |
| 5 | 跑多方扩展实验（Table 2: 2/3/4/5 方） | P1 | 待执行 |
| 6 | 跑消融实验（Table 3: 去掉各损失项） | P1 | 待执行 |

### Phase 2: 额外实验

| # | 任务 | 对应论文内容 |
|---|------|-------------|
| 7 | Pareto 曲线实验：扫描 β ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5} | Figure: MTA vs ASR trade-off |
| 8 | 超参数敏感性实验：task_dim, private_dim, α_schedule | Appendix 表格 |
| 9 | t-SNE 可视化：Z_task 和 Z_private 的分布对比 | Figure: 解耦效果 |
| 10 | 训练曲线可视化：loss 各项变化趋势 | Figure: 收敛分析 |

### Phase 3: 论文写作

| # | 任务 | 说明 |
|---|------|------|
| 11 | 3 个定理的推导（隐私保证、效用保证、收敛性） | 证明思路在 `PrivDisen_Research_Plan.md` 中 |
| 12 | 论文正文写作（LaTeX） | Introduction → Related Work → Method → Theory → Experiments → Conclusion |
| 13 | 图表制作（matplotlib/pgfplots） | 架构图、实验表格、Pareto 曲线、t-SNE |

---

## 四、关键文件索引

| 文件路径 | 说明 |
|---------|------|
| `models/vdm.py` | ★ 变分解耦模块（核心创新） |
| `models/adversarial.py` | ★ ALC + GRL + α 调度 |
| `trainers/privdisen_trainer.py` | ★ 完整训练流程 |
| `trainers/vfl_trainer.py` | Vanilla VFL 基线 |
| `baselines/svfl.py` | SVFL 基线（Zhang et al. 2023） |
| `baselines/labobf.py` | LabObf 基线（He et al. 2024） |
| `baselines/kdk.py` | KDk 基线（Arazzi et al. 2025） |
| `baselines/mid.py` | MID 基线（Zou et al. 2023） |
| `data/datasets.py` | 数据集加载（含国内镜像下载） |
| `data/vfl_partition.py` | N 方 VFL 特征划分 |
| `configs/default.yaml` | 默认超参数 |
| `scripts/run_all_experiments.py` | ★ 自动化实验脚本（自动记录+保存结果） |

---

## 五、用户偏好

- 使用 **Windows** 开发（路径是 `E:\python_project\PrivDisen\`），Mac 是 CodeBuddy 端
- Python 3.9 + Anaconda 虚拟环境
- 需要 **中文** 文档和提示
- 偏好**国内镜像源**（pip 用清华源，数据集走阿里云 OSS / ModelScope）
- 代码需要放到 **GitHub** 上
- 实验需要 **CUDA GPU**
