# GOAI Protein–Ligand Dynamics

GOAI「小分子–蛋白质结合轨迹预测」初赛项目。当前主线是在作者发布的 NeuralMD 上训练一个几何、时序、概率残差校正器，重点改善 MISATO_1000 的 T3 长时域误差与配体整体漂移。

## 当前结论与诚实边界

- 已在 100 个 unseen test complexes 上复现官方 NeuralMD-ODE seed 42 基线。
- 已完成 `GNN + GRU + (μ, σ)` 残差模型、train/val 缓存、训练、断点续跑、test 评估和 Kaggle 主 Notebook。
- 已用合成缓存跑通端到端反向传播、验证集选模和训练恢复；仓库测试为 46 项。
- **新的概率残差模型尚未在 Kaggle 完成 800/100 训练和 100-test 实跑，因此当前不公布任何新模型成绩。**

本项目使用作者预训练的 NeuralMD-ODE 和 NeuralMD-SDE checkpoint。NeuralMD 第一版保持冻结；我们训练的是后接的概率时序残差模块，不把官方权重描述成自主训练成果，也不把后处理实验称为完整重训论文。

## 模型

对冻结 NeuralMD-ODE 生成的完整 T3 轨迹，训练标签为

\[
R_{t,i}=X^{\mathrm{true}}_{t,i}-X^{\mathrm{NeuralMD}}_{t,i}.
\]

共享权重的几何 GNN 编码配体邻居、蛋白 CA 邻居和原子特征；单向 GRU 学习每个原子的时间依赖。模型输出

\[
\mu_{t,i}\in\mathbb{R}^3,\qquad \sigma_{t,i}>0,
\]

并得到最佳点预测

\[
X^{\mathrm{corrected}}_{t,i}=X^{\mathrm{NeuralMD}}_{t,i}+\mu_{t,i}.
\]

概率解释为

\[
R_{t,i}\sim\mathcal N(\mu_{t,i},\sigma_{t,i}^2 I_3).
\]

均值向量由配体–配体、配体–蛋白与速度方向的标量加权组合构成，满足旋转等变和全局平移不变。`σ` 头默认与共享均值特征隔离梯度；均值由 MSE 优化，NLL 负责校准尺度，防止概率目标牺牲第一优先级的点预测 RMSE。

## 数据隔离与消融

| 阶段 | 数据 | 用途 |
| --- | ---: | --- |
| train | 800 complexes | 生成冻结 ODE 轨迹并训练残差模型 |
| val | 100 complexes | early stopping 与 checkpoint 选择 |
| test | 100 complexes | 训练完成后一次性最终比较 |

训练缓存接口只接受 `train` 和 `val`，无法表示 `test`。最终至少比较：

| 方法 | 含义 |
| --- | --- |
| `neuralmd_ode` | 官方 ODE seed 42 |
| `neuralmd_sde_seed42_single_sample` | 官方 SDE seed 42 单次随机轨迹 |
| `ode_mu` | 每帧几何 GNN 残差均值 |
| `ode_mu_sigma` | 每帧均值 + 各向同性概率尺度 |
| `ode_temporal_mu_sigma` | 几何 GNN + 单向 GRU + 概率输出 |

主成功标准依次为：降低 T3 mean/final RMSE；stability 不出现实质退化；NLL 和三维径向 coverage 作为概率校准证据。采样不会被用来“改善 RMSE”，点预测始终使用 `μ`。

## Kaggle 运行

训练主 Notebook：[`notebooks/neuralmd_probabilistic_residual.ipynb`](notebooks/neuralmd_probabilistic_residual.ipynb)

官方基线 Notebook：[`notebooks/neuralmd_misato1000.ipynb`](notebooks/neuralmd_misato1000.ipynb)

主 Notebook 的固定顺序为：

1. 校验 T4、PyG CUDA、代码 commit、MISATO_1000 和两个官方 checkpoint。
2. 生成 3 train + 3 val 的真实 T3 缓存。
3. 三个残差消融各训练 2 epochs。
4. 在 3 个 test complexes 上跑完整 smoke，只用于排错。
5. 断点扩展到 800 train / 100 val 并完整训练。
6. 在同一次 ODE test rollout 上比较三个校正器，另跑官方 SDE 基线。
7. 只整理 CSV 与 PNG；权重和缓存留在 Kaggle 工作目录。

缓存使用 `--resume` 时会逐文件校验数组、帧范围和 `residual == true - prediction`。训练使用 `latest.pth` 恢复模型、优化器、early-stopping、历史和 shuffle 随机状态；更换缓存或模型合同会拒绝续训。

## 已完成的官方 ODE 基线

| 任务 | 预测步数 | Mean RMSE (Å) | Final RMSE (Å) | Final stability (%) | Final COM error (Å) | Final Rg error (Å) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper | 99 | 4.4413 | 6.0007 | 62.81 | 5.5126 | 0.2845 |
| T1 | 10 | 1.7706 | 2.3580 | 73.78 | 2.0063 | 0.2203 |
| T2 | 20 | 2.6414 | 3.1949 | 68.69 | 2.8959 | 0.2671 |
| T3 | 80 | 4.1561 | 5.6040 | 61.68 | 5.1300 | 0.2971 |

T3 的逐帧 RMSE 从 1.2100 Å 增至 5.6040 Å；末帧 COM error 为 5.1300 Å，而 Rg error 只有 0.2971 Å。最差样本 `4EZ5` 的 T3 final RMSE 为 38.7501 Å、final COM error 为 38.5543 Å、final Rg error 为 0.0837 Å。这是第一版优先建模时间累积误差和整体漂移的直接依据。

![Official NeuralMD failure curves](results/neuralmd_failure_curves.png)

## 仓库结构

```text
notebooks/neuralmd_probabilistic_residual.ipynb  # 新模型 Kaggle 主流程
notebooks/neuralmd_misato1000.ipynb              # 已执行官方基线
scripts/cache_neuralmd_residuals.py               # 仅 train/val 缓存
scripts/train_probabilistic_residual.py            # 三种消融训练与恢复
scripts/evaluate_probabilistic_residual.py         # unseen test 公平比较
scripts/run_official_neuralmd.py                    # 官方 ODE/SDE 评估
src/temporal_residual_model.py                      # 几何 GNN + GRU + μ/σ
src/residual_training.py                            # 数据集与 loss
src/probabilistic_evaluation.py                     # NLL 与 3D coverage
results/                                            # 已验证官方基线 CSV/PNG
tests/                                              # 数学、几何、隔离与端到端合同
```

## 本地测试

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

数据、缓存、checkpoint、Word 和 PDF 不上传 GitHub；只公开代码、Notebook、真实运行 CSV 和结果图。缺失结果不会伪造。

## 来源

- [NeuralMD code](https://github.com/chao1224/NeuralMD)
- [NeuralMD paper](https://doi.org/10.1038/s41467-025-67808-z)
- [NeuralMD checkpoints](https://huggingface.co/chao1224/NeuralMD/tree/main)
- [MISATO dataset](https://github.com/t7morgen/misato-dataset)
- [MISATO paper](https://doi.org/10.1038/s43588-024-00627-2)
