# GOAI Protein–Ligand Dynamics

GOAI「小分子–蛋白质结合轨迹预测」初赛实验。

当前主线是：**先复现官方 NeuralMD，再根据完整逐帧结果确定改进方向**。早期的 Static、Linear、Residual MLP 与键长投影实验仅保留为轻量诊断，不作为初赛主模型。

## 数据

- 全量 MISATO `MD.hdf5`：132.8 GB
- 主实验子集：NeuralMD 作者提供的 `MISATO_1000`，官方文件大小 7,455,614,516 bytes，800 / 100 / 100 complexes
- 早期诊断子集：`MISATO_100`，80 / 10 / 10 complexes
- 数据文件不上传 GitHub

`MISATO_100` 包含 80 / 10 / 10 个训练、验证和测试复合物。下载后运行：

```bash
python -m scripts.download_misato100
python -m scripts.inspect_misato --pdb-id 3SNC
```

已验证 `3SNC` 的配体重原子轨迹形状为 `(100, 39, 3)`。

## Kaggle：先跑官方 NeuralMD

上传并运行 [`notebooks/02_official_neuralmd_misato1000.ipynb`](notebooks/02_official_neuralmd_misato1000.ipynb)：

1. Kaggle 打开 Internet，Accelerator 选择 **T4 x2**，不要选择 P100；
2. Notebook 下载并严格校验官方 `MISATO_1000` 和 seed 42 NeuralMD-ODE checkpoint；
3. 先对 3 个 unseen complexes 做 smoke run；
4. 再评测完整 100-complex test split；
5. 输出论文口径及 T1 / T2 / T3 的逐帧 CSV、坏样本 CSV 和失效曲线 PNG。

Notebook 使用官方模型、数据加载器、自定义 ODE solver 和六项官方指标。仓库中的 wrapper 只补充上游脚本缺少的 checkpoint 加载、eval-only、逐帧导出功能。

## 早期轻量实验

直接上传并运行 [`notebooks/01_misato100_end_to_end.ipynb`](notebooks/01_misato100_end_to_end.ipynb)。首次自动下载 725 MB 子集时打开 Internet；也可以先把 `MISATO_100` 作为 Kaggle Dataset 挂载。Notebook 会把 CSV、PNG 和 checkpoint 写到 `/kaggle/working/goai_results/`。

## 复现

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.download_misato100
python -m scripts.run_baselines --split test
python -m scripts.train_one_step
python -m scripts.train_multistep --bond-weight 10
python -m scripts.run_models \
  --split test \
  --checkpoint MLP=checkpoints/one_step_mlp.pt \
  --checkpoint Ours=checkpoints/multistep_geometry.pt \
  --project Ours \
  --name final
```

## 方法

Residual MLP 根据最近两步速度预测下一步速度。Ours 从同一 MLP checkpoint 出发，增加 5 步 rollout loss、键长 loss，并对输出执行 5 次键长约束投影。键由初始构型和共价半径规则推断。

## 早期诊断结果（不是 NeuralMD）

以下是 MISATO_100 测试子集（10 个 unseen complexes）上的自定义诊断结果，不是官方评分。表中报告最后一个预测时刻的 ligand RMSD：

| 任务 | Static | Linear | MLP | Ours |
| --- | ---: | ---: | ---: | ---: |
| T1: 10 → 10 | 2.0601 | 11.7462 | 2.0750 | **2.0279** |
| T2: 80 → 20 | 4.6316 | 20.1996 | **4.6023** | 4.8293 |
| T3: 20 → 80 | **6.9745** | 82.6385 | 8.0019 | 7.6985 |

最后一个预测时刻的平均 bond-length error：

| 任务 | Static | Linear | MLP | Ours |
| --- | ---: | ---: | ---: | ---: |
| T1: 10 → 10 | 0.0365 | 2.3423 | 0.1021 | **0.0008** |
| T2: 80 → 20 | 0.0371 | 5.1318 | 0.1719 | **0.0010** |
| T3: 20 → 80 | 0.0383 | 27.9960 | 0.7524 | **0.0068** |

![RMSD vs prediction horizon](results/final_rmsd_test.png)

![Bond error vs prediction horizon](results/final_bond_test.png)

这些结果只支持“显式几何约束能显著减少结构失真”，不能证明该小模型全面提高长时坐标精度。T3 中 Ours 优于普通 MLP，但仍弱于 Static。因此不再凭这组玩具结果直接设计主模型，下一步以官方 NeuralMD 的完整测试结果为依据。

## 进度

- [x] 读取单条 ligand 轨迹
- [x] 构造 T1 / T2 / T3
- [x] Static 与 Linear baseline
- [x] RMSD–horizon 曲线
- [x] Residual MLP
- [x] Multi-step + geometry
- [x] 官方 MISATO_1000 数据与 checkpoint 合同校验
- [x] NeuralMD checkpoint 加载与 eval-only wrapper
- [x] Kaggle 官方 NeuralMD smoke/full-run Notebook
- [ ] 在 Kaggle 跑完 100-complex test split 并提交原始 CSV / PNG
- [ ] 根据主导失效模式确定第一项 NeuralMD 改进

## 来源

- [MISATO dataset](https://github.com/t7morgen/misato-dataset)
- [NeuralMD](https://github.com/chao1224/NeuralMD)
- [MISATO paper](https://doi.org/10.1038/s43588-024-00627-2)
- [NeuralMD paper](https://doi.org/10.1038/s41467-025-67808-z)
