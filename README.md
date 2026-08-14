# GOAI Protein–Ligand Dynamics

GOAI「小分子–蛋白质结合轨迹预测」初赛项目。当前仓库只保留一条主线：在官方 `MISATO_1000` 数据和官方 seed 42 NeuralMD-ODE checkpoint 上完成可复现评测，再根据完整测试集的逐帧失效模式设计改进。

## 当前进展

- [x] 固定 NeuralMD、torchdiffeq 与项目代码版本
- [x] 校验 `MISATO_1000` 数据合同（800 / 100 / 100 complexes）
- [x] 校验官方 seed 42 NeuralMD-ODE checkpoint
- [x] 完成 3-complex smoke run
- [x] 完成全部 100 个 unseen test complexes 评测
- [x] 公开 Notebook、汇总 CSV、坏样本 CSV 与结果图
- [ ] 从 Kaggle 工作目录补充逐帧和逐 complex 原始 CSV
- [ ] 用小型对照实验验证长时间漂移的成因

## 复现实验

主 Notebook：[`notebooks/neuralmd_misato1000.ipynb`](notebooks/neuralmd_misato1000.ipynb)

已验证环境：Kaggle Tesla T4、PyTorch 2.10.0+cu128、PyG 2.5.3、torch-scatter 2.1.2、torch-cluster 1.6.3。Notebook 会下载并校验官方数据和 checkpoint，先执行 3-complex smoke run，再评测完整测试集。

仓库中的 [`scripts/run_official_neuralmd.py`](scripts/run_official_neuralmd.py) 只为官方模型补充 checkpoint 加载、eval-only 和逐帧导出，不更改模型结构；[`scripts/build_official_neuralmd_notebook.py`](scripts/build_official_neuralmd_notebook.py) 用于重建 Notebook。

## 完整测试结果

| 任务 | 预测步数 | Mean RMSE (Å) | Final RMSE (Å) | Final stability (%) | Final COM error (Å) | Final Rg error (Å) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper | 99 | 4.4413 | 6.0007 | 62.81 | 5.5126 | 0.2845 |
| T1 | 10 | 1.7706 | 2.3580 | 73.78 | 2.0063 | 0.2203 |
| T2 | 20 | 2.6414 | 3.1949 | 68.69 | 2.8959 | 0.2671 |
| T3 | 80 | 4.1561 | 5.6040 | 61.68 | 5.1300 | 0.2971 |

T3 的逐帧 RMSE 从 1.2100 Å 增至 5.6040 Å，增长 4.63 倍；stability 从 89.26% 降至 61.68%。末帧 COM error 为 5.1300 Å，而 Rg error 只有 0.2971 Å。这首先指向长时间 rollout 的误差累积与配体整体漂移，同时伴随稳定性下降，而不是单纯的内部尺度失真。

最差样本 `4EZ5` 的 T3 final RMSE 为 38.7501 Å，其中 final COM error 为 38.5543 Å、final Rg error 为 0.0837 Å。下一步将围绕速度初始化、ODE 误差累积、条件编码和训练/推理时域差异做小型消融。

![Official NeuralMD failure curves](results/neuralmd_failure_curves.png)

## 已公开结果

- [`results/neuralmd_smoke_summary_seed42.csv`](results/neuralmd_smoke_summary_seed42.csv)：3-complex smoke run
- [`results/neuralmd_summary_seed42.csv`](results/neuralmd_summary_seed42.csv)：100-complex 完整汇总
- [`results/neuralmd_weakness_diagnosis_seed42.csv`](results/neuralmd_weakness_diagnosis_seed42.csv)：首末帧失效诊断
- [`results/neuralmd_t3_worst_complexes_seed42.csv`](results/neuralmd_t3_worst_complexes_seed42.csv)：T3 最差 15 个样本
- [`results/neuralmd_failure_curves.png`](results/neuralmd_failure_curves.png)：失效曲线

数据、checkpoint、Word 和 PDF 不上传 GitHub。逐帧 `neuralmd_frames.csv` 与逐 complex `neuralmd_complexes.csv` 尚未从 Kaggle 工作目录导出，本仓库不会伪造缺失结果。

## 仓库结构

```text
notebooks/neuralmd_misato1000.ipynb   # 已执行的 Kaggle Notebook
scripts/run_official_neuralmd.py      # 官方模型评测入口
scripts/build_official_neuralmd_notebook.py
src/neuralmd_official.py              # 数据合同与 rollout 工具
results/neuralmd_*.csv                 # 真实运行结果
results/neuralmd_failure_curves.png
tests/                                # 合同、Notebook 与公开结果校验
```

## 本地检查

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

完整 GPU 评测请在 Kaggle T4 环境运行主 Notebook。

## 来源

- [NeuralMD](https://github.com/chao1224/NeuralMD)
- [NeuralMD paper](https://doi.org/10.1038/s41467-025-67808-z)
- [MISATO dataset](https://github.com/t7morgen/misato-dataset)
- [MISATO paper](https://doi.org/10.1038/s43588-024-00627-2)
