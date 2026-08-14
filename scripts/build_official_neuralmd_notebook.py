"""Build the standalone Kaggle notebook for the official NeuralMD baseline."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """# NeuralMD 官方基线：MISATO_1000 实跑与不足诊断

这不是玩具模型。Notebook 直接使用 NeuralMD 作者发布的：

- `MISATO_1000`（800/100/100 complexes，约 7.46 GB）；
- NeuralMD-ODE 官方代码与自定义 `torchdiffeq`；
- `MISATO_1000_seed_42/model.pth` 官方权重；
- MAE、RMSE、Matching、Stability、Ligand collision、Binding collision 官方指标。

执行顺序：3 个测试 complex 的 smoke run → 100 个 unseen test complexes 完整评测 → T1/T2/T3 逐帧失效分析。

> Kaggle 设置：打开 Internet，Accelerator 选择 **T4 x2**（代码只用 GPU 0）。不要选 P100；当前 Kaggle PyTorch 已不包含 P100 的 `sm_60` 内核。首次运行需要下载约 7.46 GB。"""
    ),
    markdown("## 0. GPU 与运行目录检查"),
    code(
        """from pathlib import Path
import os
import shutil
import subprocess
import sys

import torch

WORK = Path("/kaggle/working/neuralmd_official")
RESULTS = Path("/kaggle/working/neuralmd_official_results")
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuralmd-matplotlib")

if not torch.cuda.is_available():
    raise RuntimeError("请在 Kaggle Notebook Settings 中启用 T4 GPU。")

capability = torch.cuda.get_device_capability(0)
print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "compute capability:", capability)
if capability < (7, 0):
    raise RuntimeError(
        "检测到 P100/sm_60。请把 Kaggle Accelerator 改为 T4 x2 后重新运行；"
        "这样可避免 no kernel image is available。"
    )

free_gib = shutil.disk_usage("/kaggle/working").free / 1024**3
print(f"Kaggle working 可用空间: {free_gib:.2f} GiB")
if free_gib < 9:
    raise RuntimeError("/kaggle/working 可用空间不足 9 GiB，无法安全下载和处理 MISATO_1000。")"""
    ),
    markdown("## 1. 安装与当前 PyTorch 匹配的 PyG 依赖"),
    code(
        """def pip_install(*packages, extra_args=()):
    command = [sys.executable, "-m", "pip", "install", "-q", *packages, *extra_args]
    print(" ".join(command))
    subprocess.check_call(command)


torch_parts = torch.__version__.split("+")[0].split(".")
pyg_torch = f"{torch_parts[0]}.{torch_parts[1]}.0"
cuda_tag = "cu" + torch.version.cuda.replace(".", "")
pyg_wheels = f"https://data.pyg.org/whl/torch-{pyg_torch}+{cuda_tag}.html"

# NeuralMD 官方环境使用 PyG 2.5；固定版本，避免新版 MessagePassing 行为漂移。
pip_install("huggingface_hub", "h5py", "pandas", "tqdm", "torch-ema", "torch_geometric==2.5.3")
pip_install(
    "torch_scatter==2.1.2",
    "torch_cluster==1.6.3",
    extra_args=("--no-index", "--force-reinstall", "-f", pyg_wheels),
)

# 用独立进程验证，避免当前 Notebook 内核缓存旧版 PyG 模块。
probe_code = r'''
import torch
import torch_cluster
import torch_geometric
import torch_scatter
from torch_geometric.nn import radius_graph

print("PyG:", torch_geometric.__version__)
print("torch_scatter:", torch_scatter.__version__)
print("torch_cluster:", torch_cluster.__version__)
if torch_geometric.__version__ != "2.5.3":
    raise RuntimeError(f"PyG 版本错误: {torch_geometric.__version__}")

probe_pos = torch.tensor(
    [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
    dtype=torch.float32,
    device="cuda:0",
)
probe_batch = torch.zeros(3, dtype=torch.long, device="cuda:0")
probe_edges = radius_graph(probe_pos, r=1.0, batch=probe_batch, max_num_neighbors=8)
torch.cuda.synchronize()
if probe_edges.shape[1] != 2:
    raise RuntimeError(f"torch_cluster CUDA 探针失败: edge shape={tuple(probe_edges.shape)}")
print("PyG CUDA probe: OK")
'''
subprocess.check_call([sys.executable, "-c", probe_code])"""
    ),
    markdown("## 2. 固定代码版本：GOAI evaluator、NeuralMD、torchdiffeq"),
    code(
        """GOAI = WORK / "goai-protein-ligand-dynamics"
OFFICIAL = WORK / "NeuralMD"
TORCHDIFFEQ = WORK / "torchdiffeq"
GOAI_COMMIT = "233a492e4eeacb4c0ae4c86fb2e4745597cb7f47"
OFFICIAL_COMMIT = "a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe"
TORCHDIFFEQ_COMMIT = "3d7c7ec8c534a9b18b8b7c7d1fea0c235e6468d0"


def run_command(command, cwd=None):
    print(" ".join(map(str, command)))
    subprocess.check_call(command, cwd=cwd)


def checkout_repo(url, destination, commit):
    # 只清理上次克隆失败留下的空壳目录，不碰数据和结果目录。
    if destination.exists() and not (destination / ".git").is_dir():
        shutil.rmtree(destination)

    if not destination.exists():
        run_command(["git", "clone", "--filter=blob:none", "--no-checkout", url, destination])

    run_command(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
    run_command(["git", "checkout", "--detach", "--force", "FETCH_HEAD"], cwd=destination)

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    if revision != commit:
        raise RuntimeError(f"{destination.name} 版本错误: {revision} != {commit}")
    print(f"{destination.name}: {revision}")


checkout_repo("https://github.com/ecjtusyy/goai-protein-ligand-dynamics.git", GOAI, GOAI_COMMIT)
checkout_repo("https://github.com/chao1224/NeuralMD.git", OFFICIAL, OFFICIAL_COMMIT)
checkout_repo("https://github.com/chao1224/torchdiffeq.git", TORCHDIFFEQ, TORCHDIFFEQ_COMMIT)
pip_install("-e", str(TORCHDIFFEQ))
pip_install("-e", str(OFFICIAL))"""
    ),
    markdown("## 3. 获取并严格校验官方 MISATO_1000"),
    code(
        """from huggingface_hub import snapshot_download

DATA_PARENT = WORK / "data"
DATASET = DATA_PARENT / "MISATO_1000"
mounted = list(Path("/kaggle/input").glob("**/MISATO_1000/raw/MD.hdf5"))

if mounted:
    source_raw = mounted[0].parent
    target_raw = DATASET / "raw"
    target_raw.mkdir(parents=True, exist_ok=True)
    for name in ("MD.hdf5", "train_MD.txt", "val_MD.txt", "test_MD.txt"):
        source = source_raw / name
        if not source.is_file():
            raise FileNotFoundError(f"Kaggle 输入数据不完整: {source}")
        destination = target_raw / name
        if destination.is_symlink() and destination.resolve() != source.resolve():
            destination.unlink()
        if not destination.exists():
            destination.symlink_to(source)
    print("使用 Kaggle 已挂载数据:", source_raw)
else:
    snapshot_download(
        repo_id="chao1224/NeuralMD",
        repo_type="dataset",
        allow_patterns=["MISATO_1000/raw/*"],
        local_dir=DATA_PARENT,
    )
    print("已从 Hugging Face 下载 MISATO_1000")

sys.path.insert(0, str(GOAI))
from src.neuralmd_official import verify_misato1000

dataset_metadata = verify_misato1000(DATASET, strict_size=True)
print(dataset_metadata)

# 原始 7.46 GB HDF5 保留；只在运行时版本变化时重建 PyG 测试缓存。
cache_dir = DATASET / "processed_semi_flexible"
cache_marker = cache_dir / ".goai_runtime"
cache_contract = f"pyg=2.5.3\\nofficial={OFFICIAL_COMMIT}\\n"
old_contract = cache_marker.read_text() if cache_marker.is_file() else None
if old_contract != cache_contract:
    cached_test = cache_dir / "geometric_data_processed_test.pt"
    if cached_test.exists():
        cached_test.unlink()
        print("已删除旧版 PyG 生成的测试缓存:", cached_test)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_marker.write_text(cache_contract)
print("PyG dataset cache contract: OK")"""
    ),
    markdown("## 4. 下载 seed 42 官方 NeuralMD-ODE 权重"),
    code(
        """from huggingface_hub import hf_hub_download

CHECKPOINT = Path(
    hf_hub_download(
        repo_id="chao1224/NeuralMD",
        filename="NeuralMD_ODE/MISATO_1000_seed_42/model.pth",
        local_dir=WORK / "checkpoints",
    )
)
HYPERPARAMETERS = Path(
    hf_hub_download(
        repo_id="chao1224/NeuralMD",
        filename="NeuralMD_ODE/MISATO_1000_seed_42/hyperparameter.txt",
        local_dir=WORK / "checkpoints",
    )
)
print("checkpoint:", CHECKPOINT, CHECKPOINT.stat().st_size, "bytes")
hyperparameters_text = HYPERPARAMETERS.read_text()
print(hyperparameters_text)

if CHECKPOINT.stat().st_size != 8_955_570:
    raise RuntimeError("官方 checkpoint 大小不正确，请删除损坏文件后重新运行本格。")

required_hyperparameters = (
    "NeuralMD_step_size",
    "NeuralMD_scaling",
    "NeuralMD_velocity_refined_value_coefficient",
)
missing_hyperparameters = [
    name for name in required_hyperparameters if name not in hyperparameters_text
]
if missing_hyperparameters:
    raise RuntimeError(f"官方超参数文件缺少字段: {missing_hyperparameters}")"""
    ),
    markdown("## 5. 真实 smoke run：3 个 unseen test complexes"),
    code(
        """SMOKE = RESULTS / "smoke_seed42"


def run_evaluation(output_dir, limit=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m", "scripts.run_official_neuralmd",
        "--official-repo", str(OFFICIAL),
        "--dataset-dir", str(DATASET),
        "--checkpoint", str(CHECKPOINT),
        "--output-dir", str(output_dir),
        "--device", "cuda:0",
        "--seed", "42",
        "--tasks", "paper", "T1", "T2", "T3",
    ]
    if limit is not None:
        command.extend(["--limit-complexes", str(limit)])

    print("[NeuralMD]", " ".join(command))
    subprocess.check_call(command, cwd=str(GOAI))

    summary_path = output_dir / "neuralmd_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"评估结束但未生成汇总文件: {summary_path}")
    return summary_path


smoke_summary = run_evaluation(SMOKE, limit=3)
print("\\n[NeuralMD] 3-complex smoke test 完成")
print(smoke_summary.read_text())"""
    ),
    markdown(
        """## 6. 完整评测：100 个 unseen test complexes

Smoke run 成功后，本单元默认继续跑完整测试集。若 Kaggle 会话时间不足，可先把 `RUN_FULL_TEST` 改为 `False`；但初赛正式结论必须来自完整 100 个测试 complex。"""
    ),
    code(
        """RUN_FULL_TEST = True
FULL = RESULTS / "full_seed42"

if RUN_FULL_TEST:
    full_summary = run_evaluation(FULL)
    print("\\n[NeuralMD] 100-complex 完整评测完成")
    print(full_summary.read_text())
else:
    print("完整评测已跳过；当前只有 3-complex smoke 结果。")"""
    ),
    markdown("## 7. 用逐帧指标定位 NeuralMD 的不足"),
    code(
        """import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

analysis_dir = FULL if (FULL / "neuralmd_frames.csv").exists() else SMOKE
frames = pd.read_csv(analysis_dir / "neuralmd_frames.csv")
complexes = pd.read_csv(analysis_dir / "neuralmd_complexes.csv")
summary = pd.read_csv(analysis_dir / "neuralmd_summary.csv")
display(summary)

diagnosis = []
for task, group in frames.groupby("task", sort=False):
    curve = group.groupby("step", as_index=False).mean(numeric_only=True)
    first = curve.iloc[0]
    final = curve.iloc[-1]
    diagnosis.append({
        "task": task,
        "complexes": group["pdb_id"].nunique(),
        "first_rmse": first["rmse"],
        "final_rmse": final["rmse"],
        "rmse_growth_ratio": final["rmse"] / max(first["rmse"], 1e-12),
        "first_stability": first["stability"],
        "final_stability": final["stability"],
        "stability_drop": first["stability"] - final["stability"],
        "final_com_error": final["com_error"],
        "final_rg_error": final["rg_error"],
        "final_ligand_collision": final["ligand_collision"],
        "final_binding_collision": final["binding_collision"],
    })

diagnosis = pd.DataFrame(diagnosis)
diagnosis.to_csv(RESULTS / "weakness_diagnosis.csv", index=False)
display(diagnosis)"""
    ),
    code(
        """metrics = [
    ("rmse", "Coordinate RMSE (Å)"),
    ("stability", "Stability (%)"),
    ("com_error", "Center-of-mass error (Å)"),
    ("rg_error", "Radius-of-gyration error (Å)"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for axis, (metric, label) in zip(axes.ravel(), metrics):
    for task, group in frames.groupby("task", sort=False):
        curve = group.groupby("step")[metric].mean()
        axis.plot(curve.index, curve.values, label=task, linewidth=2)
    axis.set(xlabel="Rollout step", ylabel=label, title=label)
    axis.grid(alpha=0.25)
axes[0, 0].legend(frameon=False)
fig.suptitle("Official NeuralMD-ODE on MISATO_1000 unseen test complexes")
fig.tight_layout()
curve_path = RESULTS / "neuralmd_failure_curves.png"
fig.savefig(curve_path, dpi=180, bbox_inches="tight")
plt.show()

t3_worst = (
    complexes.query("task == 'T3'")
    .sort_values("mean_rmse", ascending=False)
    .head(15)
)
t3_worst.to_csv(RESULTS / "neuralmd_t3_worst_complexes.csv", index=False)
display(t3_worst[[
    "pdb_id", "ligand_atoms", "protein_residues", "mean_rmse",
    "final_rmse", "final_stability", "final_com_error", "final_rg_error"
]])"""
    ),
    markdown(
        """## 如何读结果（先诊断，不抢跑改模型）

- `RMSE` 随 rollout 快速增大：优先研究误差累积和训练/推理 horizon 不一致。
- `Stability` 明显下降、`rg_error` 增大：优先研究分子内部几何约束。
- `com_error` 主导而 `rg_error` 较小：优先研究蛋白–配体相互作用与整体漂移。
- ligand/protein 规模与坏样本相关：优先研究固定 cutoff、局部环境容量或显存友好的多尺度表示。
- 若只有少数 complex 发生突变：检查论文已指出的 out-of-distribution sudden positional changes，再决定是否做不确定性/异常运动分支。

先保留这些原始 CSV 和 PNG。下一步改进必须针对完整测试集上实际出现的主导失效模式，而不是预设“加一个模块就会更好”。"""
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


if __name__ == "__main__":
    output = Path("notebooks/02_official_neuralmd_misato1000.ipynb")
    output.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=1) + "\n")
    print(output)
