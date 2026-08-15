"""构建 NeuralMD 概率时序残差模型的 Kaggle 主 Notebook。"""

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
        """# NeuralMD + GNN + GRU 概率残差：初赛完整训练与消融

本 Notebook 不把官方预训练权重冒充自主训练结果。第一版明确采用：

1. 冻结作者发布的 NeuralMD-ODE seed 42 checkpoint；
2. 用 train/val 的 T3 轨迹构造 `R = X_true - X_NeuralMD`；
3. 比较官方 ODE、官方 SDE 单样本、`ODE+μ`、`ODE+μ+σ`、`ODE+temporal+μ+σ`；
4. 只用 val 选择 checkpoint，最后一次性在 100 个 unseen test complexes 上比较；
5. 公开代码、Notebook、CSV 和结果图，不公开数据缓存与模型权重。

运行顺序严格是 smoke → full。Kaggle Accelerator 请选择 **T4 x2**，代码使用 `cuda:0`。"""
    ),
    markdown("## 0. 配置、GPU 与磁盘门禁"),
    code(
        """from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import torch

WORK = Path("/kaggle/working/neuralmd_probabilistic")
RESULTS = Path("/kaggle/working/neuralmd_probabilistic_results")
PUBLIC = RESULTS / "public"
for directory in (WORK, RESULTS, PUBLIC):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/neuralmd-matplotlib")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

RUN_FULL_PIPELINE = True
SMOKE_COMPLEXES = 3
SMOKE_EPOCHS = 2
FULL_EPOCHS = 20
FULL_PATIENCE = 5
VARIANTS = ("ode_mu", "ode_mu_sigma", "ode_temporal_mu_sigma")

if not torch.cuda.is_available():
    raise RuntimeError("请在 Kaggle Notebook Settings 中启用 T4 GPU。")
capability = torch.cuda.get_device_capability(0)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "capability:", capability)
if capability < (7, 0):
    raise RuntimeError("请把 P100 改为 T4 x2；当前 PyG CUDA wheel 不支持 sm_60。")

free_gib = shutil.disk_usage("/kaggle/working").free / 1024**3
print(f"可用空间: {free_gib:.2f} GiB")
if free_gib < 10:
    raise RuntimeError("可用空间不足 10 GiB，不能安全处理 MISATO_1000。")"""
    ),
    markdown("## 1. 安装与 Kaggle PyTorch 匹配的 NeuralMD 依赖"),
    code(
        """def pip_install(*packages, extra_args=()):
    command = [sys.executable, "-m", "pip", "install", "-q", *packages, *extra_args]
    print(" ".join(command))
    subprocess.check_call(command)


torch_parts = torch.__version__.split("+")[0].split(".")
pyg_torch = f"{torch_parts[0]}.{torch_parts[1]}.0"
cuda_tag = "cu" + torch.version.cuda.replace(".", "")
pyg_wheels = f"https://data.pyg.org/whl/torch-{pyg_torch}+{cuda_tag}.html"

pip_install("huggingface_hub", "h5py", "pandas", "matplotlib", "tqdm", "torch-ema")
pip_install("torch_geometric==2.5.3", extra_args=("--force-reinstall", "--no-deps"))
pip_install(
    "torch_scatter==2.1.2",
    "torch_cluster==1.6.3",
    extra_args=("--no-index", "--force-reinstall", "--no-deps", "-f", pyg_wheels),
)

probe = r'''
import torch
import torch_geometric
from torch_geometric.nn import radius_graph
assert torch_geometric.__version__ == "2.5.3"
pos = torch.tensor([[0., 0., 0.], [0.5, 0., 0.], [3., 0., 0.]], device="cuda:0")
batch = torch.zeros(3, dtype=torch.long, device="cuda:0")
edges = radius_graph(pos, r=1.0, batch=batch, max_num_neighbors=8)
torch.cuda.synchronize()
assert edges.shape[1] == 2, edges.shape
print("PyG CUDA probe: OK")
'''
subprocess.check_call([sys.executable, "-c", probe])"""
    ),
    markdown("## 2. 固定 GOAI、NeuralMD 与 torchdiffeq 版本"),
    code(
        """GOAI = WORK / "goai-protein-ligand-dynamics"
OFFICIAL = WORK / "NeuralMD"
TORCHDIFFEQ = WORK / "torchdiffeq"
GOAI_COMMIT = "85e0b112f0479713cdc11100d71419baaca6c6a5"
OFFICIAL_COMMIT = "a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe"
TORCHDIFFEQ_COMMIT = "3d7c7ec8c534a9b18b8b7c7d1fea0c235e6468d0"


def run_command(command, cwd=None):
    print(" ".join(map(str, command)))
    subprocess.check_call(command, cwd=cwd)


def checkout_repo(url, destination, commit):
    if destination.exists() and not (destination / ".git").is_dir():
        shutil.rmtree(destination)
    if not destination.exists():
        run_command(["git", "clone", "--filter=blob:none", "--no-checkout", url, destination])
    run_command(["git", "fetch", "--depth", "1", "origin", commit], cwd=destination)
    run_command(["git", "checkout", "--detach", "--force", "FETCH_HEAD"], cwd=destination)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
    if revision != commit:
        raise RuntimeError(f"{destination.name} 版本错误: {revision} != {commit}")
    print(destination.name, revision)


checkout_repo("https://github.com/ecjtusyy/goai-protein-ligand-dynamics.git", GOAI, GOAI_COMMIT)
checkout_repo("https://github.com/chao1224/NeuralMD.git", OFFICIAL, OFFICIAL_COMMIT)
checkout_repo("https://github.com/chao1224/torchdiffeq.git", TORCHDIFFEQ, TORCHDIFFEQ_COMMIT)
pip_install("-e", str(TORCHDIFFEQ))
pip_install("-e", str(OFFICIAL))
subprocess.check_call([sys.executable, "-m", "pytest", "-q"], cwd=GOAI)"""
    ),
    markdown("## 3. 获取并严格校验 MISATO_1000 与官方 ODE/SDE checkpoints"),
    code(
        """from huggingface_hub import hf_hub_download, snapshot_download

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
            raise FileNotFoundError(source)
        destination = target_raw / name
        if destination.is_symlink() and destination.resolve() != source.resolve():
            destination.unlink()
        if not destination.exists():
            destination.symlink_to(source)
else:
    snapshot_download(
        repo_id="chao1224/NeuralMD",
        repo_type="dataset",
        allow_patterns=["MISATO_1000/raw/*"],
        local_dir=DATA_PARENT,
    )

sys.path.insert(0, str(GOAI))
from src.neuralmd_official import verify_misato1000

print(verify_misato1000(DATASET, strict_size=True))

# 上游 PyG processed 文件与运行时版本绑定，只删除三个明确的可重建缓存。
processed = DATASET / "processed_semi_flexible"
marker = processed / ".goai_runtime"
contract = f"pyg=2.5.3\\nofficial={OFFICIAL_COMMIT}\\n"
if not marker.is_file() or marker.read_text() != contract:
    for split in ("train", "val", "test"):
        cached = processed / f"geometric_data_processed_{split}.pt"
        if cached.exists():
            cached.unlink()
    processed.mkdir(parents=True, exist_ok=True)
    marker.write_text(contract)

ODE_CHECKPOINT = Path(hf_hub_download(
    repo_id="chao1224/NeuralMD",
    filename="NeuralMD_ODE/MISATO_1000_seed_42/model.pth",
    local_dir=WORK / "checkpoints",
))
if ODE_CHECKPOINT.stat().st_size != 8_955_570:
    raise RuntimeError("官方 NeuralMD-ODE checkpoint 不完整。")
SDE_CHECKPOINT = Path(hf_hub_download(
    repo_id="chao1224/NeuralMD",
    filename="NeuralMD_SDE/MISATO_1000_seed_42/model.pth",
    local_dir=WORK / "checkpoints",
))
if SDE_CHECKPOINT.stat().st_size != 8_954_730:
    raise RuntimeError("官方 NeuralMD-SDE checkpoint 不完整。")
print("ODE checkpoint:", ODE_CHECKPOINT)
print("SDE checkpoint:", SDE_CHECKPOINT)"""
    ),
    markdown("## 4. 统一命令：缓存、训练、unseen test 评估"),
    code(
        """CACHE_ROOT = RESULTS / "residual_cache_t3"
SMOKE_TRAIN_ROOT = RESULTS / "smoke_training"
FULL_TRAIN_ROOT = RESULTS / "full_training"
SMOKE_EVAL = RESULTS / "smoke_evaluation"
FULL_EVAL = RESULTS / "full_evaluation"
SMOKE_SDE = RESULTS / "smoke_sde"
FULL_SDE = RESULTS / "full_sde"


def cache_split(split, limit=None):
    command = [
        sys.executable, "-m", "scripts.cache_neuralmd_residuals",
        "--official-repo", str(OFFICIAL),
        "--dataset-dir", str(DATASET),
        "--checkpoint", str(ODE_CHECKPOINT),
        "--output-dir", str(CACHE_ROOT),
        "--split", split,
        "--task", "T3",
        "--device", "cuda:0",
        "--seed", "42",
        "--resume",
    ]
    if limit is not None:
        command.extend(["--limit-complexes", str(limit)])
    run_command(command, cwd=GOAI)


def train_variant(variant, root, epochs, patience, hidden_dim, frame_chunk_size=16):
    output = root / variant
    checkpoint = output / "best_model.pth"
    latest_path = output / "latest.pth"

    # Notebook 重跑时，完整训练不再进入无意义的 resume 流程。
    if checkpoint.is_file() and latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        saved_model = latest.get("model_config", {})
        same_contract = (
            latest.get("variant") == variant
            and saved_model.get("hidden_dim") == hidden_dim
            and saved_model.get("rbf_channels") == 16
            and saved_model.get("frame_chunk_size") == frame_chunk_size
        )
        if same_contract and int(latest.get("epoch", 0)) >= epochs:
            print(f"[REUSE] {variant}: 已完成 {latest['epoch']} epochs")
            return checkpoint

    command = [
        sys.executable, "-m", "scripts.train_probabilistic_residual",
        "--cache-root", str(CACHE_ROOT),
        "--output-dir", str(output),
        "--variant", variant,
        "--device", "cuda:0",
        "--seed", "42",
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--hidden-dim", str(hidden_dim),
        "--rbf-channels", "16",
        "--frame-chunk-size", str(frame_chunk_size),
    ]
    if latest_path.is_file():
        command.append("--resume")
    run_command(command, cwd=GOAI)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"训练结束但缺少 best checkpoint: {checkpoint}")
    return checkpoint


def evaluate_checkpoints(checkpoints, output, limit=None):
    command = [
        sys.executable, "-m", "scripts.evaluate_probabilistic_residual",
        "--official-repo", str(OFFICIAL),
        "--dataset-dir", str(DATASET),
        "--ode-checkpoint", str(ODE_CHECKPOINT),
        "--output-dir", str(output),
        "--device", "cuda:0",
        "--seed", "42",
        "--overwrite",
    ]
    for checkpoint in checkpoints:
        command.extend(["--residual-checkpoint", str(checkpoint)])
    if limit is not None:
        command.extend(["--limit-complexes", str(limit)])
    run_command(command, cwd=GOAI)


def evaluate_sde(output, limit=None):
    command = [
        sys.executable, "-m", "scripts.run_official_neuralmd",
        "--official-repo", str(OFFICIAL),
        "--dataset-dir", str(DATASET),
        "--checkpoint", str(SDE_CHECKPOINT),
        "--output-dir", str(output),
        "--dynamics", "sde",
        "--tasks", "T3",
        "--device", "cuda:0",
        "--seed", "42",
    ]
    if limit is not None:
        command.extend(["--limit-complexes", str(limit)])
    run_command(command, cwd=GOAI)


def combined_frames(residual_output, sde_output):
    residual = pd.read_csv(residual_output / "comparison_frames.csv")
    sde = pd.read_csv(sde_output / "neuralmd_frames.csv")
    sde.insert(0, "method", "neuralmd_sde_seed42_single_sample")
    return pd.concat([residual, sde], ignore_index=True)


def combined_summary(frames):
    rows = []
    for method, group in frames.groupby("method", sort=False):
        final = group[group["step"] == group["step"].max()]
        rows.append({
            "method": method,
            "complexes": group["pdb_id"].nunique(),
            "mean_rmse": group["rmse"].mean(),
            "final_rmse": final["rmse"].mean(),
            "mean_stability": group["stability"].mean(),
            "final_stability": final["stability"].mean(),
            "final_com_error": final["com_error"].mean(),
            "final_rg_error": final["rg_error"].mean(),
        })
    return pd.DataFrame(rows)"""
    ),
    markdown("## 5. Smoke A：只生成 3 train + 3 val 的真实 T3 残差"),
    code(
        """smoke_cache_started = time.perf_counter()
cache_split("train", limit=SMOKE_COMPLEXES)
cache_split("val", limit=SMOKE_COMPLEXES)
print(f"smoke cache: {time.perf_counter() - smoke_cache_started:.1f}s")

for split in ("train", "val"):
    manifest = __import__("json").loads((CACHE_ROOT / split / "manifest.json").read_text())
    assert manifest["split"] == split
    assert manifest["task"] == "T3"
    assert manifest["complexes"] == SMOKE_COMPLEXES
    print(split, manifest["complexes"], "complexes")"""
    ),
    markdown("## 6. Smoke B：三种消融各训练 2 epochs"),
    code(
        """smoke_checkpoints = []
for variant in VARIANTS:
    checkpoint = train_variant(
        variant,
        SMOKE_TRAIN_ROOT,
        epochs=SMOKE_EPOCHS,
        patience=SMOKE_EPOCHS,
        hidden_dim=32,
    )
    smoke_checkpoints.append(checkpoint)
    print(variant, checkpoint)

import numpy as np
import pandas as pd

nll_columns = ["train_nll", "val_nll"]
for variant in VARIANTS:
    history = pd.read_csv(SMOKE_TRAIN_ROOT / variant / "history.csv")
    display(history)
    assert len(history) == SMOKE_EPOCHS

    # epoch、耗时、MSE 与 RMSE 对所有模型都必须是有限值。
    regular_metrics = history.drop(columns=nll_columns).select_dtypes(include="number")
    assert np.isfinite(regular_metrics.to_numpy()).all(), f"{variant} 常规指标含 NaN/Inf"

    if variant == "ode_mu":
        # 确定性消融没有 sigma 头，因此 NLL 按合同为空。
        assert history[nll_columns].isna().all().all(), "ode_mu 不应生成 NLL"
    else:
        # 两个概率消融必须真正给出有限的 NLL。
        nll_values = history[nll_columns].to_numpy(dtype=float)
        assert np.isfinite(nll_values).all(), f"{variant} NLL 含 NaN/Inf"

    print(f"[OK] {variant} history contract")"""
    ),
    markdown("## 7. Smoke C：同一次 ODE rollout 比较三种校正器"),
    code(
        """evaluate_checkpoints(smoke_checkpoints, SMOKE_EVAL, limit=SMOKE_COMPLEXES)
evaluate_sde(SMOKE_SDE, limit=SMOKE_COMPLEXES)
smoke_frames = combined_frames(SMOKE_EVAL, SMOKE_SDE)
smoke_summary = combined_summary(smoke_frames)
display(smoke_summary[[
    "method", "complexes", "mean_rmse", "final_rmse",
    "mean_stability", "final_stability", "final_com_error", "final_rg_error"
]])
assert set(smoke_summary["method"]) == {
    "neuralmd_ode", "neuralmd_sde_seed42_single_sample", *VARIANTS
}
assert smoke_summary.select_dtypes("number").notna().all().all()
print("Smoke 全链路通过；3-complex 结果只用于排错，不作为初赛结论。")"""
    ),
    markdown(
        """## 8. Full：扩展到 800 train / 100 val，训练后只评估一次 100 test

该单元默认执行完整流程。缓存和训练都支持断点续跑。若一次 Kaggle session 时间不够，重新运行 Notebook 会校验并跳过已完成缓存，并从最后一个完整 epoch 继续。"""
    ),
    code(
        """full_checkpoints = []
if RUN_FULL_PIPELINE:
    cache_split("train")
    cache_split("val")
    for split, expected in (("train", 800), ("val", 100)):
        manifest = __import__("json").loads((CACHE_ROOT / split / "manifest.json").read_text())
        assert manifest["complexes"] == expected, manifest

    for variant in VARIANTS:
        full_checkpoints.append(train_variant(
            variant,
            FULL_TRAIN_ROOT,
            epochs=FULL_EPOCHS,
            patience=FULL_PATIENCE,
            hidden_dim=64,
            frame_chunk_size=4,
        ))
    evaluate_checkpoints(full_checkpoints, FULL_EVAL)
    evaluate_sde(FULL_SDE)
else:
    print("RUN_FULL_PIPELINE=False：当前只保留 smoke 结果。")"""
    ),
    markdown("## 9. 结果图、消融表与公开文件门禁"),
    code(
        """import matplotlib.pyplot as plt
import numpy as np

using_full = (FULL_EVAL / "comparison_frames.csv").is_file() and (FULL_SDE / "neuralmd_frames.csv").is_file()
analysis_dir = FULL_EVAL if using_full else SMOKE_EVAL
analysis_sde = FULL_SDE if using_full else SMOKE_SDE
frames = combined_frames(analysis_dir, analysis_sde)
summary = combined_summary(frames)
frames.to_csv(PUBLIC / "probabilistic_residual_frames.csv", index=False)
summary.to_csv(PUBLIC / "probabilistic_residual_comparison.csv", index=False)

calibration_path = analysis_dir / "calibration_summary.csv"
if calibration_path.is_file():
    calibration = pd.read_csv(calibration_path)
    calibration.to_csv(PUBLIC / "probabilistic_residual_calibration.csv", index=False)
    display(calibration)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for method, group in frames.groupby("method", sort=False):
    curve = group.groupby("step", as_index=False).mean(numeric_only=True)
    axes[0].plot(curve["step"], curve["rmse"], label=method)
    axes[1].plot(curve["step"], curve["stability"], label=method)
    axes[2].plot(curve["step"], curve["com_error"], label=method)
for axis, title, ylabel in zip(
    axes,
    ("T3 coordinate error", "T3 stability", "T3 COM drift"),
    ("Mean atom distance (Å)", "Stability (%)", "COM error (Å)"),
):
    axis.set(xlabel="Rollout step", ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
figure_path = PUBLIC / "probabilistic_residual_t3_curves.png"
fig.savefig(figure_path, dpi=180, bbox_inches="tight")
plt.show()

training_root = FULL_TRAIN_ROOT if full_checkpoints else SMOKE_TRAIN_ROOT
for variant in VARIANTS:
    source = training_root / variant / "history.csv"
    if source.is_file():
        shutil.copy2(source, PUBLIC / f"training_history_{variant}.csv")

for forbidden in ("*.pth", "*.pt", "*.npz", "*.docx", "*.pdf"):
    matches = list(PUBLIC.rglob(forbidden))
    if matches:
        raise RuntimeError(f"公开目录含禁止文件: {matches}")

display(summary[[
    "method", "complexes", "mean_rmse", "final_rmse",
    "mean_stability", "final_stability", "final_com_error", "final_rg_error"
]])
print("公开文件:")
for path in sorted(PUBLIC.iterdir()):
    print("-", path.name, path.stat().st_size, "bytes")
archive = shutil.make_archive(str(RESULTS / "github_public_results"), "zip", PUBLIC)
print("可下载公开结果包:", archive)"""
    ),
    markdown(
        """## 结论填写规则

- 只有完整 100-test 表可以写进初赛结论；3-test smoke 不能宣称提升。
- `ODE+μ` 优于 ODE：说明残差均值校正有效。
- `ODE+μ+σ` 的 RMSE 不应被采样改善；σ 用 NLL/coverage 评价。
- temporal 版本若改善 final RMSE/COM 且 stability 不明显下降，才支持“时间相关性改善长期稳定性”。
- 如果 full 结果没有提升，也如实保留 CSV；下一轮先诊断，不伪造结果。"""
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


def main() -> None:
    output = Path(__file__).parents[1] / "notebooks/neuralmd_probabilistic_residual.ipynb"
    output.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=1) + "\n")
    print(output)


if __name__ == "__main__":
    main()
