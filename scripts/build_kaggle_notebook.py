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
        """# GOAI 方向二：MISATO_100 端到端可行性实验

这个 Notebook 在公开的 **MISATO_100（725 MB）** 上完成：

1. 下载并检查 80/10/10 complex 划分；
2. 运行 Static / Linear；
3. 训练共享速度 Residual MLP；
4. 加入 5 步 rollout、键长 loss 与键长投影；
5. 在 10 个未见测试 complex 上比较 T1 / T2 / T3。

> Kaggle 设置：首次自动下载需打开 **Internet**。CPU 可以完整运行；GPU 只用于加速小模型训练。数据与测试集只用于本次可行性诊断，结果不是官方榜单分数。"""
    ),
    code(
        """from pathlib import Path
import csv
import os
import random

import h5py
os.environ.setdefault("MPLCONFIGDIR", "/tmp/goai-matplotlib")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("/kaggle/working/goai_results") if Path("/kaggle/working").exists() else Path("kaggle_results")
RESULTS.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE)
print("results:", RESULTS.resolve())"""
    ),
    markdown("## 1. 下载并检查 MISATO_100"),
    code(
        """def find_or_download_data():
    local_candidates = [
        Path.cwd() / "data/MISATO_100/raw",
        Path.cwd().parent / "data/MISATO_100/raw",
    ]
    for candidate in local_candidates:
        if (candidate / "MD.hdf5").exists():
            return candidate

    kaggle_inputs = list(Path("/kaggle/input").glob("**/MISATO_100/raw/MD.hdf5")) if Path("/kaggle/input").exists() else []
    if kaggle_inputs:
        return kaggle_inputs[0].parent

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    target = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("data")
    snapshot_download(
        repo_id="chao1224/NeuralMD",
        repo_type="dataset",
        allow_patterns=["MISATO_100/raw/*"],
        local_dir=target,
    )
    return target / "MISATO_100/raw"


RAW = find_or_download_data()
print("data:", RAW)
for name in ("train", "val", "test"):
    ids = [x.strip() for x in (RAW / f"{name}_MD.txt").read_text().splitlines() if x.strip()]
    print(f"{name:>5}: {len(ids)} complexes")"""
    ),
    code(
        """def load_ligand(pdb_id):
    with h5py.File(RAW / "MD.hdf5", "r") as h5:
        group = h5[pdb_id.upper()]
        ligand_begin = int(group["molecules_begin_atom_index"][-1])
        atoms = np.asarray(group["atoms_number"][ligand_begin:], dtype=np.int64)
        trajectory = np.asarray(group["trajectory_coordinates"][:, ligand_begin:, :], dtype=np.float32)

    heavy = atoms != 1
    return trajectory[:, heavy], atoms[heavy]


example_id = [x.strip() for x in (RAW / "train_MD.txt").read_text().splitlines() if x.strip()][0]
example_trajectory, example_atoms = load_ligand(example_id)
print(example_id, "trajectory", example_trajectory.shape, "atoms", example_atoms.shape)
assert example_trajectory.shape[0] >= 100 and example_trajectory.shape[-1] == 3
assert np.isfinite(example_trajectory).all()"""
    ),
    markdown("## 2. Static / Linear baseline"),
    code(
        """TASKS = {"T1": (10, 10), "T2": (80, 20), "T3": (20, 80)}


def static_forecast(history, horizon):
    return np.repeat(history[-1][None], horizon, axis=0)


def linear_forecast(history, horizon):
    velocity = history[-1] - history[-2]
    steps = np.arange(1, horizon + 1, dtype=np.float32)[:, None, None]
    return history[-1][None] + steps * velocity[None]


def rmsd_curve(prediction, target):
    squared_distance = np.sum((prediction - target) ** 2, axis=-1)
    return np.sqrt(np.mean(squared_distance, axis=-1))


def read_ids(split):
    return [x.strip().upper() for x in (RAW / f"{split}_MD.txt").read_text().splitlines() if x.strip()]


def evaluate_baselines(split="test"):
    curves = {task: {model: [] for model in ("Static", "Linear")} for task in TASKS}
    for pdb_id in read_ids(split):
        trajectory, _ = load_ligand(pdb_id)
        for task, (history_size, horizon) in TASKS.items():
            history = trajectory[:history_size]
            target = trajectory[history_size:history_size + horizon]
            curves[task]["Static"].append(rmsd_curve(static_forecast(history, horizon), target))
            curves[task]["Linear"].append(rmsd_curve(linear_forecast(history, horizon), target))
    return curves


baseline_curves = evaluate_baselines()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for axis, (task, (_, horizon)) in zip(axes, TASKS.items()):
    for model in ("Static", "Linear"):
        mean = np.mean(baseline_curves[task][model], axis=0)
        axis.plot(np.arange(1, horizon + 1), mean, label=model, linewidth=2)
        print(task, model, f"final RMSD={mean[-1]:.4f}")
    axis.set(title=f"{task}: {TASKS[task][0]} → {horizon}", xlabel="Prediction horizon")
    axis.grid(alpha=.25)
axes[0].set_ylabel("Ligand RMSD (Å)")
axes[-1].legend(frameon=False)
fig.tight_layout()
plt.show()"""
    ),
    markdown("## 3. Residual MLP 与几何工具"),
    code(
        """class VelocityMLP(nn.Module):
    def __init__(self, velocity_scale, hidden_size=64):
        super().__init__()
        self.atom_embedding = nn.Embedding(119, 8)
        self.network = nn.Sequential(
            nn.Linear(14, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 3),
        )
        self.register_buffer("velocity_scale", torch.tensor(float(velocity_scale)))

    def predict_velocity(self, previous, current, atoms):
        scale = self.velocity_scale.clamp_min(1e-6)
        features = torch.cat([
            current / scale,
            (current - previous) / scale,
            self.atom_embedding(atoms),
        ], dim=-1)
        return current + scale * self.network(features)

    def rollout(self, history, atoms, horizon):
        previous_velocity = history[-2] - history[-3]
        current_velocity = history[-1] - history[-2]
        position = history[-1]
        output = []
        for _ in range(horizon):
            next_velocity = self.predict_velocity(previous_velocity, current_velocity, atoms)
            position = position + next_velocity
            output.append(position)
            previous_velocity, current_velocity = current_velocity, next_velocity
        return torch.stack(output)


COVALENT_RADII = {5:.84, 6:.76, 7:.71, 8:.66, 9:.57, 14:1.11, 15:1.07, 16:1.05, 17:1.02, 34:1.20, 35:1.20, 53:1.39}


def infer_bonds(coordinates, atoms, tolerance=1.25):
    radii = np.array([COVALENT_RADII.get(int(z), .77) for z in atoms])
    distance = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
    bonded = (distance > .4) & (distance < tolerance * (radii[:, None] + radii[None]))
    return np.stack(np.where(np.triu(bonded, k=1))).astype(np.int64)


def bond_error(prediction, reference, edges, squared=False):
    if edges.numel() == 0:
        return prediction.new_zeros(())
    source, target = edges
    predicted = torch.linalg.vector_norm(prediction[..., source, :] - prediction[..., target, :], dim=-1)
    expected = torch.linalg.vector_norm(reference[source] - reference[target], dim=-1)
    difference = predicted - expected
    return difference.square().mean() if squared else difference.abs().mean()


def project_bonds(coordinates, reference, edges, iterations=5):
    projected = coordinates.clone()
    if edges.numel() == 0:
        return projected
    source, target = edges
    expected = torch.linalg.vector_norm(reference[source] - reference[target], dim=-1)
    for _ in range(iterations):
        for k in range(edges.shape[1]):
            i, j = int(source[k]), int(target[k])
            vector = projected[..., j, :] - projected[..., i, :]
            length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)
            correction = .5 * (length - expected[k]) * vector / length
            projected[..., i, :] += correction
            projected[..., j, :] -= correction
    return projected"""
    ),
    markdown("## 4. 一步训练"),
    code(
        """def load_split(split):
    output = []
    for pdb_id in read_ids(split):
        positions, atoms = load_ligand(pdb_id)
        output.append((pdb_id, torch.from_numpy(positions), torch.from_numpy(atoms)))
    return output


train_trajectories = load_split("train")
previous_list, current_list, target_list, atom_list = [], [], [], []
for _, positions, atoms in train_trajectories:
    velocity = positions[1:] - positions[:-1]
    steps = len(velocity) - 2
    previous_list.append(velocity[:-2].reshape(-1, 3))
    current_list.append(velocity[1:-1].reshape(-1, 3))
    target_list.append(velocity[2:].reshape(-1, 3))
    atom_list.append(atoms.repeat(steps))

previous = torch.cat(previous_list).to(DEVICE)
current = torch.cat(current_list).to(DEVICE)
target = torch.cat(target_list).to(DEVICE)
atom_numbers = torch.cat(atom_list).to(DEVICE)
velocity_scale = float(target.std().clamp_min(1e-6))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
mlp = VelocityMLP(velocity_scale).to(DEVICE)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
BATCH_SIZE = 4096
one_step_log = []

for epoch in range(1, 31):
    order = torch.randperm(len(target), device=DEVICE)
    total = 0.0
    mlp.train()
    for batch in order.split(BATCH_SIZE):
        prediction = mlp.predict_velocity(previous[batch], current[batch], atom_numbers[batch])
        loss = nn.functional.mse_loss(prediction / velocity_scale, target[batch] / velocity_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss) * len(batch)
    mean_loss = total / len(target)
    one_step_log.append(mean_loss)
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch={epoch:02d} loss={mean_loss:.6f}")

torch.save({"model_state": mlp.state_dict(), "velocity_scale": velocity_scale}, RESULTS / "one_step_mlp.pt")"""
    ),
    markdown("## 5. 5 步 rollout + bond loss"),
    code(
        """random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
ours = VelocityMLP(velocity_scale).to(DEVICE)
ours.load_state_dict(mlp.state_dict())
optimizer = torch.optim.Adam(ours.parameters(), lr=3e-4)
HORIZON = 5
BOND_WEIGHT = 10.0

bonds = {
    pdb_id: torch.from_numpy(infer_bonds(positions[0].numpy(), atoms.numpy())).to(DEVICE)
    for pdb_id, positions, atoms in train_trajectories
}

for epoch in range(1, 26):
    order = torch.randperm(len(train_trajectories))
    rollout_total = bond_total = 0.0
    ours.train()
    for index in order.tolist():
        pdb_id, positions_cpu, atoms_cpu = train_trajectories[index]
        positions = positions_cpu.to(DEVICE)
        atoms = atoms_cpu.to(DEVICE)
        max_start = len(positions) - HORIZON - 3
        start = int(torch.randint(max_start + 1, ()).item())
        history = positions[start:start + 3]
        target_rollout = positions[start + 3:start + 3 + HORIZON]
        prediction = ours.rollout(history, atoms, HORIZON)

        rollout_loss = nn.functional.mse_loss(prediction / velocity_scale, target_rollout / velocity_scale)
        geometry_loss = bond_error(prediction, positions[0], bonds[pdb_id], squared=True) / velocity_scale ** 2
        loss = rollout_loss + BOND_WEIGHT * geometry_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ours.parameters(), 5.0)
        optimizer.step()
        rollout_total += float(rollout_loss)
        bond_total += float(geometry_loss)
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch={epoch:02d} rollout={rollout_total/len(train_trajectories):.6f} bond={bond_total/len(train_trajectories):.6f}")

torch.save({"model_state": ours.state_dict(), "velocity_scale": velocity_scale}, RESULTS / "multistep_geometry.pt")"""
    ),
    markdown("## 6. 未见测试 complexes：四种方法对比"),
    code(
        """def model_prediction(model, history, atoms, horizon, project=False):
    history_t = torch.from_numpy(history).to(DEVICE)
    atoms_t = torch.from_numpy(atoms).to(DEVICE)
    with torch.no_grad():
        prediction = model.rollout(history_t, atoms_t, horizon)
        if project:
            edges = torch.from_numpy(infer_bonds(history[0], atoms)).to(DEVICE)
            prediction = project_bonds(prediction, history_t[0], edges)
    return prediction.cpu().numpy()


models = {
    "Static": lambda history, atoms, horizon: static_forecast(history, horizon),
    "Linear": lambda history, atoms, horizon: linear_forecast(history, horizon),
    "MLP": lambda history, atoms, horizon: model_prediction(mlp, history, atoms, horizon),
    "Ours": lambda history, atoms, horizon: model_prediction(ours, history, atoms, horizon, project=True),
}
curves = {task: {name: {"rmsd": [], "bond": []} for name in models} for task in TASKS}

mlp.eval()
ours.eval()
for pdb_id in read_ids("test"):
    trajectory, atoms = load_ligand(pdb_id)
    reference = torch.from_numpy(trajectory[0])
    edges = torch.from_numpy(infer_bonds(trajectory[0], atoms))
    for task, (history_size, horizon) in TASKS.items():
        history = trajectory[:history_size]
        target_task = trajectory[history_size:history_size + horizon]
        for name, predictor in models.items():
            prediction = predictor(history, atoms, horizon)
            curves[task][name]["rmsd"].append(rmsd_curve(prediction, target_task))
            curves[task][name]["bond"].append([
                float(bond_error(torch.from_numpy(frame), reference, edges)) for frame in prediction
            ])


rows = []
for task in TASKS:
    for name in models:
        for metric in ("rmsd", "bond"):
            values = np.asarray(curves[task][name][metric])
            rows.append({
                "task": task,
                "model": name,
                "metric": metric,
                "mean_over_horizon": float(values.mean(axis=0).mean()),
                "final_mean": float(values.mean(axis=0)[-1]),
                "final_std": float(values.std(axis=0)[-1]),
                "n_complexes": len(values),
            })

with (RESULTS / "final_summary_test.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for metric in ("rmsd", "bond"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, (task, (_, horizon)) in zip(axes, TASKS.items()):
        for name in models:
            mean = np.mean(curves[task][name][metric], axis=0)
            axis.plot(np.arange(1, horizon + 1), mean, label=name, linewidth=2)
        axis.set(title=f"{task}: {TASKS[task][0]} → {horizon}", xlabel="Prediction horizon")
        axis.grid(alpha=.25)
        if metric == "bond":
            axis.set_yscale("log")
    axes[0].set_ylabel("Ligand RMSD (Å)" if metric == "rmsd" else "Bond error (Å)")
    axes[-1].legend(frameon=False)
    fig.suptitle("MISATO_100 test split (n=10)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"final_{metric}_test.png", dpi=180)
    plt.show()

for metric in ("rmsd", "bond"):
    print("\\n", metric.upper())
    for row in rows:
        if row["metric"] == metric:
            print(row["task"], f'{row["model"]:>6}', f'final={row["final_mean"]:.4f} ± {row["final_std"]:.4f}')"""
    ),
    markdown(
        """## 7. 如何解读

- 重点检查 **RMSD vs horizon**：若长预测步下快速上升，说明 rollout 不稳定。
- 再看对数纵轴的 **bond error vs horizon**：几何投影应明显降低键长失真。
- 这套小模型不保证三项 RMSD 全面超过 Static。出现负结果时应如实保留；下一步需要加入蛋白口袋环境与等变表示。
- 输出文件位于 `/kaggle/working/goai_results/`，可直接从 Kaggle 右侧 Output 下载 CSV、PNG 和 checkpoint。"""
    ),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "notebooks/01_misato100_end_to_end.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(output)


if __name__ == "__main__":
    main()
