import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/goai-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.run_baselines import TASKS, read_ids, write_csv
from src.baselines import linear_forecast, static_forecast
from src.geometry import bond_length_error, infer_bonds, project_bond_lengths
from src.metrics import rmsd_curve
from src.misato import load_ligand_trajectory
from src.model import VelocityMLP


def load_model(path: Path) -> VelocityMLP:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = VelocityMLP(checkpoint["velocity_scale"], checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--name", default="models")
    parser.add_argument("--project", action="append", default=[])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw = root / "data/MISATO_100/raw"
    results = root / "results"
    results.mkdir(exist_ok=True)
    pdb_ids = read_ids(raw / f"{args.split}_MD.txt")

    predictors = {
        "Static": lambda history, atoms, horizon: static_forecast(history, horizon),
        "Linear": lambda history, atoms, horizon: linear_forecast(history, horizon),
    }
    for item in args.checkpoint:
        label, relative_path = item.split("=", maxsplit=1)
        model = load_model(root / relative_path)

        def predict(history, atoms, horizon, model=model):
            with torch.no_grad():
                return model.rollout(
                    torch.from_numpy(history),
                    torch.from_numpy(atoms),
                    horizon,
                ).numpy()

        predictors[label] = predict

    curves = {
        task: {model: {"rmsd": [], "bond": []} for model in predictors}
        for task in TASKS
    }

    for pdb_id in pdb_ids:
        _, trajectory, atom_numbers = load_ligand_trajectory(raw / "MD.hdf5", pdb_id)
        reference = torch.from_numpy(trajectory[0])
        edges = torch.from_numpy(infer_bonds(trajectory[0], atom_numbers))

        for task, (history_size, horizon) in TASKS.items():
            history = trajectory[:history_size]
            target = trajectory[history_size : history_size + horizon]
            for label, predictor in predictors.items():
                prediction = predictor(history, atom_numbers, horizon)
                if label in args.project:
                    prediction = project_bond_lengths(
                        torch.from_numpy(prediction),
                        reference,
                        edges,
                    ).numpy()
                curves[task][label]["rmsd"].append(rmsd_curve(prediction, target))
                bond = [
                    float(bond_length_error(torch.from_numpy(frame), reference, edges))
                    for frame in prediction
                ]
                curves[task][label]["bond"].append(bond)

    detail_rows = []
    summary_rows = []
    figures = {
        metric: plt.subplots(1, 3, figsize=(13, 3.8))
        for metric in ("rmsd", "bond")
    }

    for task_index, (task, (_, horizon)) in enumerate(TASKS.items()):
        steps = np.arange(1, horizon + 1)
        for label in predictors:
            for metric in ("rmsd", "bond"):
                values = np.asarray(curves[task][label][metric])
                mean = values.mean(axis=0)
                std = values.std(axis=0)
                figures[metric][1][task_index].plot(steps, mean, label=label, linewidth=2)
                for step, mean_value, std_value in zip(steps, mean, std):
                    detail_rows.append(
                        {
                            "split": args.split,
                            "task": task,
                            "model": label,
                            "metric": metric,
                            "horizon": int(step),
                            "mean": float(mean_value),
                            "std": float(std_value),
                            "n_complexes": len(pdb_ids),
                        }
                    )
                summary_rows.append(
                    {
                        "split": args.split,
                        "task": task,
                        "model": label,
                        "metric": metric,
                        "mean_over_horizon": float(mean.mean()),
                        "final_mean": float(mean[-1]),
                        "final_std": float(std[-1]),
                        "n_complexes": len(pdb_ids),
                    }
                )

        for metric, (_, axes) in figures.items():
            axes[task_index].set_title(f"{task}: {TASKS[task][0]} → {horizon}")
            axes[task_index].set_xlabel("Prediction horizon")
            axes[task_index].grid(alpha=0.25)
            if metric == "bond":
                axes[task_index].set_yscale("log")

    for metric, (figure, axes) in figures.items():
        axes[0].set_ylabel("Ligand RMSD (Å)" if metric == "rmsd" else "Bond error (Å)")
        axes[-1].legend(frameon=False)
        figure.suptitle(f"MISATO_100 {args.split} split (n={len(pdb_ids)})")
        figure.tight_layout()
        figure.savefig(results / f"{args.name}_{metric}_{args.split}.png", dpi=180)

    write_csv(results / f"{args.name}_curves_{args.split}.csv", detail_rows)
    write_csv(results / f"{args.name}_summary_{args.split}.csv", summary_rows)
    for row in summary_rows:
        if row["metric"] == "rmsd":
            print(
                row["task"],
                row["model"],
                f'final={row["final_mean"]:.4f} ± {row["final_std"]:.4f}',
            )


if __name__ == "__main__":
    main()
