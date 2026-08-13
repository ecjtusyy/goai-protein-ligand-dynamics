import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/goai-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from src.baselines import linear_forecast, static_forecast
from src.metrics import rmsd_curve
from src.misato import load_ligand_trajectory


TASKS = {"T1": (10, 10), "T2": (80, 20), "T3": (20, 80)}
MODELS = {"Static": static_forecast, "Linear": linear_forecast}


def read_ids(path: Path) -> list[str]:
    return [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw = root / "data/MISATO_100/raw"
    results = root / "results"
    results.mkdir(exist_ok=True)

    pdb_ids = read_ids(raw / f"{args.split}_MD.txt")
    curves = {task: {model: [] for model in MODELS} for task in TASKS}

    for pdb_id in pdb_ids:
        _, trajectory, _ = load_ligand_trajectory(raw / "MD.hdf5", pdb_id)
        for task, (history_size, horizon) in TASKS.items():
            history = trajectory[:history_size]
            target = trajectory[history_size : history_size + horizon]
            for model, forecast in MODELS.items():
                prediction = forecast(history, horizon)
                curves[task][model].append(rmsd_curve(prediction, target))

    detail_rows = []
    summary_rows = []
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for axis, (task, (_, horizon)) in zip(axes, TASKS.items()):
        steps = np.arange(1, horizon + 1)
        for model in MODELS:
            values = np.stack(curves[task][model])
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            axis.plot(steps, mean, label=model, linewidth=2)

            for step, mean_value, std_value in zip(steps, mean, std):
                detail_rows.append(
                    {
                        "split": args.split,
                        "task": task,
                        "model": model,
                        "horizon": int(step),
                        "mean_rmsd": float(mean_value),
                        "std_rmsd": float(std_value),
                        "n_complexes": len(pdb_ids),
                    }
                )

            summary_rows.append(
                {
                    "split": args.split,
                    "task": task,
                    "model": model,
                    "mean_over_horizon": float(mean.mean()),
                    "final_rmsd_mean": float(mean[-1]),
                    "final_rmsd_std": float(std[-1]),
                    "n_complexes": len(pdb_ids),
                }
            )

        axis.set_title(f"{task}: {TASKS[task][0]} → {horizon}")
        axis.set_xlabel("Prediction horizon")
        axis.grid(alpha=0.25)

    axes[0].set_ylabel("Ligand RMSD")
    axes[-1].legend(frameon=False)
    figure.suptitle(f"MISATO_100 {args.split} split (n={len(pdb_ids)})")
    figure.tight_layout()

    prefix = results / f"baseline_{args.split}"
    write_csv(prefix.with_suffix(".csv"), detail_rows)
    write_csv(results / f"baseline_summary_{args.split}.csv", summary_rows)
    figure.savefig(results / f"rmsd_horizon_{args.split}.png", dpi=180)

    for row in summary_rows:
        print(
            row["task"],
            row["model"],
            f'final={row["final_rmsd_mean"]:.4f} ± {row["final_rmsd_std"]:.4f}',
        )


if __name__ == "__main__":
    main()
