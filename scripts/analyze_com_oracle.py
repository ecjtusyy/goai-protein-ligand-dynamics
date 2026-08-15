"""仅用 train/val 残差缓存评估质量加权 COM Oracle 上界。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.com_oracle import analyze_com_oracle
from src.residual_cache import TRAINING_SPLITS
from src.residual_training import ResidualCacheDataset


OUTPUT_FILES = (
    "com_oracle_frames.csv",
    "com_oracle_complexes.csv",
    "com_oracle_summary.csv",
    "com_oracle_decision.csv",
    "com_oracle_curves.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=TRAINING_SPLITS,
        default=list(TRAINING_SPLITS),
    )
    parser.add_argument("--success-threshold-pct", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _relative_improvement(baseline: float, oracle: float) -> float:
    return 100.0 * (baseline - oracle) / baseline if baseline > 0 else 0.0


def _complex_row(split: str, pdb_id: str, frame_rows: list[dict]) -> dict:
    return {
        "split": split,
        "pdb_id": pdb_id,
        "ligand_atoms": frame_rows[0]["ligand_atoms"],
        "frames": len(frame_rows),
        "mean_baseline_point_rmse": float(
            np.mean([row["baseline_point_rmse"] for row in frame_rows])
        ),
        "mean_oracle_point_rmse": float(
            np.mean([row["oracle_point_rmse"] for row in frame_rows])
        ),
        "mean_point_improvement_pct": _relative_improvement(
            float(np.mean([row["baseline_point_rmse"] for row in frame_rows])),
            float(np.mean([row["oracle_point_rmse"] for row in frame_rows])),
        ),
        "final_baseline_point_rmse": frame_rows[-1]["baseline_point_rmse"],
        "final_oracle_point_rmse": frame_rows[-1]["oracle_point_rmse"],
        "final_point_improvement_pct": _relative_improvement(
            frame_rows[-1]["baseline_point_rmse"],
            frame_rows[-1]["oracle_point_rmse"],
        ),
        "mean_weighted_com_error_fraction": float(
            np.mean([row["weighted_com_error_fraction"] for row in frame_rows])
        ),
        "final_weighted_com_error_fraction": frame_rows[-1][
            "weighted_com_error_fraction"
        ],
        "max_pythagorean_abs_error": float(
            np.max([row["pythagorean_abs_error"] for row in frame_rows])
        ),
    }


def summarize(frame_rows: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    for split in dict.fromkeys(row["split"] for row in frame_rows):
        rows = [row for row in frame_rows if row["split"] == split]
        final_step = max(row["step"] for row in rows)
        final_rows = [row for row in rows if row["step"] == final_step]
        baseline_mean = float(np.mean([row["baseline_point_rmse"] for row in rows]))
        oracle_mean = float(np.mean([row["oracle_point_rmse"] for row in rows]))
        baseline_final = float(
            np.mean([row["baseline_point_rmse"] for row in final_rows])
        )
        oracle_final = float(np.mean([row["oracle_point_rmse"] for row in final_rows]))
        mean_improvement = _relative_improvement(baseline_mean, oracle_mean)
        final_improvement = _relative_improvement(baseline_final, oracle_final)
        summary_rows.append(
            {
                "split": split,
                "complexes": len({row["pdb_id"] for row in rows}),
                "frames": len(rows),
                "baseline_mean_point_rmse": baseline_mean,
                "oracle_mean_point_rmse": oracle_mean,
                "mean_point_improvement_pct": mean_improvement,
                "baseline_final_point_rmse": baseline_final,
                "oracle_final_point_rmse": oracle_final,
                "final_point_improvement_pct": final_improvement,
                "mean_weighted_com_error_fraction": float(
                    np.mean([row["weighted_com_error_fraction"] for row in rows])
                ),
                "stability_delta_pp_theory": 0.0,
                "mean_gate_pass": mean_improvement >= threshold,
                "final_gate_pass": final_improvement >= threshold,
                "stability_gate_pass": True,
            }
        )

    val = next((row for row in summary_rows if row["split"] == "val"), None)
    if val is None:
        raise ValueError("val split is required for the COM Oracle decision")
    proceed = bool(
        val["mean_gate_pass"] and val["final_gate_pass"] and val["stability_gate_pass"]
    )
    decision_rows = [
        {
            "decision_split": "val",
            "success_threshold_pct": threshold,
            "mean_point_improvement_pct": val["mean_point_improvement_pct"],
            "final_point_improvement_pct": val["final_point_improvement_pct"],
            "stability_delta_pp_theory": 0.0,
            "proceed_to_com_temporal_model": proceed,
            "decision": "PROCEED" if proceed else "STOP",
        }
    ]
    return summary_rows, decision_rows


def _plot(path: Path, frame_rows: list[dict]) -> None:
    splits = list(dict.fromkeys(row["split"] for row in frame_rows))
    figure, axes = plt.subplots(1, len(splits), figsize=(7 * len(splits), 4.5), squeeze=False)
    for axis, split in zip(axes[0], splits):
        rows = [row for row in frame_rows if row["split"] == split]
        steps = sorted({row["step"] for row in rows})
        baseline = [
            np.mean(
                [row["baseline_point_rmse"] for row in rows if row["step"] == step]
            )
            for step in steps
        ]
        oracle = [
            np.mean([row["oracle_point_rmse"] for row in rows if row["step"] == step])
            for step in steps
        ]
        axis.plot(steps, baseline, label="NeuralMD-ODE", linewidth=2)
        axis.plot(steps, oracle, label="Mass-COM Oracle", linewidth=2)
        axis.set(
            title=f"{split}: T3 COM Oracle",
            xlabel="Rollout step",
            ylabel="Mean atom distance (Å)",
        )
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    temporary = path.with_name(f".{path.stem}.tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.success_threshold_pct <= 0:
        raise ValueError("--success-threshold-pct must be positive")
    if "val" not in args.splits:
        raise ValueError("val split is required; test is forbidden and train alone cannot decide")

    existing = [args.output_dir / name for name in OUTPUT_FILES if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"COM Oracle outputs already exist: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    complex_rows = []
    for split in args.splits:
        dataset = ResidualCacheDataset(args.cache_root / split, expected_split=split)
        for index, item in enumerate(dataset, start=1):
            pdb_id = str(item["pdb_id"])
            metrics = analyze_com_oracle(
                item["neuralmd_positions"].numpy(),
                item["true_positions"].numpy(),
                item["residual"].numpy(),
                item["ligand_masses"].numpy(),
            )
            target_frames = item["target_frames"].numpy()
            rows = []
            for frame in range(target_frames.shape[0]):
                row = {
                    "split": split,
                    "pdb_id": pdb_id,
                    "step": frame + 1,
                    "target_frame": int(target_frames[frame]),
                    "ligand_atoms": int(item["ligand_masses"].numel()),
                    **{name: float(values[frame]) for name, values in metrics.items()},
                }
                rows.append(row)
                frame_rows.append(row)
            complex_rows.append(_complex_row(split, pdb_id, rows))
            print(f"[{index:03d}/{len(dataset):03d}] {split}/{pdb_id}", flush=True)

    summary_rows, decision_rows = summarize(frame_rows, args.success_threshold_pct)
    _write_csv(args.output_dir / "com_oracle_frames.csv", frame_rows)
    _write_csv(args.output_dir / "com_oracle_complexes.csv", complex_rows)
    _write_csv(args.output_dir / "com_oracle_summary.csv", summary_rows)
    _write_csv(args.output_dir / "com_oracle_decision.csv", decision_rows)
    _plot(args.output_dir / "com_oracle_curves.png", frame_rows)
    print(json.dumps({"summary": summary_rows, "decision": decision_rows[0]}, indent=2))


if __name__ == "__main__":
    main()
