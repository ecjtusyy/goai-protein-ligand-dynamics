"""对 history-conditioned COM corrector 做可恢复的两阶段小型超参数筛选。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SweepConfig:
    config_id: str
    learning_rate: float
    final_weight: float


@dataclass(frozen=True)
class RunRequest:
    stage: str
    config: SweepConfig
    seed: int
    epochs: int
    patience: int
    output_dir: Path


SWEEP_CONFIGS = tuple(
    SweepConfig(
        config_id=(
            f"lr_{str(learning_rate).replace('.', 'p')}"
            f"__fw_{str(final_weight).replace('.', 'p')}"
        ),
        learning_rate=learning_rate,
        final_weight=final_weight,
    )
    for learning_rate in (5e-4, 1e-3, 1.5e-3)
    for final_weight in (0.10, 0.25)
)

RESULT_FILES = (
    "history.csv",
    "best_model.pth",
    "latest.pth",
    "com_temporal_summary.csv",
    "com_temporal_decision.csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--coarse-seed", type=int, default=42)
    parser.add_argument("--verification-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--coarse-epochs", type=int, default=15)
    parser.add_argument("--full-epochs", type=int, default=40)
    parser.add_argument("--coarse-patience", type=int, default=4)
    parser.add_argument("--full-patience", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rbf-channels", type=int, default=8)
    parser.add_argument("--protein-cutoff", type=float, default=12.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--success-threshold-pct", type=float, default=5.0)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.coarse_epochs < 1 or args.full_epochs < args.coarse_epochs:
        raise ValueError("full epochs must be at least the positive coarse epochs")
    if args.coarse_patience < 1 or args.full_patience < args.coarse_patience:
        raise ValueError("full patience must be at least the positive coarse patience")
    if not 1 <= args.top_k <= len(SWEEP_CONFIGS):
        raise ValueError(f"top-k must be between 1 and {len(SWEEP_CONFIGS)}")
    if args.coarse_seed not in args.verification_seeds:
        raise ValueError("verification seeds must include the coarse seed")
    if len(set(args.verification_seeds)) != len(args.verification_seeds):
        raise ValueError("verification seeds must be unique")
    if args.success_threshold_pct <= 0:
        raise ValueError("success threshold must be positive")


def _sweep_contract(args: argparse.Namespace) -> dict:
    return {
        "cache_root": str(args.cache_root.resolve()),
        "device": args.device,
        "coarse_seed": args.coarse_seed,
        "verification_seeds": list(args.verification_seeds),
        "coarse_epochs": args.coarse_epochs,
        "full_epochs": args.full_epochs,
        "coarse_patience": args.coarse_patience,
        "full_patience": args.full_patience,
        "top_k": args.top_k,
        "hidden_dim": args.hidden_dim,
        "rbf_channels": args.rbf_channels,
        "protein_cutoff": args.protein_cutoff,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "success_threshold_pct": args.success_threshold_pct,
        "configs": [asdict(config) for config in SWEEP_CONFIGS],
        "selection_metric": "maximise min(mean improvement, final improvement)",
        "stable_rule": "at least a strict majority of verification seeds pass both gates",
    }


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_single_csv(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}")
    return rows[0]


def _completion_path(request: RunRequest) -> Path:
    return request.output_dir / "sweep_completion.json"


def _stage_complete(request: RunRequest) -> bool:
    path = _completion_path(request)
    if not path.is_file() or any(
        not (request.output_dir / name).is_file() for name in RESULT_FILES
    ):
        return False
    payload = json.loads(path.read_text())
    completed_rank = {"coarse": 1, "full": 2}.get(payload.get("stage"), 0)
    requested_rank = {"coarse": 1, "full": 2}[request.stage]
    return (
        completed_rank >= requested_rank
        and payload.get("config") == asdict(request.config)
        and int(payload.get("seed", -1)) == request.seed
        and int(payload.get("epochs", -1)) >= request.epochs
        and int(payload.get("patience", -1)) >= request.patience
    )


def _mark_complete(request: RunRequest) -> None:
    missing = [
        name for name in RESULT_FILES if not (request.output_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"training returned without required results: {missing}")
    _write_json_atomic(
        _completion_path(request),
        {
            "stage": request.stage,
            "config": asdict(request.config),
            "seed": request.seed,
            "epochs": request.epochs,
            "patience": request.patience,
        },
    )


def _run_training(request: RunRequest, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "scripts.train_com_temporal_corrector",
        "--cache-root",
        str(args.cache_root),
        "--output-dir",
        str(request.output_dir),
        "--device",
        args.device,
        "--seed",
        str(request.seed),
        "--epochs",
        str(request.epochs),
        "--patience",
        str(request.patience),
        "--learning-rate",
        str(request.config.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--gradient-clip",
        str(args.gradient_clip),
        "--final-weight",
        str(request.config.final_weight),
        "--hidden-dim",
        str(args.hidden_dim),
        "--rbf-channels",
        str(args.rbf_channels),
        "--protein-cutoff",
        str(args.protein_cutoff),
        "--history-conditioning",
        "--success-threshold-pct",
        str(args.success_threshold_pct),
    ]
    if (request.output_dir / "latest.pth").is_file():
        command.append("--resume")
    elif request.output_dir.exists() and any(request.output_dir.iterdir()):
        command.append("--overwrite")
    print("\n" + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=REPO_ROOT)


def _result_row(request: RunRequest) -> dict:
    summary = _read_single_csv(request.output_dir / "com_temporal_summary.csv")
    mean_improvement = float(summary["mean_point_improvement_pct"])
    final_improvement = float(summary["final_point_improvement_pct"])
    values = (mean_improvement, final_improvement)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite improvement in {request.output_dir}")
    minimum = min(values)
    return {
        "stage": request.stage,
        "config_id": request.config.config_id,
        "seed": request.seed,
        "learning_rate": request.config.learning_rate,
        "final_weight": request.config.final_weight,
        "best_epoch": int(summary["best_epoch"]),
        "mean_point_improvement_pct": mean_improvement,
        "final_point_improvement_pct": final_improvement,
        "min_gate_improvement_pct": minimum,
        "selection_score": 1.0 - minimum / 100.0,
        "model_gate_pass": str(summary["model_gate_pass"]).lower() == "true",
        "output_dir": str(request.output_dir),
        "checkpoint": str(request.output_dir / "best_model.pth"),
    }


def _rank_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["min_gate_improvement_pct"]),
            -float(row["mean_point_improvement_pct"]),
            -float(row["final_point_improvement_pct"]),
            str(row["config_id"]),
            int(row["seed"]),
        ),
    )


def _aggregate_configs(rows: list[dict], threshold: float) -> list[dict]:
    aggregates = []
    for config in SWEEP_CONFIGS:
        selected = [row for row in rows if row["config_id"] == config.config_id]
        if not selected:
            continue
        minima = np.asarray(
            [row["min_gate_improvement_pct"] for row in selected], dtype=float
        )
        mean_values = np.asarray(
            [row["mean_point_improvement_pct"] for row in selected], dtype=float
        )
        final_values = np.asarray(
            [row["final_point_improvement_pct"] for row in selected], dtype=float
        )
        pass_count = int(sum(value >= threshold for value in minima))
        aggregates.append(
            {
                "config_id": config.config_id,
                "learning_rate": config.learning_rate,
                "final_weight": config.final_weight,
                "seeds": len(selected),
                "mean_improvement_mean_pct": float(mean_values.mean()),
                "mean_improvement_std_pct": float(mean_values.std()),
                "final_improvement_mean_pct": float(final_values.mean()),
                "final_improvement_std_pct": float(final_values.std()),
                "median_min_gate_improvement_pct": float(np.median(minima)),
                "worst_seed_min_gate_improvement_pct": float(minima.min()),
                "passing_seeds": pass_count,
                "stable_gate_pass": pass_count >= len(selected) // 2 + 1,
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            -float(row["median_min_gate_improvement_pct"]),
            -float(row["worst_seed_min_gate_improvement_pct"]),
            str(row["config_id"]),
        ),
    )


def _plot_results(path: Path, rows: list[dict], threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for row in rows:
        axis.scatter(
            row["mean_point_improvement_pct"],
            row["final_point_improvement_pct"],
            s=48,
        )
        axis.annotate(
            f"{row['config_id']} s{row['seed']}",
            (row["mean_point_improvement_pct"], row["final_point_improvement_pct"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axvline(threshold, color="black", linestyle="--", linewidth=1)
    axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
    axis.set(
        title="History-conditioned COM sweep",
        xlabel="Mean point improvement (%)",
        ylabel="Final point improvement (%)",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    temporary = path.with_name(f".{path.stem}.tmp.png")
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def execute_sweep(args: argparse.Namespace, *, runner=None) -> dict:
    _validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / "sweep_contract.json"
    contract = _sweep_contract(args)
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "existing sweep output uses a different experiment contract; "
                "choose a new --output-root"
            )
    else:
        _write_json_atomic(contract_path, contract)
    if runner is None:
        runner = _run_training

    coarse_rows = []
    snapshot_dir = args.output_root / "coarse_snapshots"
    for config in SWEEP_CONFIGS:
        output_dir = args.output_root / "runs" / config.config_id / f"seed_{args.coarse_seed}"
        request = RunRequest(
            stage="coarse",
            config=config,
            seed=args.coarse_seed,
            epochs=args.coarse_epochs,
            patience=args.coarse_patience,
            output_dir=output_dir,
        )
        snapshot_path = snapshot_dir / f"{config.config_id}.json"
        if snapshot_path.is_file():
            row = json.loads(snapshot_path.read_text())
        else:
            if not _stage_complete(request):
                runner(request, args)
                _mark_complete(request)
            row = _result_row(request)
            _write_json_atomic(snapshot_path, row)
        coarse_rows.append(row)

    coarse_rows = _rank_rows(coarse_rows)
    _write_csv_atomic(args.output_root / "coarse_results.csv", coarse_rows)
    selected_path = args.output_root / "selected_configs.json"
    if selected_path.is_file():
        selected_ids = json.loads(selected_path.read_text())["selected_config_ids"]
    else:
        selected_ids = [row["config_id"] for row in coarse_rows[: args.top_k]]
        _write_json_atomic(
            selected_path,
            {
                "selection_metric": "maximise min(mean improvement, final improvement)",
                "selected_config_ids": selected_ids,
            },
        )
    config_by_id = {config.config_id: config for config in SWEEP_CONFIGS}
    if (
        len(selected_ids) != args.top_k
        or len(set(selected_ids)) != len(selected_ids)
        or any(config_id not in config_by_id for config_id in selected_ids)
    ):
        raise ValueError("selected_configs.json does not match this sweep contract")

    full_rows = []
    for config_id in selected_ids:
        config = config_by_id[config_id]
        for seed in args.verification_seeds:
            request = RunRequest(
                stage="full",
                config=config,
                seed=seed,
                epochs=args.full_epochs,
                patience=args.full_patience,
                output_dir=args.output_root / "runs" / config_id / f"seed_{seed}",
            )
            if not _stage_complete(request):
                runner(request, args)
                _mark_complete(request)
            full_rows.append(_result_row(request))

    full_rows = _rank_rows(full_rows)
    aggregates = _aggregate_configs(full_rows, args.success_threshold_pct)
    best_run = full_rows[0]
    robust_config = aggregates[0]
    stable_pass = bool(robust_config["stable_gate_pass"])
    single_pass = bool(best_run["min_gate_improvement_pct"] >= args.success_threshold_pct)
    if stable_pass:
        decision_name = "PASS_STABLE"
    elif single_pass:
        decision_name = "PASS_SINGLE_SEED_ONLY"
    else:
        decision_name = "STOP_AND_REVISE_LOSS"
    decision = {
        "success_threshold_pct": args.success_threshold_pct,
        "best_config_id": best_run["config_id"],
        "best_seed": best_run["seed"],
        "best_mean_point_improvement_pct": best_run["mean_point_improvement_pct"],
        "best_final_point_improvement_pct": best_run["final_point_improvement_pct"],
        "best_min_gate_improvement_pct": best_run["min_gate_improvement_pct"],
        "best_checkpoint": best_run["checkpoint"],
        "robust_config_id": robust_config["config_id"],
        "robust_passing_seeds": robust_config["passing_seeds"],
        "verification_seed_count": robust_config["seeds"],
        "single_checkpoint_gate_pass": single_pass,
        "stable_multi_seed_gate_pass": stable_pass,
        "proceed_to_local_residual_model": stable_pass,
        "decision": decision_name,
    }
    _write_csv_atomic(args.output_root / "full_results.csv", full_rows)
    _write_csv_atomic(args.output_root / "config_aggregates.csv", aggregates)
    _write_csv_atomic(args.output_root / "sweep_decision.csv", [decision])
    _plot_results(
        args.output_root / "sweep_improvements.png",
        full_rows,
        args.success_threshold_pct,
    )
    payload = {"decision": decision, "robust_config": robust_config}
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main(argv: list[str] | None = None) -> None:
    execute_sweep(parse_args(argv))


if __name__ == "__main__":
    main()
