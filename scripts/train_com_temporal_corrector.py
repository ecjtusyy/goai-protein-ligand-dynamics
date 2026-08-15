"""在冻结的 T3 train/val 缓存上训练质量中心时序修正器。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.com_temporal_model import COMTemporalCorrector, com_temporal_loss
from src.residual_training import ResidualCacheDataset


RESULT_FILES = (
    "com_temporal_val_frames.csv",
    "com_temporal_val_complexes.csv",
    "com_temporal_summary.csv",
    "com_temporal_decision.csv",
    "com_temporal_val_curves.png",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--final-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rbf-channels", type=int, default=8)
    parser.add_argument("--protein-cutoff", type=float, default=12.0)
    parser.add_argument("--history-conditioning", action="store_true")
    parser.add_argument("--success-threshold-pct", type=float, default=5.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer hyperparameters")
    if args.final_weight < 0 or args.gradient_clip <= 0:
        raise ValueError("invalid loss hyperparameters")
    if args.success_threshold_pct <= 0:
        raise ValueError("--success-threshold-pct must be positive")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_improvement(baseline: float, corrected: float) -> float:
    return 100.0 * (baseline - corrected) / baseline if baseline > 0 else 0.0


def _restore_generator_state(generator: torch.Generator, state: torch.Tensor) -> None:
    """DataLoader Generator 固定在 CPU，避免恢复时被 map 到 CUDA。"""

    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
        raise TypeError("generator_state must be a torch.uint8 tensor")
    generator.set_state(state.detach().cpu().contiguous())


def _inputs(item: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    positions = item["neuralmd_positions"].to(device=device, dtype=torch.float32)
    masses = item["ligand_masses"].to(device=device, dtype=positions.dtype)
    protein = item["protein_ca_positions"].to(device=device, dtype=positions.dtype)
    residual = item["residual"].to(device=device, dtype=positions.dtype)
    observed = item.get("observed_positions")
    if observed is not None:
        observed = observed.to(device=device, dtype=positions.dtype)
    return positions, masses, protein, residual, observed


def _point_metrics(predicted_com: torch.Tensor, residual: torch.Tensor) -> dict[str, torch.Tensor]:
    baseline_distance = torch.linalg.vector_norm(residual, dim=-1)
    corrected_error = residual - predicted_com[:, None, :]
    corrected_distance = torch.linalg.vector_norm(corrected_error, dim=-1)
    return {
        "baseline_mean_point_rmse": baseline_distance.mean(),
        "corrected_mean_point_rmse": corrected_distance.mean(),
        "baseline_final_point_rmse": baseline_distance[-1].mean(),
        "corrected_final_point_rmse": corrected_distance[-1].mean(),
    }


def run_epoch(
    model: COMTemporalCorrector,
    loader: DataLoader,
    device: torch.device,
    final_weight: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "com_mean_mse": 0.0,
        "com_final_mse": 0.0,
        "baseline_mean_point_rmse": 0.0,
        "corrected_mean_point_rmse": 0.0,
        "baseline_final_point_rmse": 0.0,
        "corrected_final_point_rmse": 0.0,
    }
    complexes = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for item in loader:
            positions, masses, protein, residual, observed = _inputs(item, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(
                positions,
                masses,
                protein,
                observed_positions=observed if model.history_conditioning else None,
            )
            loss, mean_mse, final_mse = com_temporal_loss(
                prediction.mean,
                residual,
                masses,
                final_weight=final_weight,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            metrics = _point_metrics(prediction.mean, residual)
            totals["loss"] += float(loss.detach())
            totals["com_mean_mse"] += float(mean_mse.detach())
            totals["com_final_mse"] += float(final_mse.detach())
            for name, value in metrics.items():
                totals[name] += float(value.detach())
            complexes += 1

    if complexes == 0:
        raise RuntimeError("cache loader produced no complexes")
    return {name: value / complexes for name, value in totals.items()}


def _selection_score(metrics: dict[str, float]) -> float:
    """最小化 mean/final 中表现更差的一项，直接对应双 5% 门槛。"""

    mean_ratio = metrics["corrected_mean_point_rmse"] / metrics[
        "baseline_mean_point_rmse"
    ]
    final_ratio = metrics["corrected_final_point_rmse"] / metrics[
        "baseline_final_point_rmse"
    ]
    return max(mean_ratio, final_ratio)


def _save_checkpoint(
    path: Path,
    model: COMTemporalCorrector,
    model_config: dict,
    config: dict,
    epoch: int,
    score: float,
    val_metrics: dict[str, float],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "config": config,
        "epoch": epoch,
        "selection_score": score,
        "val_metrics": val_metrics,
    }
    temporary = path.with_suffix(".tmp.pth")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _evaluate_best(
    model: COMTemporalCorrector,
    dataset: ResidualCacheDataset,
    device: torch.device,
    *,
    best_epoch: int,
    threshold: float,
    output_dir: Path,
) -> dict:
    model.eval()
    complex_rows = []
    frame_accumulator: list[list[tuple[float, float]]] | None = None
    max_rg_delta = 0.0

    with torch.no_grad():
        for item in dataset:
            positions, masses, protein, residual, observed = _inputs(item, device)
            predicted_com = model(
                positions,
                masses,
                protein,
                observed_positions=observed if model.history_conditioning else None,
            ).mean
            baseline = torch.linalg.vector_norm(residual, dim=-1).mean(dim=1)
            corrected = torch.linalg.vector_norm(
                residual - predicted_com[:, None, :], dim=-1
            ).mean(dim=1)
            if frame_accumulator is None:
                frame_accumulator = [[] for _ in range(positions.shape[0])]
            if len(frame_accumulator) != positions.shape[0]:
                raise ValueError("all T3 validation trajectories must have equal length")
            for frame, (base, fixed) in enumerate(zip(baseline, corrected)):
                frame_accumulator[frame].append((float(base), float(fixed)))

            corrected_positions = positions + predicted_com[:, None, :]
            baseline_centered = positions - positions.mean(dim=1, keepdim=True)
            corrected_centered = corrected_positions - corrected_positions.mean(
                dim=1, keepdim=True
            )
            baseline_rg = torch.sqrt(baseline_centered.square().sum(dim=-1).mean(dim=1))
            corrected_rg = torch.sqrt(corrected_centered.square().sum(dim=-1).mean(dim=1))
            max_rg_delta = max(
                max_rg_delta,
                float((baseline_rg - corrected_rg).abs().max()),
            )

            baseline_mean = float(baseline.mean())
            corrected_mean = float(corrected.mean())
            baseline_final = float(baseline[-1])
            corrected_final = float(corrected[-1])
            complex_rows.append(
                {
                    "split": "val",
                    "pdb_id": str(item["pdb_id"]),
                    "frames": positions.shape[0],
                    "ligand_atoms": positions.shape[1],
                    "baseline_mean_point_rmse": baseline_mean,
                    "corrected_mean_point_rmse": corrected_mean,
                    "mean_point_improvement_pct": _relative_improvement(
                        baseline_mean, corrected_mean
                    ),
                    "baseline_final_point_rmse": baseline_final,
                    "corrected_final_point_rmse": corrected_final,
                    "final_point_improvement_pct": _relative_improvement(
                        baseline_final, corrected_final
                    ),
                }
            )

    if frame_accumulator is None:
        raise RuntimeError("validation dataset is empty")
    frame_rows = []
    for frame, values in enumerate(frame_accumulator):
        baseline = float(np.mean([value[0] for value in values]))
        corrected = float(np.mean([value[1] for value in values]))
        frame_rows.append(
            {
                "split": "val",
                "step": frame + 1,
                "complexes": len(values),
                "baseline_point_rmse": baseline,
                "corrected_point_rmse": corrected,
                "point_improvement_pct": _relative_improvement(baseline, corrected),
            }
        )

    baseline_mean = float(np.mean([row["baseline_point_rmse"] for row in frame_rows]))
    corrected_mean = float(np.mean([row["corrected_point_rmse"] for row in frame_rows]))
    baseline_final = frame_rows[-1]["baseline_point_rmse"]
    corrected_final = frame_rows[-1]["corrected_point_rmse"]
    mean_improvement = _relative_improvement(baseline_mean, corrected_mean)
    final_improvement = _relative_improvement(baseline_final, corrected_final)
    mean_pass = mean_improvement >= threshold
    final_pass = final_improvement >= threshold
    stability_pass = max_rg_delta <= 1e-4
    passed = bool(mean_pass and final_pass and stability_pass)

    summary = {
        "split": "val",
        "complexes": len(complex_rows),
        "frames": sum(row["frames"] for row in complex_rows),
        "best_epoch": best_epoch,
        "baseline_mean_point_rmse": baseline_mean,
        "corrected_mean_point_rmse": corrected_mean,
        "mean_point_improvement_pct": mean_improvement,
        "baseline_final_point_rmse": baseline_final,
        "corrected_final_point_rmse": corrected_final,
        "final_point_improvement_pct": final_improvement,
        "max_rigid_translation_rg_delta": max_rg_delta,
        "stability_delta_pp_theory": 0.0,
        "mean_gate_pass": mean_pass,
        "final_gate_pass": final_pass,
        "stability_gate_pass": stability_pass,
        "model_gate_pass": passed,
    }
    decision = {
        "decision_split": "val",
        "success_threshold_pct": threshold,
        "mean_point_improvement_pct": mean_improvement,
        "final_point_improvement_pct": final_improvement,
        "stability_delta_pp_theory": 0.0,
        "proceed_to_local_residual_model": passed,
        "decision": "PASS" if passed else "STOP_AND_TUNE_COM",
    }
    _write_csv(output_dir / "com_temporal_val_frames.csv", frame_rows)
    _write_csv(output_dir / "com_temporal_val_complexes.csv", complex_rows)
    _write_csv(output_dir / "com_temporal_summary.csv", [summary])
    _write_csv(output_dir / "com_temporal_decision.csv", [decision])

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    steps = [row["step"] for row in frame_rows]
    axis.plot(
        steps,
        [row["baseline_point_rmse"] for row in frame_rows],
        label="NeuralMD-ODE",
        linewidth=2,
    )
    axis.plot(
        steps,
        [row["corrected_point_rmse"] for row in frame_rows],
        label="ODE + learned COM temporal",
        linewidth=2,
    )
    axis.set(
        title="val: learned COM temporal corrector",
        xlabel="Rollout step",
        ylabel="Mean atom distance (Å)",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    temporary = output_dir / ".com_temporal_val_curves.tmp.png"
    figure.savefig(temporary, dpi=180, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output_dir / "com_temporal_val_curves.png")
    return {"summary": summary, "decision": decision}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    _set_seed(args.seed)

    train_dataset = ResidualCacheDataset(
        args.cache_root / "train",
        expected_split="train",
        require_history=args.history_conditioning,
    )
    val_dataset = ResidualCacheDataset(
        args.cache_root / "val",
        expected_split="val",
        require_history=args.history_conditioning,
    )
    for dataset in (train_dataset, val_dataset):
        if dataset.manifest.get("task") != "T3":
            raise ValueError("COM temporal training requires T3 caches")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)
    model_config = {
        "hidden_dim": args.hidden_dim,
        "rbf_channels": args.rbf_channels,
        "protein_cutoff": args.protein_cutoff,
    }
    # False 时保持旧 checkpoint 的配置合同和 state_dict 完全兼容。
    if args.history_conditioning:
        model_config["history_conditioning"] = True
    model = COMTemporalCorrector(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_path = args.output_dir / "best_model.pth"
    latest_path = args.output_dir / "latest.pth"
    history_path = args.output_dir / "history.csv"
    managed_files = (best_path, latest_path, history_path) + tuple(
        args.output_dir / name for name in RESULT_FILES
    )
    if args.resume and not latest_path.is_file():
        raise FileNotFoundError(f"missing resumable checkpoint: {latest_path}")
    if not args.resume and not args.overwrite and any(path.exists() for path in managed_files):
        raise FileExistsError(f"training output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_config,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "final_weight": args.final_weight,
        "gradient_clip": args.gradient_clip,
        "success_threshold_pct": args.success_threshold_pct,
        "selection_metric": "max(val_mean_ratio, val_final_ratio)",
        "safe_fallback": "epoch_0_zero_correction",
        "train_manifest_sha256": _sha256(args.cache_root / "train/manifest.json"),
        "val_manifest_sha256": _sha256(args.cache_root / "val/manifest.json"),
    }
    if args.history_conditioning:
        config["history_conditioning"] = True
    comparable_keys = {"epochs", "patience"}
    history: list[dict] = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1

    if args.resume:
        latest = torch.load(latest_path, map_location="cpu", weights_only=True)
        saved = {key: value for key, value in latest["config"].items() if key not in comparable_keys}
        current = {key: value for key, value in config.items() if key not in comparable_keys}
        if latest["model_config"] != model_config or saved != current:
            raise ValueError("resume arguments or cache manifests do not match the saved run")
        model.load_state_dict(latest["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        _restore_generator_state(generator, latest["generator_state"])
        history = latest["history"]
        best_score = float(latest["best_score"])
        best_epoch = int(latest["best_epoch"])
        stale_epochs = int(latest["stale_epochs"])
        start_epoch = int(latest["epoch"]) + 1
    else:
        # epoch 0 是严格等价于 NeuralMD 的安全候选，不允许训练结果把它覆盖坏。
        initial_val = run_epoch(model, val_loader, device, args.final_weight)
        best_score = _selection_score(initial_val)
        _save_checkpoint(
            best_path,
            model,
            model_config,
            config,
            epoch=0,
            score=best_score,
            val_metrics=initial_val,
        )

    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.final_weight,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
        )
        val_metrics = run_epoch(model, val_loader, device, args.final_weight)
        score = _selection_score(val_metrics)
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
            "val_mean_improvement_pct": _relative_improvement(
                val_metrics["baseline_mean_point_rmse"],
                val_metrics["corrected_mean_point_rmse"],
            ),
            "val_final_improvement_pct": _relative_improvement(
                val_metrics["baseline_final_point_rmse"],
                val_metrics["corrected_final_point_rmse"],
            ),
            "selection_score": score,
        }
        history.append(row)
        _write_csv(history_path, history)
        print(json.dumps(row), flush=True)

        if score < best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _save_checkpoint(
                best_path,
                model,
                model_config,
                config,
                epoch,
                score,
                val_metrics,
            )
        else:
            stale_epochs += 1

        latest = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "config": config,
            "epoch": epoch,
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "generator_state": generator.get_state(),
        }
        temporary = latest_path.with_suffix(".tmp.pth")
        torch.save(latest, temporary)
        temporary.replace(latest_path)
        if stale_epochs >= args.patience:
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.to(device)
    result = _evaluate_best(
        model,
        val_dataset,
        device,
        best_epoch=int(best["epoch"]),
        threshold=args.success_threshold_pct,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
