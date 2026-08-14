"""在冻结 NeuralMD 的 train/val 缓存上训练概率残差模型。"""

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

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.residual_training import ResidualCacheDataset, residual_loss
from src.temporal_residual_model import TemporalProbabilisticResidual


VARIANTS = {
    "ode_mu": {"temporal": False, "probabilistic": False},
    "ode_mu_sigma": {"temporal": False, "probabilistic": True},
    "ode_temporal_mu_sigma": {"temporal": True, "probabilistic": True},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="ode_temporal_mu_sigma")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rbf-channels", type=int, default=16)
    parser.add_argument("--ligand-cutoff", type=float, default=6.0)
    parser.add_argument("--protein-cutoff", type=float, default=8.0)
    parser.add_argument("--frame-chunk-size", type=int, default=16)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer hyperparameters")
    if args.uncertainty_weight < 0 or args.gradient_clip <= 0:
        raise ValueError("invalid loss or clipping hyperparameters")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(item: dict, device: torch.device) -> tuple:
    return (
        item["neuralmd_positions"].to(device),
        item["ligand_atom_types"].to(device),
        item["ligand_masses"].to(device),
        item["protein_ca_positions"].to(device),
        item["protein_residue_types"].to(device),
    )


def run_epoch(
    model: TemporalProbabilisticResidual,
    loader: DataLoader,
    device: torch.device,
    uncertainty_weight: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float | None]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "mean_mse": 0.0, "point_rmse": 0.0, "final_rmse": 0.0}
    nll_total = 0.0
    nll_count = 0
    complexes = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for item in loader:
            residual = item["residual"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(*_to_device(item, device))
            terms = residual_loss(
                prediction,
                residual,
                uncertainty_weight=uncertainty_weight,
            )
            if training:
                terms.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            final_error = torch.linalg.vector_norm(prediction.mean[-1] - residual[-1], dim=-1).mean()
            totals["loss"] += float(terms.total.detach())
            totals["mean_mse"] += float(terms.mean_mse.detach())
            totals["point_rmse"] += float(terms.point_rmse.detach())
            totals["final_rmse"] += float(final_error.detach())
            if terms.nll is not None:
                nll_total += float(terms.nll.detach())
                nll_count += 1
            complexes += 1

    if complexes == 0:
        raise RuntimeError("cache loader produced no complexes")
    metrics = {key: value / complexes for key, value in totals.items()}
    metrics["nll"] = nll_total / nll_count if nll_count else None
    return metrics


def _write_history(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    _set_seed(args.seed)

    train_dataset = ResidualCacheDataset(args.cache_root / "train", expected_split="train")
    val_dataset = ResidualCacheDataset(args.cache_root / "val", expected_split="val")
    for dataset in (train_dataset, val_dataset):
        if dataset.manifest.get("task") != "T3":
            raise ValueError("residual training currently requires T3 caches")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)

    variant = VARIANTS[args.variant]
    model_config = {
        "hidden_dim": args.hidden_dim,
        "rbf_channels": args.rbf_channels,
        "ligand_cutoff": args.ligand_cutoff,
        "protein_cutoff": args.protein_cutoff,
        "frame_chunk_size": args.frame_chunk_size,
        "detach_uncertainty_features": True,
        "min_scale": 1e-3,
        "initial_scale": 1.0,
        **variant,
    }
    model = TemporalProbabilisticResidual(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = args.output_dir / "best_model.pth"
    latest_path = args.output_dir / "latest.pth"
    history_path = args.output_dir / "history.csv"
    if args.resume and not latest_path.is_file():
        raise FileNotFoundError(f"missing resumable checkpoint: {latest_path}")
    if not args.resume and not args.overwrite and (
        checkpoint_path.exists() or latest_path.exists() or history_path.exists()
    ):
        raise FileExistsError(f"training output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "variant": args.variant,
        "model": model_config,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "uncertainty_weight": args.uncertainty_weight,
        "gradient_clip": args.gradient_clip,
        "selection_metric": "val_point_rmse",
        "train_manifest_sha256": _sha256(args.cache_root / "train/manifest.json"),
        "val_manifest_sha256": _sha256(args.cache_root / "val/manifest.json"),
    }
    best_metric = float("inf")
    stale_epochs = 0
    history = []
    start_epoch = 1
    if args.resume:
        latest = torch.load(latest_path, map_location=device, weights_only=True)
        if latest["variant"] != args.variant or latest["model_config"] != model_config:
            raise ValueError("resume arguments do not match the saved model contract")
        comparable_config = {key: value for key, value in config.items() if key not in {"epochs", "patience"}}
        saved_config = {
            key: value for key, value in latest["config"].items() if key not in {"epochs", "patience"}
        }
        if saved_config != comparable_config:
            raise ValueError("resume arguments or cache manifests do not match the saved run")
        model.load_state_dict(latest["model_state_dict"], strict=True)
        optimizer.load_state_dict(latest["optimizer_state_dict"])
        generator.set_state(latest["generator_state"])
        best_metric = float(latest["best_metric"])
        stale_epochs = int(latest["stale_epochs"])
        history = latest["history"]
        start_epoch = int(latest["epoch"]) + 1
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.uncertainty_weight,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
        )
        val_metrics = run_epoch(model, val_loader, device, args.uncertainty_weight)
        row = {"epoch": epoch, "seconds": time.perf_counter() - epoch_started}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        _write_history(history_path, history)
        print(json.dumps(row), flush=True)

        selection_metric = float(val_metrics["point_rmse"])
        if selection_metric < best_metric:
            best_metric = selection_metric
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "variant": args.variant,
                "epoch": epoch,
                "val_point_rmse": selection_metric,
                "config": config,
            }
            temporary = checkpoint_path.with_suffix(".tmp.pth")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale_epochs += 1

        latest = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "variant": args.variant,
            "epoch": epoch,
            "best_metric": best_metric,
            "stale_epochs": stale_epochs,
            "history": history,
            "generator_state": generator.get_state(),
            "config": config,
        }
        temporary = latest_path.with_suffix(".tmp.pth")
        torch.save(latest, temporary)
        temporary.replace(latest_path)
        if stale_epochs >= args.patience:
            break

    print(json.dumps({"best_val_point_rmse": best_metric, "checkpoint": str(checkpoint_path)}))


if __name__ == "__main__":
    main()
