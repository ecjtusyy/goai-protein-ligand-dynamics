"""在 unseen test 上公平比较 NeuralMD-ODE 与训练后的残差校正器。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.run_official_neuralmd import (
    ensure_supported_device,
    git_commit,
    import_upstream,
    predict_trajectory,
    preflight_model,
    published_model_args,
    safe_load_checkpoint,
    trajectory_metrics,
    write_csv,
)
from src.neuralmd_official import ROLLOUT_WINDOWS, verify_misato1000
from src.probabilistic_evaluation import probabilistic_metrics
from src.temporal_residual_model import TemporalProbabilisticResidual


METRIC_NAMES = (
    "mae",
    "rmse",
    "matching",
    "stability",
    "ligand_collision",
    "binding_collision",
    "com_error",
    "rg_error",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--ode-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--residual-checkpoint",
        type=Path,
        action="append",
        required=True,
        help="可重复传入；多个消融共享一次 NeuralMD-ODE rollout",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-complexes", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def frame_metric_rows(method: str, pdb_id: str, metrics: dict) -> list[dict]:
    rows = []
    window = ROLLOUT_WINDOWS["T3"]
    for step in range(window.horizon):
        rows.append(
            {
                "method": method,
                "task": "T3",
                "pdb_id": pdb_id,
                "step": step + 1,
                "target_frame": window.target_start + step,
                "ligand_atoms": metrics["ligand_atoms"],
                "protein_residues": metrics["protein_residues"],
                **{name: metrics[name][step] for name in METRIC_NAMES},
            }
        )
    return rows


def summarize_comparison(frame_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    complex_rows = []
    summary_rows = []
    methods = list(dict.fromkeys(row["method"] for row in frame_rows))
    for method in methods:
        method_rows = [row for row in frame_rows if row["method"] == method]
        pdb_ids = list(dict.fromkeys(row["pdb_id"] for row in method_rows))
        for pdb_id in pdb_ids:
            rows = [row for row in method_rows if row["pdb_id"] == pdb_id]
            item = {
                "method": method,
                "task": "T3",
                "pdb_id": pdb_id,
                "ligand_atoms": rows[0]["ligand_atoms"],
                "protein_residues": rows[0]["protein_residues"],
            }
            for metric in METRIC_NAMES:
                item[f"mean_{metric}"] = float(np.mean([row[metric] for row in rows]))
                item[f"final_{metric}"] = rows[-1][metric]
            complex_rows.append(item)

        item = {"method": method, "task": "T3", "complexes": len(pdb_ids), "frames": len(method_rows)}
        for metric in METRIC_NAMES:
            item[f"mean_{metric}"] = float(np.mean([row[metric] for row in method_rows]))
            final_values = [row[f"final_{metric}"] for row in complex_rows if row["method"] == method]
            item[f"final_{metric}"] = float(np.mean(final_values))
        summary_rows.append(item)
    return complex_rows, summary_rows


def load_residual_model(torch, checkpoint_path: Path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TemporalProbabilisticResidual(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.limit_complexes is not None and args.limit_complexes < 1:
        raise ValueError("--limit-complexes must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"evaluation output already exists: {args.output_dir}")

    dataset_metadata = verify_misato1000(args.dataset_dir, strict_size=args.strict_size)
    upstream = import_upstream(args.official_repo)
    torch = upstream.torch
    device = ensure_supported_device(torch, args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_args = published_model_args()
    ode_model = upstream.Model(model_args).to(device)
    ode_checkpoint = safe_load_checkpoint(torch, args.ode_checkpoint, device)
    ode_model.load_state_dict(ode_checkpoint["binding_model"], strict=True)
    ode_model.eval()
    residual_models = []
    for path in args.residual_checkpoint:
        residual_model, residual_checkpoint = load_residual_model(torch, path, device)
        residual_models.append((residual_checkpoint["variant"], residual_model, path))
    variants = [variant for variant, _, _ in residual_models]
    if len(set(variants)) != len(variants):
        raise ValueError(f"duplicate residual variants: {variants}")

    dataset = upstream.Dataset(str(args.dataset_dir), mode="test")
    test_ids = [
        line.strip().upper()
        for line in (args.dataset_dir / "raw/test_MD.txt").read_text().splitlines()
        if line.strip()
    ]
    if len(dataset) != len(test_ids):
        raise RuntimeError("processed test dataset and test split ID count differ")
    if args.limit_complexes is not None:
        limit = min(args.limit_complexes, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(limit))
        test_ids = test_ids[:limit]
    loader = upstream.DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False)
    preflight = preflight_model(upstream, ode_model, next(iter(loader)), model_args, device)
    print("[preflight] " + json.dumps(preflight, ensure_ascii=False), flush=True)

    frame_rows = []
    calibration_rows = []
    window = ROLLOUT_WINDOWS["T3"]
    with torch.no_grad():
        for index, (pdb_id, batch) in enumerate(zip(test_ids, loader), start=1):
            print(f"[{index:03d}/{len(test_ids):03d}] test/{pdb_id}", flush=True)
            device_batch, neuralmd_positions, target = predict_trajectory(
                upstream, ode_model, batch, window, model_args, device
            )
            baseline = trajectory_metrics(upstream, device_batch, neuralmd_positions, target)
            frame_rows.extend(frame_metric_rows("neuralmd_ode", pdb_id, baseline))

            for variant, residual_model, _ in residual_models:
                residual_prediction = residual_model(
                    neuralmd_positions,
                    device_batch.ligand_x,
                    device_batch.ligand_mass,
                    device_batch.protein_pos[device_batch.mask_ca],
                    device_batch.protein_backbone_residue,
                )
                corrected = neuralmd_positions + residual_prediction.mean
                corrected_metrics = trajectory_metrics(upstream, device_batch, corrected, target)
                frame_rows.extend(frame_metric_rows(variant, pdb_id, corrected_metrics))
                if residual_prediction.scale is not None:
                    calibration_rows.append(
                        {
                            "method": variant,
                            "pdb_id": pdb_id,
                            **probabilistic_metrics(
                                residual_prediction.mean,
                                residual_prediction.scale,
                                target - neuralmd_positions,
                            ),
                        }
                    )

    complex_rows, summary_rows = summarize_comparison(frame_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "comparison_frames.csv", frame_rows)
    write_csv(args.output_dir / "comparison_complexes.csv", complex_rows)
    write_csv(args.output_dir / "comparison_summary.csv", summary_rows)
    calibration_summary = []
    if calibration_rows:
        write_csv(args.output_dir / "calibration_complexes.csv", calibration_rows)
        for variant in variants:
            rows = [row for row in calibration_rows if row["method"] == variant]
            if not rows:
                continue
            calibration_summary.append(
                {
                    "method": variant,
                    "complexes": len(rows),
                    **{
                        key: float(np.mean([row[key] for row in rows]))
                        for key in rows[0]
                        if key not in {"method", "pdb_id"}
                    },
                }
            )
        write_csv(args.output_dir / "calibration_summary.csv", calibration_summary)

    experiment = {
        "dataset": dataset_metadata,
        "task": "T3",
        "seed": args.seed,
        "limit_complexes": args.limit_complexes,
        "variants": variants,
        "ode_checkpoint_sha256": hashlib.sha256(args.ode_checkpoint.read_bytes()).hexdigest(),
        "residual_checkpoint_sha256": {
            variant: hashlib.sha256(path.read_bytes()).hexdigest()
            for variant, _, path in residual_models
        },
        "official_repo_commit": git_commit(args.official_repo),
        "goai_repo_commit": git_commit(REPO_ROOT),
        "preflight": preflight,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    print(json.dumps({"comparison": summary_rows, "calibration": calibration_summary}, indent=2))


if __name__ == "__main__":
    main()
