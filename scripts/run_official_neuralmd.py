"""Evaluate an official NeuralMD-ODE checkpoint on MISATO_1000.

The upstream training script does not expose checkpoint loading or eval-only
arguments. This wrapper keeps the published model, data loader, integrator and
metrics unchanged while adding those two missing experiment controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

# When this file is launched as ``python scripts/run_official_neuralmd.py``,
# Python puts ``scripts/`` rather than the repository root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.neuralmd_official import (
    MISATO_1000_HDF5_BYTES,
    NEURALMD_ODE_CHECKPOINT_BYTES,
    ROLLOUT_WINDOWS,
    rollout_contract,
    verify_misato1000,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=ROLLOUT_WINDOWS, default=list(ROLLOUT_WINDOWS))
    parser.add_argument("--limit-complexes", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-size", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def published_model_args() -> SimpleNamespace:
    """Architecture/integration values shipped with MISATO_1000 checkpoints."""

    return SimpleNamespace(
        model_3d_ligand="FrameNet01",
        model_3d_protein="FrameNetProtein03",
        emb_dim=128,
        NeuralMD_velocity_refined_value_coefficient=0.01,
        use_MLP_velocity=False,
        FrameNet_cutoff=5.0,
        FrameNet_num_layers=4,
        FrameNet_complex_layer=1,
        FrameNet_num_radial=100,
        FrameNet_rbf_type="RBF_repredding_01",
        FrameNet_gamma=None,
        FrameNet_readout="mean",
        NeuralMD_step_size=5.0,
        NeuralMD_scaling=100.0,
        ODE_method="euler",
    )


def import_upstream(official_repo: Path):
    required = [
        official_repo / "NeuralMD/datasets/MISATO/dataset_MISATO_semi_flexible.py",
        official_repo / "examples/models/NeuralMD_Binding01_2nd_ODE.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Official NeuralMD checkout is incomplete: " + ", ".join(missing))

    sys.path.insert(0, str(official_repo.resolve()))
    sys.path.insert(0, str((official_repo / "examples").resolve()))

    # PyTorch 2.6 changed torch.load's default. The processed file is generated
    # locally from the verified official HDF5, so upstream's legacy load is safe.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    from torchdiffeq import odeint
    from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
    from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleMultiTrajectory
    from NeuralMD.evaluation import (
        get_binding_collision_list_semi_flexible,
        get_ligand_collision_list,
        get_matching_list,
        get_stability_list,
    )
    from models.NeuralMD_Binding01_2nd_ODE import NeuralMD_Binding01

    return SimpleNamespace(
        torch=torch,
        odeint=odeint,
        DataLoaderMISATO=DataLoaderMISATO,
        Dataset=DatasetMISATOSemiFlexibleMultiTrajectory,
        Model=NeuralMD_Binding01,
        matching=get_matching_list,
        stability=get_stability_list,
        ligand_collision=get_ligand_collision_list,
        binding_collision=get_binding_collision_list_semi_flexible,
    )


def safe_load_checkpoint(torch, checkpoint: Path, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size != NEURALMD_ODE_CHECKPOINT_BYTES:
        raise ValueError(
            f"Checkpoint has {checkpoint.stat().st_size:,} bytes; "
            f"expected {NEURALMD_ODE_CHECKPOINT_BYTES:,}."
        )
    try:
        return torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 2.2 compatibility
        return torch.load(checkpoint, map_location=device)


def ensure_supported_device(torch, requested: str):
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        capability = torch.cuda.get_device_capability(device)
        if capability < (7, 0):
            raise RuntimeError(
                f"GPU compute capability is sm_{capability[0]}{capability[1]}. "
                "Use a Kaggle T4/L4/A100 runtime; the current Kaggle PyTorch build "
                "does not execute NeuralMD's PyG kernels on P100 (sm_60)."
            )
    return device


def _to_float_list(values) -> list[float]:
    return [float(value.detach().cpu()) if hasattr(value, "detach") else float(value) for value in values]


def build_condition(batch):
    """Build the exact conditioning tuple used by the upstream evaluator."""

    return (
        batch.ligand_x,
        batch.batch_ligand,
        batch.ligand_mass,
        batch.protein_pos[batch.mask_n],
        batch.protein_pos[batch.mask_ca],
        batch.protein_pos[batch.mask_c],
        batch.protein_backbone_residue,
        batch.batch_residue,
    )


def preflight_model(upstream, model, batch, model_args, device) -> dict:
    """Run one RHS evaluation before starting a long ODE rollout."""

    torch = upstream.torch
    batch = batch.to(device)
    trajectory = batch.ligand_trajectory_pos
    position = trajectory[:, 0, :]
    velocity = trajectory[:, 1, :] - trajectory[:, 0, :]

    # The public NeuralMD evaluator uses torch.no_grad(), not inference_mode().
    with torch.no_grad():
        acceleration, refined_velocity = model(
            torch.zeros((), dtype=torch.float32, device=device),
            (velocity, position),
            condition=build_condition(batch),
        )

    if acceleration.shape != position.shape:
        raise RuntimeError(f"Preflight acceleration {acceleration.shape} != position {position.shape}")
    if refined_velocity.shape != velocity.shape:
        raise RuntimeError(f"Preflight velocity {refined_velocity.shape} != velocity {velocity.shape}")
    if not torch.isfinite(acceleration).all() or not torch.isfinite(refined_velocity).all():
        raise RuntimeError("NeuralMD preflight produced NaN or Inf")

    return {
        "ligand_atoms": int(batch.ligand_x.numel()),
        "protein_residues": int(batch.protein_backbone_residue.numel()),
        "acceleration_abs_max": float(acceleration.abs().max().cpu()),
        "velocity_abs_max": float(refined_velocity.abs().max().cpu()),
        "step_size": model_args.NeuralMD_step_size / model_args.NeuralMD_scaling,
    }


def rollout_one(upstream, model, batch, window, model_args, device):
    torch = upstream.torch
    batch = batch.to(device)
    trajectory = batch.ligand_trajectory_pos

    position = trajectory[:, window.position_frame, :]
    velocity = (
        trajectory[:, window.velocity_to_frame, :]
        - trajectory[:, window.velocity_from_frame, :]
    )
    times = torch.arange(window.horizon + 1, dtype=torch.float32, device=device)
    times = times / model_args.NeuralMD_scaling

    condition = build_condition(batch)
    _, positions = upstream.odeint(
        model,
        (velocity, position),
        times,
        condition=condition,
        method=model_args.ODE_method,
        options={"step_size": model_args.NeuralMD_step_size / model_args.NeuralMD_scaling},
    )
    prediction = positions[1:]
    target = trajectory[:, window.target_start : window.target_stop, :].transpose(0, 1)
    if prediction.shape != target.shape:
        raise RuntimeError(f"prediction {prediction.shape} != target {target.shape}")

    error = prediction - target
    rmse = torch.linalg.vector_norm(error, dim=-1).mean(dim=1).cpu().tolist()
    mae = error.abs().sum(dim=-1).mean(dim=1).cpu().tolist()

    prediction_cpu = prediction.cpu()
    target_cpu = target.cpu()
    ligand_batch = batch.batch_ligand.cpu()
    matching = _to_float_list(upstream.matching(target_cpu, prediction_cpu, ligand_batch))
    stability = upstream.stability(target_cpu, prediction_cpu, ligand_batch)
    ligand_collision = upstream.ligand_collision(prediction_cpu, batch.ligand_x.cpu(), ligand_batch)

    residue_batch = batch.batch_residue.cpu()
    protein_batch = residue_batch.unsqueeze(0).expand(3, -1).contiguous().view(-1)
    protein_x = torch.ones(protein_batch.shape[0])
    protein_x[batch.mask_n.cpu()] = 6
    protein_x[batch.mask_ca.cpu()] = 5
    protein_x[batch.mask_c.cpu()] = 5
    binding_collision = upstream.binding_collision(
        prediction_cpu,
        batch.ligand_x.cpu(),
        batch.protein_pos.cpu(),
        protein_x,
        batch_ligand=ligand_batch,
        batch_protein=protein_batch,
    )

    pred_center = prediction_cpu.mean(dim=1)
    target_center = target_cpu.mean(dim=1)
    com_error = torch.linalg.vector_norm(pred_center - target_center, dim=-1).tolist()
    pred_rg = torch.sqrt(((prediction_cpu - pred_center[:, None]) ** 2).sum(-1).mean(-1))
    target_rg = torch.sqrt(((target_cpu - target_center[:, None]) ** 2).sum(-1).mean(-1))
    rg_error = (pred_rg - target_rg).abs().tolist()

    return {
        "mae": mae,
        "rmse": rmse,
        "matching": matching,
        "stability": stability,
        "ligand_collision": ligand_collision,
        "binding_collision": binding_collision,
        "com_error": com_error,
        "rg_error": rg_error,
        "ligand_atoms": int(batch.ligand_x.numel()),
        "protein_residues": int(batch.protein_backbone_residue.numel()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(frame_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metric_names = [
        "mae",
        "rmse",
        "matching",
        "stability",
        "ligand_collision",
        "binding_collision",
        "com_error",
        "rg_error",
    ]
    complex_rows = []
    summary_rows = []
    tasks = list(dict.fromkeys(row["task"] for row in frame_rows))
    for task in tasks:
        task_rows = [row for row in frame_rows if row["task"] == task]
        pdb_ids = list(dict.fromkeys(row["pdb_id"] for row in task_rows))
        for pdb_id in pdb_ids:
            rows = [row for row in task_rows if row["pdb_id"] == pdb_id]
            item = {
                "task": task,
                "pdb_id": pdb_id,
                "ligand_atoms": rows[0]["ligand_atoms"],
                "protein_residues": rows[0]["protein_residues"],
            }
            for metric in metric_names:
                item[f"mean_{metric}"] = float(np.mean([row[metric] for row in rows]))
                item[f"final_{metric}"] = rows[-1][metric]
            complex_rows.append(item)

        item = {"task": task, "complexes": len(pdb_ids), "frames": len(task_rows)}
        for metric in metric_names:
            item[f"mean_{metric}"] = float(np.mean([row[metric] for row in task_rows]))
        summary_rows.append(item)
    return complex_rows, summary_rows


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    metadata = verify_misato1000(args.dataset_dir, strict_size=args.strict_size)
    upstream = import_upstream(args.official_repo)
    torch = upstream.torch
    device = ensure_supported_device(torch, args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_args = published_model_args()
    model = upstream.Model(model_args).to(device)
    checkpoint = safe_load_checkpoint(torch, args.checkpoint, device)
    model.load_state_dict(checkpoint["binding_model"], strict=True)
    model.eval()

    dataset = upstream.Dataset(str(args.dataset_dir), mode="test")
    test_ids = [line.strip().upper() for line in (args.dataset_dir / "raw/test_MD.txt").read_text().splitlines() if line.strip()]
    if len(dataset) != len(test_ids):
        raise RuntimeError(f"Processed test dataset has {len(dataset)} items but split has {len(test_ids)} IDs")
    if args.limit_complexes is not None:
        if args.limit_complexes < 1:
            raise ValueError("--limit-complexes must be positive")
        limit = min(args.limit_complexes, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(limit))
        test_ids = test_ids[:limit]

    loader = upstream.DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False)
    preflight_batch = next(iter(loader))
    preflight = preflight_model(upstream, model, preflight_batch, model_args, device)
    print("[preflight] " + json.dumps(preflight, ensure_ascii=False), flush=True)

    frame_rows = []
    # Match the official evaluator. inference_mode() is intentionally avoided:
    # NeuralMD's legacy PyG layers use operations that are only guaranteed under
    # the upstream no_grad() contract.
    with torch.no_grad():
        for index, (pdb_id, batch) in enumerate(zip(test_ids, loader), start=1):
            print(f"[{index:03d}/{len(test_ids):03d}] {pdb_id}", flush=True)
            for task in args.tasks:
                window = ROLLOUT_WINDOWS[task]
                print(f"  {task}: {window.horizon} rollout steps", flush=True)
                try:
                    metrics = rollout_one(upstream, model, batch, window, model_args, device)
                except Exception as error:
                    raise RuntimeError(
                        f"NeuralMD rollout failed at complex={pdb_id}, task={task}, "
                        f"torch={torch.__version__}, cuda={torch.version.cuda}"
                    ) from error
                for step in range(window.horizon):
                    frame_rows.append(
                        {
                            "task": task,
                            "pdb_id": pdb_id,
                            "step": step + 1,
                            "target_frame": window.target_start + step,
                            "ligand_atoms": metrics["ligand_atoms"],
                            "protein_residues": metrics["protein_residues"],
                            **{name: metrics[name][step] for name in metrics if isinstance(metrics[name], list)},
                        }
                    )

    complex_rows, summary_rows = summarize(frame_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "neuralmd_frames.csv", frame_rows)
    write_csv(args.output_dir / "neuralmd_complexes.csv", complex_rows)
    write_csv(args.output_dir / "neuralmd_summary.csv", summary_rows)

    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    experiment = {
        "dataset": metadata,
        "expected_hdf5_bytes": MISATO_1000_HDF5_BYTES,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "official_repo_commit": git_commit(args.official_repo),
        "goai_repo_commit": git_commit(Path.cwd()),
        "seed": args.seed,
        "tasks": args.tasks,
        "limit_complexes": args.limit_complexes,
        "rollout_contract": rollout_contract(),
        "preflight": preflight,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
