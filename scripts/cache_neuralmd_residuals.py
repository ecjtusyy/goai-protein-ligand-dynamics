"""用冻结的官方 NeuralMD-ODE 为 train/val 生成 T3 残差训练缓存。"""

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
)
from src.neuralmd_official import ROLLOUT_WINDOWS, verify_misato1000
from src.residual_cache import (
    TRAINING_SPLITS,
    build_cache_payload,
    read_training_split_ids,
    validate_complex_cache,
    write_complex_cache,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=TRAINING_SPLITS, required=True)
    parser.add_argument("--task", choices=ROLLOUT_WINDOWS, default="T3")
    parser.add_argument("--limit-complexes", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-size", action=argparse.BooleanOptionalAction, default=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def target_frames(task: str) -> np.ndarray:
    window = ROLLOUT_WINDOWS[task]
    return np.arange(window.target_start, window.target_stop, dtype=np.int64)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.limit_complexes is not None and args.limit_complexes < 1:
        raise ValueError("--limit-complexes must be positive")

    dataset_metadata = verify_misato1000(args.dataset_dir, strict_size=args.strict_size)
    split_ids = read_training_split_ids(args.dataset_dir, args.split)
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = upstream.Dataset(str(args.dataset_dir), mode=args.split)
    if len(dataset) != len(split_ids):
        raise RuntimeError(
            f"Processed {args.split} dataset has {len(dataset)} items but split has "
            f"{len(split_ids)} IDs"
        )
    if args.limit_complexes is not None:
        limit = min(args.limit_complexes, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(limit))
        split_ids = split_ids[:limit]

    loader = upstream.DataLoaderMISATO(dataset, batch_size=1, num_workers=0, shuffle=False)
    preflight = preflight_model(upstream, model, next(iter(loader)), model_args, device)
    print("[preflight] " + json.dumps(preflight, ensure_ascii=False), flush=True)

    window = ROLLOUT_WINDOWS[args.task]
    frames = target_frames(args.task)
    split_output = args.output_dir / args.split
    written_files = []
    # 与官方评估器一致使用 no_grad；缓存只生成标签，不构建反向传播图。
    with torch.no_grad():
        for index, (pdb_id, batch) in enumerate(zip(split_ids, loader), start=1):
            print(f"[{index:03d}/{len(split_ids):03d}] {args.split}/{pdb_id}", flush=True)
            destination = split_output / "complexes" / f"{pdb_id}.npz"
            if args.resume and destination.is_file():
                validate_complex_cache(destination, frames)
                print("  已校验并跳过现有缓存", flush=True)
                written_files.append(str(destination.relative_to(split_output)))
                continue
            try:
                device_batch, prediction, target = predict_trajectory(
                    upstream, model, batch, window, model_args, device
                )
                payload = build_cache_payload(device_batch, prediction, target, frames)
                destination = write_complex_cache(
                    split_output,
                    pdb_id,
                    payload,
                    overwrite=args.overwrite,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Residual cache failed at split={args.split}, complex={pdb_id}, "
                    f"task={args.task}"
                ) from error
            written_files.append(str(destination.relative_to(split_output)))

    manifest = {
        "schema_version": 1,
        "split": args.split,
        "task": args.task,
        "complexes": len(written_files),
        "files": written_files,
        "target_frames": frames.tolist(),
        "dataset": dataset_metadata,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "official_repo_commit": git_commit(args.official_repo),
        "goai_repo_commit": git_commit(REPO_ROOT),
        "seed": args.seed,
        "limit_complexes": args.limit_complexes,
        "resumed": args.resume,
        "preflight": preflight,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    split_output.mkdir(parents=True, exist_ok=True)
    manifest_path = split_output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "complexes": len(written_files)}, indent=2))


if __name__ == "__main__":
    main()
