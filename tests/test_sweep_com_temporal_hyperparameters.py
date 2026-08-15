import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import sweep_com_temporal_hyperparameters as SWEEP


def args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "sweep",
        device="cpu",
        coarse_seed=42,
        verification_seeds=[42, 43, 44],
        coarse_epochs=5,
        full_epochs=9,
        coarse_patience=2,
        full_patience=4,
        top_k=2,
        hidden_dim=8,
        rbf_channels=4,
        protein_cutoff=12.0,
        weight_decay=1e-5,
        gradient_clip=1.0,
        success_threshold_pct=5.0,
    )


def write_result(request: SWEEP.RunRequest, mean: float, final: float) -> None:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_epoch": 3,
        "mean_point_improvement_pct": mean,
        "final_point_improvement_pct": final,
        "model_gate_pass": mean >= 5.0 and final >= 5.0,
    }
    with (request.output_dir / "com_temporal_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    for name in (
        "history.csv",
        "best_model.pth",
        "latest.pth",
        "com_temporal_decision.csv",
    ):
        (request.output_dir / name).write_text("test\n")


def test_two_stage_sweep_selects_top_configs_and_is_resumable(tmp_path: Path) -> None:
    parameters = args(tmp_path)
    calls = []
    coarse_score = {
        config.config_id: index + 1.0
        for index, config in enumerate(SWEEP.SWEEP_CONFIGS)
    }

    def fake_runner(request, _):
        calls.append(request)
        base = coarse_score[request.config.config_id]
        if request.stage == "coarse":
            write_result(request, base, base + 0.2)
        else:
            seed_bonus = {42: 0.0, 43: 0.4, 44: -0.3}[request.seed]
            write_result(request, base + seed_bonus, base + 0.2 + seed_bonus)

    payload = SWEEP.execute_sweep(parameters, runner=fake_runner)

    assert len(calls) == 12
    selected = json.loads(
        (parameters.output_root / "selected_configs.json").read_text()
    )["selected_config_ids"]
    assert selected == [
        SWEEP.SWEEP_CONFIGS[-1].config_id,
        SWEEP.SWEEP_CONFIGS[-2].config_id,
    ]
    assert payload["decision"]["decision"] == "PASS_STABLE"
    assert payload["decision"]["best_seed"] == 43
    assert (parameters.output_root / "sweep_improvements.png").is_file()

    calls.clear()
    SWEEP.execute_sweep(parameters, runner=fake_runner)
    assert calls == []

    parameters.full_epochs += 1
    with pytest.raises(ValueError, match="different experiment contract"):
        SWEEP.execute_sweep(parameters, runner=fake_runner)


def test_aggregate_requires_a_majority_of_seeds() -> None:
    config = SWEEP.SWEEP_CONFIGS[0]
    rows = [
        {
            "config_id": config.config_id,
            "mean_point_improvement_pct": value,
            "final_point_improvement_pct": value + 0.1,
            "min_gate_improvement_pct": value,
        }
        for value in (5.2, 4.9, 4.8)
    ]

    aggregate = SWEEP._aggregate_configs(rows, threshold=5.0)[0]

    assert aggregate["passing_seeds"] == 1
    assert aggregate["stable_gate_pass"] is False
