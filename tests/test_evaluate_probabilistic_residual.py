import inspect

from scripts import evaluate_probabilistic_residual as EVALUATOR


def metric_values(value: float) -> dict:
    return {
        "ligand_atoms": 2,
        "protein_residues": 3,
        **{name: [value] * 80 for name in EVALUATOR.METRIC_NAMES},
    }


def test_comparison_summary_keeps_baseline_and_corrected_separate() -> None:
    rows = []
    rows.extend(EVALUATOR.frame_metric_rows("neuralmd_ode", "1ABC", metric_values(2.0)))
    rows.extend(EVALUATOR.frame_metric_rows("ode_mu", "1ABC", metric_values(1.0)))

    complexes, summary = EVALUATOR.summarize_comparison(rows)

    assert len(complexes) == 2
    assert [row["method"] for row in summary] == ["neuralmd_ode", "ode_mu"]
    assert summary[0]["final_rmse"] == 2.0
    assert summary[1]["mean_rmse"] == 1.0


def test_evaluator_uses_only_unseen_test_for_final_comparison() -> None:
    source = inspect.getsource(EVALUATOR.main)

    assert 'mode="test"' in source
    assert "test_MD.txt" in source
    assert "shuffle=False" in source
    assert "with torch.no_grad():" in source
    assert "optimizer" not in source


def test_cli_accepts_multiple_residual_checkpoints() -> None:
    arguments = [
        "--official-repo",
        "/tmp/NeuralMD",
        "--dataset-dir",
        "/tmp/MISATO_1000",
        "--ode-checkpoint",
        "/tmp/ode.pth",
        "--residual-checkpoint",
        "/tmp/mu.pth",
        "--residual-checkpoint",
        "/tmp/temporal.pth",
        "--output-dir",
        "/tmp/results",
    ]

    parsed = EVALUATOR.parse_args(arguments)

    assert [path.name for path in parsed.residual_checkpoint] == ["mu.pth", "temporal.pth"]
