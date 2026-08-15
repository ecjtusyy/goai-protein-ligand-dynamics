import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/neuralmd_probabilistic_residual.ipynb"


def notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_notebook_has_smoke_before_full_and_pinned_code() -> None:
    source = notebook_source()

    assert 'GOAI_COMMIT = "78cbebfafed119a4e61bd66146a5235dff67cac5"' in source
    assert source.index("Smoke A") < source.index("Full：")
    assert 'cache_split("train", limit=SMOKE_COMPLEXES)' in source
    assert 'cache_split("val", limit=SMOKE_COMPLEXES)' in source
    assert 'cache_split("train")' in source
    assert 'cache_split("val")' in source


def test_notebook_runs_all_ablation_contracts_and_publication_gate() -> None:
    source = notebook_source()

    for variant in ("ode_mu", "ode_mu_sigma", "ode_temporal_mu_sigma"):
        assert variant in source
    assert "scripts.cache_neuralmd_residuals" in source
    assert "scripts.train_probabilistic_residual" in source
    assert "scripts.evaluate_probabilistic_residual" in source
    assert "NeuralMD_SDE/MISATO_1000_seed_42/model.pth" in source
    assert '"--dynamics", "sde"' in source
    assert "neuralmd_sde_seed42_single_sample" in source
    assert 'print(f"[REUSE] {variant}: 已完成 {latest[\'epoch\']} epochs")' in source
    assert 'if variant == "ode_mu":' in source
    assert "history[nll_columns].isna().all().all()" in source
    assert "np.isfinite(nll_values).all()" in source
    assert 'for forbidden in ("*.pth", "*.pt", "*.npz", "*.docx", "*.pdf")' in source
    assert "3-complex 结果只用于排错" in source


def test_notebook_is_unexecuted_instead_of_faking_results() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
