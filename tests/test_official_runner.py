import importlib.util
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).parents[1]
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "src.neuralmd_official", ROOT / "src/neuralmd_official.py"
)
assert CONTRACT_SPEC and CONTRACT_SPEC.loader
CONTRACT = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = CONTRACT
CONTRACT_SPEC.loader.exec_module(CONTRACT)

SRC = ModuleType("src")
SRC.__path__ = []
SRC.neuralmd_official = CONTRACT
sys.modules["src"] = SRC

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_official_neuralmd", ROOT / "scripts/run_official_neuralmd.py"
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = RUNNER
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_published_model_args_match_checkpoint_hyperparameters() -> None:
    args = RUNNER.published_model_args()

    assert args.FrameNet_num_radial == 100
    assert args.NeuralMD_step_size == 5
    assert args.NeuralMD_scaling == 100
    assert args.NeuralMD_velocity_refined_value_coefficient == 0.01
    assert args.use_MLP_velocity is False


def test_summarize_preserves_task_and_final_frame() -> None:
    metrics = {
        "mae": 1.0,
        "rmse": 2.0,
        "matching": 3.0,
        "stability": 90.0,
        "ligand_collision": 4.0,
        "binding_collision": 5.0,
        "com_error": 6.0,
        "rg_error": 7.0,
    }
    rows = [
        {
            "task": "T1",
            "pdb_id": "1ABC",
            "step": step,
            "target_frame": 9 + step,
            "ligand_atoms": 20,
            "protein_residues": 200,
            **(metrics | {"rmse": float(step)}),
        }
        for step in (1, 2)
    ]

    complexes, summary = RUNNER.summarize(rows)

    assert complexes[0]["mean_rmse"] == 1.5
    assert complexes[0]["final_rmse"] == 2.0
    assert summary[0]["complexes"] == 1
    assert summary[0]["frames"] == 2
