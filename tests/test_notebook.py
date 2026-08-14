import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("filename", "required_text"),
    [
        ("01_misato100_end_to_end.ipynb", "MISATO_100"),
        ("02_official_neuralmd_misato1000.ipynb", "MISATO_1000"),
    ],
)
def test_kaggle_notebook_is_valid_and_compilable(filename: str, required_text: str) -> None:
    path = Path("notebooks") / filename
    notebook = json.loads(path.read_text())

    assert notebook["nbformat"] == 4
    assert any(required_text in "".join(cell["source"]) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_official_notebook_pins_neuralmd_runtime_contract() -> None:
    path = Path("notebooks/02_official_neuralmd_misato1000.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert 'torch_geometric==2.5.3' in source
    assert 'torch_scatter==2.1.2' in source
    assert 'torch_cluster==1.6.3' in source
    assert 'PyG CUDA probe: OK' in source
    assert 'PyG dataset cache contract: OK' in source
    assert '"-m", "scripts.run_official_neuralmd"' in source
    assert 'RUNNER = GOAI' not in source
