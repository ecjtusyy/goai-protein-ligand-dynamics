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
