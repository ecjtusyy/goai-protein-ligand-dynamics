import json
from pathlib import Path


def test_kaggle_notebook_is_valid_and_compilable() -> None:
    path = Path("notebooks/01_misato100_end_to_end.ipynb")
    notebook = json.loads(path.read_text())

    assert notebook["nbformat"] == 4
    assert any("MISATO_100" in "".join(cell["source"]) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
