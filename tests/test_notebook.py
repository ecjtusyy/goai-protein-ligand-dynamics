import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/neuralmd_misato1000.ipynb")


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def test_notebook_is_valid_compilable_and_executed() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    serialized = json.dumps(notebook)

    assert notebook["nbformat"] == 4
    assert "MISATO_1000" in serialized

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 9
    for index, cell in enumerate(code_cells):
        compile(_source(cell), f"notebook-cell-{index}", "exec")
        assert not any(output.get("output_type") == "error" for output in cell.get("outputs", []))

    # 这些值来自完整 100-complex 运行，防止误提交未执行的模板。
    assert "100/100" in serialized
    assert "4.1561420764364305" in serialized
    assert "image/png" in serialized


def test_notebook_pins_neuralmd_runtime_contract() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    source = "\n".join(_source(cell) for cell in notebook["cells"])

    assert "torch_geometric==2.5.3" in source
    assert "torch_scatter==2.1.2" in source
    assert "torch_cluster==1.6.3" in source
    assert "--force-reinstall" in source
    assert "--no-deps" in source
    assert "PyG radius_graph CUDA probe: OK" in source
    assert '"-m", "scripts.run_official_neuralmd"' in source
    assert "RUNNER = GOAI" not in source
