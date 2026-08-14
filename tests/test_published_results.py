import csv
from pathlib import Path


RESULTS = Path("results")


def _read_csv(filename: str) -> list[dict[str, str]]:
    with (RESULTS / filename).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_full_summary_contains_all_tasks_and_complexes() -> None:
    rows = _read_csv("neuralmd_summary_seed42.csv")

    assert [row["task"] for row in rows] == ["paper", "T1", "T2", "T3"]
    assert all(int(row["complexes"]) == 100 for row in rows)
    t3 = rows[-1]
    assert int(t3["frames"]) == 8_000
    assert abs(float(t3["mean_rmse"]) - 4.1561420764364305) < 1e-12


def test_diagnosis_supports_the_reported_failure_mode() -> None:
    rows = _read_csv("neuralmd_weakness_diagnosis_seed42.csv")
    t3 = next(row for row in rows if row["task"] == "T3")

    assert float(t3["rmse_growth_ratio"]) > 4.6
    assert float(t3["stability_drop"]) > 27.0
    assert float(t3["final_com_error"]) > 5.0
    assert float(t3["final_rg_error"]) < 0.3


def test_failure_curve_is_a_nonempty_png() -> None:
    image = RESULTS / "neuralmd_failure_curves.png"

    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image.stat().st_size > 100_000
