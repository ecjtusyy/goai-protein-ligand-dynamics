import inspect

import numpy as np
import pytest

from scripts import cache_neuralmd_residuals as RUNNER


REQUIRED = [
    "--official-repo",
    "/tmp/NeuralMD",
    "--dataset-dir",
    "/tmp/MISATO_1000",
    "--checkpoint",
    "/tmp/model.pth",
    "--output-dir",
    "/tmp/cache",
]


def test_cli_makes_test_split_unrepresentable() -> None:
    with pytest.raises(SystemExit):
        RUNNER.parse_args(REQUIRED + ["--split", "test"])


def test_resume_and_overwrite_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        RUNNER.parse_args(REQUIRED + ["--split", "train", "--resume", "--overwrite"])


def test_t3_cache_targets_frames_20_through_99() -> None:
    frames = RUNNER.target_frames("T3")

    np.testing.assert_array_equal(frames, np.arange(20, 100))


def test_runner_freezes_neuralmd_and_uses_no_grad() -> None:
    source = inspect.getsource(RUNNER.main)

    assert "parameter.requires_grad_(False)" in source
    assert "with torch.no_grad():" in source
    assert "mode=args.split" in source
    assert "shuffle=False" in source
    assert "args.output_dir / args.split" in source
