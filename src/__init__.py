"""Contracts and evaluation helpers for the official NeuralMD baseline."""

from .neuralmd_official import (
    MISATO_1000_HDF5_BYTES,
    MISATO_1000_SPLIT_COUNTS,
    NEURALMD_ODE_CHECKPOINT_BYTES,
    ROLLOUT_WINDOWS,
    rollout_contract,
    verify_misato1000,
)

__all__ = [
    "MISATO_1000_HDF5_BYTES",
    "MISATO_1000_SPLIT_COUNTS",
    "NEURALMD_ODE_CHECKPOINT_BYTES",
    "ROLLOUT_WINDOWS",
    "rollout_contract",
    "verify_misato1000",
]
