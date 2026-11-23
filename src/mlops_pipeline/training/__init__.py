"""Training pipeline: dataset loading, baseline capture, and the train orchestrator.

@defgroup training Training
@ingroup components
@brief Dataset loading, training baseline capture, and the train orchestrator.
"""

from __future__ import annotations

from .baseline import compute_baseline
from .dataset import LoadedDataset, load_dataset
from .train import TrainConfig, TrainResult, train

__all__ = [
    "LoadedDataset",
    "TrainConfig",
    "TrainResult",
    "compute_baseline",
    "load_dataset",
    "train",
]
