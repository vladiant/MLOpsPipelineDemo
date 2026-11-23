"""Dataset loading and train/test splitting for the Breast-Cancer demo (FR-1).

@ingroup training
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

__all__ = ["LoadedDataset", "load_dataset"]


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A loaded, split dataset with feature and target names."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_dataset(name: str, *, seed: int, test_size: float) -> LoadedDataset:
    """Load a supported public dataset and return a stratified train/test split.

    Args:
        name: Dataset key; only ``"breast_cancer"`` is supported this iteration.
        seed: Random seed for the split (reproducibility, FR-7).
        test_size: Held-out fraction for evaluation.

    Raises:
        ValueError: If ``name`` is not a supported dataset.
    """
    if name != "breast_cancer":
        raise ValueError(f"unsupported dataset {name!r}; only 'breast_cancer' is supported")

    bunch = load_breast_cancer()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.int64)
    feature_names = [str(n) for n in bunch.feature_names]
    target_names = [str(n) for n in bunch.target_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return LoadedDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
    )
