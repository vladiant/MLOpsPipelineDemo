"""Training-time baseline capture: per-feature quantile histograms (FR-6)."""

from __future__ import annotations

import numpy as np

from ..artifacts.schema import SCHEMA_VERSION, Baseline, FeatureBaseline

__all__ = ["compute_baseline"]


def compute_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    version_id: str,
    *,
    n_bins: int = 10,
) -> Baseline:
    """Compute the training baseline (class balance + per-feature decile histograms).

    Each feature is bucketed into ``n_bins`` quantile bins; the stored edges let the
    live side bucket identically. ``bin_proportions`` is the expected PSI distribution
    and sums to 1.0.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train)
    n_samples = int(X.shape[0])

    class_balance = _class_balance(y, target_names)

    features: list[FeatureBaseline] = []
    for col, name in enumerate(feature_names):
        column = X[:, col]
        edges = _quantile_edges(column, n_bins)
        proportions = _proportions(column, edges)
        features.append(
            FeatureBaseline(
                name=name,
                bin_edges=[float(e) for e in edges],
                bin_proportions=[float(p) for p in proportions],
                min=float(np.min(column)),
                max=float(np.max(column)),
                mean=float(np.mean(column)),
                std=float(np.std(column)),
            )
        )

    return Baseline(
        schema_version=SCHEMA_VERSION,
        version_id=version_id,
        n_samples=n_samples,
        class_balance=class_balance,
        features=features,
    )


def _class_balance(y: np.ndarray, target_names: list[str]) -> dict[str, float]:
    total = int(y.shape[0])
    balance: dict[str, float] = {}
    for index, name in enumerate(target_names):
        balance[name] = float(np.count_nonzero(y == index) / total) if total else 0.0
    return balance


def _quantile_edges(column: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(column, quantiles)
    # De-duplicate collapsed edges (constant/degenerate features) while staying monotone.
    edges = np.unique(edges)
    if edges.size < 2:
        span = max(abs(float(edges[0])), 1.0)
        edges = np.array([float(edges[0]) - span, float(edges[0]) + span])
    return edges


def _proportions(column: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(column, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(counts.shape, 1.0 / counts.size)
    return counts / total
