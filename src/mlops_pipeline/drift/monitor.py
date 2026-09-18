"""Rolling-window drift monitor computing per-feature and aggregate PSI (FR-19/FR-20)."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from ..artifacts.schema import Baseline
from .psi import bucket_proportions, classify_psi, psi_from_proportions

__all__ = ["DriftMonitor", "DriftReport"]


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Result of a drift computation over the current window."""

    per_feature: dict[str, float]
    aggregate: float
    status: str
    n_samples: int


class DriftMonitor:
    """Thread-safe rolling window of live feature vectors vs. the training baseline.

    Cold-start guard: below ``min_samples`` collected, :meth:`compute` returns a
    zeroed report with status ``none`` so a cold service emits no drift noise.
    """

    def __init__(
        self, baseline: Baseline, *, window_size: int = 500, min_samples: int = 100
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self._baseline = baseline
        self._feature_names = [feat.name for feat in baseline.features]
        self._min_samples = min_samples
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = {
            name: deque(maxlen=window_size) for name in self._feature_names
        }

    def record(self, features: Mapping[str, float]) -> None:
        """Append one feature vector to the rolling window (thread-safe)."""
        with self._lock:
            for name in self._feature_names:
                if name in features:
                    self._windows[name].append(float(features[name]))

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the window."""
        with self._lock:
            if not self._feature_names:
                return 0
            return len(self._windows[self._feature_names[0]])

    def compute(self) -> DriftReport:
        """Re-bucket the current window against the baseline and compute PSI."""
        with self._lock:
            snapshot = {name: list(self._windows[name]) for name in self._feature_names}
            n_samples = len(snapshot[self._feature_names[0]]) if self._feature_names else 0

        if n_samples < self._min_samples:
            return DriftReport(
                per_feature={name: 0.0 for name in self._feature_names},
                aggregate=0.0,
                status=classify_psi(0.0),
                n_samples=n_samples,
            )

        per_feature: dict[str, float] = {}
        for feat in self._baseline.features:
            actual = bucket_proportions(snapshot[feat.name], feat.bin_edges)
            per_feature[feat.name] = psi_from_proportions(feat.bin_proportions, actual)

        aggregate = sum(per_feature.values()) / len(per_feature) if per_feature else 0.0
        return DriftReport(
            per_feature=per_feature,
            aggregate=aggregate,
            status=classify_psi(aggregate),
            n_samples=n_samples,
        )
