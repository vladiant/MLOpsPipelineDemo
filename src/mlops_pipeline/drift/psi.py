"""Population Stability Index (PSI) — pure, I/O-free functions (DESIGN §4.4/§6.2).

@ingroup drift
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "DRIFT_MODERATE",
    "DRIFT_NONE",
    "DRIFT_SIGNIFICANT",
    "bucket_proportions",
    "classify_psi",
    "psi_from_proportions",
]

DRIFT_NONE = "none"  # PSI < 0.1
DRIFT_MODERATE = "moderate"  # 0.1 <= PSI < 0.25
DRIFT_SIGNIFICANT = "significant"  # PSI >= 0.25

_MODERATE_THRESHOLD = 0.10
_SIGNIFICANT_THRESHOLD = 0.25


def bucket_proportions(values: Sequence[float], bin_edges: Sequence[float]) -> list[float]:
    """Bucket ``values`` into the half-open bins defined by ``bin_edges``.

    ``bin_edges`` has ``k + 1`` entries defining ``k`` bins. Values below the first
    edge fall in the first bin and values at/above the last edge fall in the last
    bin (edges from training may not cover the live range). Returns proportions that
    sum to 1.0 (or all zeros when ``values`` is empty).
    """
    if len(bin_edges) < 2:
        raise ValueError("bin_edges must define at least one bin (>= 2 edges)")

    n_bins = len(bin_edges) - 1
    counts = [0] * n_bins
    for value in values:
        counts[_bin_index(value, bin_edges, n_bins)] += 1

    total = len(values)
    if total == 0:
        return [0.0] * n_bins
    return [count / total for count in counts]


def psi_from_proportions(
    expected: Sequence[float], actual: Sequence[float], *, eps: float = 1e-6
) -> float:
    """Compute PSI = sum((a - e) * ln(a / e)) with an ``eps`` floor on empty bins."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same number of bins")
    if not expected:
        raise ValueError("cannot compute PSI over zero bins")

    total = 0.0
    for exp, act in zip(expected, actual, strict=True):
        e = max(exp, eps)
        a = max(act, eps)
        total += (a - e) * math.log(a / e)
    return total


def classify_psi(psi: float) -> str:
    """Map a PSI value to ``none`` / ``moderate`` / ``significant`` (DESIGN §6.2)."""
    if psi < _MODERATE_THRESHOLD:
        return DRIFT_NONE
    if psi < _SIGNIFICANT_THRESHOLD:
        return DRIFT_MODERATE
    return DRIFT_SIGNIFICANT


def _bin_index(value: float, bin_edges: Sequence[float], n_bins: int) -> int:
    if value < bin_edges[1]:
        return 0
    for i in range(1, n_bins):
        if value < bin_edges[i + 1]:
            return i
    return n_bins - 1
