"""Drift detection: PSI math and the rolling-window monitor.

@defgroup drift Drift
@ingroup components
@brief Population Stability Index math and the rolling-window drift monitor.
"""

from __future__ import annotations

from .monitor import DriftMonitor, DriftReport
from .psi import (
    DRIFT_MODERATE,
    DRIFT_NONE,
    DRIFT_SIGNIFICANT,
    bucket_proportions,
    classify_psi,
    psi_from_proportions,
)

__all__ = [
    "DRIFT_MODERATE",
    "DRIFT_NONE",
    "DRIFT_SIGNIFICANT",
    "DriftMonitor",
    "DriftReport",
    "bucket_proportions",
    "classify_psi",
    "psi_from_proportions",
]
