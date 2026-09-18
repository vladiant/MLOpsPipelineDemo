"""Mutable application state shared across request handlers."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..drift.monitor import DriftMonitor
from ..telemetry.metrics import Metrics
from .predictor import Predictor

__all__ = ["AppState"]


@dataclass
class AppState:
    """Holds the loaded predictor, drift monitor, metrics and readiness flag."""

    predictor: Predictor | None = None
    drift_monitor: DriftMonitor | None = None
    metrics: Metrics | None = None
    model_version: str | None = None
    dataset_name: str | None = None
    ready: bool = False
    not_ready_reason: str | None = "artifact not loaded"
    _drift_status: str = field(default="none", repr=False)

    def mark_ready(self, *, model_version: str, dataset_name: str) -> None:
        self.ready = True
        self.model_version = model_version
        self.dataset_name = dataset_name
        self.not_ready_reason = None
