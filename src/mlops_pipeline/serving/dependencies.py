"""FastAPI dependency providers reading from ``app.state`` (DESIGN §4.7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request, status

if TYPE_CHECKING:
    from ..drift.monitor import DriftMonitor
    from ..telemetry.metrics import Metrics
    from .predictor import Predictor
    from .state import AppState

__all__ = [
    "get_app_state",
    "get_drift_monitor",
    "get_metrics",
    "get_predictor",
]


def get_app_state(request: Request) -> AppState:
    """Return the shared :class:`AppState` stored on the app."""
    return request.app.state.app_state


def get_predictor(state: AppState = Depends(get_app_state)) -> Predictor:
    """Return the loaded predictor, or 503 if the service is not ready (NFR-8)."""
    if state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifact not loaded",
        )
    return state.predictor


def get_drift_monitor(state: AppState = Depends(get_app_state)) -> DriftMonitor:
    """Return the drift monitor, or 503 if the service is not ready."""
    if state.drift_monitor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model artifact not loaded",
        )
    return state.drift_monitor


def get_metrics(state: AppState = Depends(get_app_state)) -> Metrics:
    """Return the telemetry instruments."""
    if state.metrics is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="telemetry not initialized",
        )
    return state.metrics
