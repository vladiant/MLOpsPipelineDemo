"""FastAPI application factory with artifact-loading lifespan (DESIGN §4.7)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI

from ..artifacts.store import ArtifactNotFoundError, ArtifactStore
from ..config import Settings, get_settings
from ..drift.monitor import DriftMonitor
from ..telemetry.logging import configure_logging, get_logger
from ..telemetry.otel import setup_telemetry
from .predictor import Predictor
from .routes import router
from .state import AppState

__all__ = ["create_app"]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app; the lifespan loads the artifact and wires telemetry."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="MLOpsPipelineDemo serving",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.app_state = AppState()
    app.include_router(router)
    return app


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    state: AppState = app.state.app_state
    logger = get_logger()

    metrics = setup_telemetry(app, settings)
    state.metrics = metrics

    store = ArtifactStore(settings.artifacts_dir)
    try:
        artifact = store.load(settings.artifact_version)
    except (ArtifactNotFoundError, FileNotFoundError) as exc:
        state.not_ready_reason = str(exc)
        logger.warning(
            "artifact_load_failed",
            extra={"event": "artifact_load_failed", "reason": str(exc)},
        )
        yield
        return

    state.predictor = Predictor(artifact)
    state.drift_monitor = DriftMonitor(
        artifact.baseline,
        window_size=settings.drift_window_size,
        min_samples=settings.drift_min_samples,
    )
    state.mark_ready(
        model_version=artifact.metadata.version_id,
        dataset_name=artifact.metadata.dataset_name,
    )
    metrics.set_model_ready(True, model_version=artifact.metadata.version_id)
    metrics.set_model_info(
        model_version=artifact.metadata.version_id,
        dataset=artifact.metadata.dataset_name,
    )
    logger.info(
        "artifact_loaded",
        extra={"event": "artifact_loaded", "model_version": artifact.metadata.version_id},
    )

    refresh_task = asyncio.create_task(_drift_refresh_loop(state, settings))
    try:
        yield
    finally:
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresh_task


async def _drift_refresh_loop(state: AppState, settings: Settings) -> None:
    """Periodically recompute PSI and update the drift gauges (DESIGN §6.3)."""
    logger = get_logger()
    while True:
        await asyncio.sleep(settings.drift_refresh_seconds)
        if state.drift_monitor is None or state.metrics is None or state.model_version is None:
            continue
        report = state.drift_monitor.compute()
        state.metrics.set_drift(
            report.per_feature, report.aggregate, model_version=state.model_version
        )
        if report.status != "none" and report.status != state._drift_status:
            logger.warning(
                "drift_detected",
                extra={
                    "event": "drift_detected",
                    "aggregate_psi": report.aggregate,
                    "status": report.status,
                    "model_version": state.model_version,
                },
            )
        state._drift_status = report.status
