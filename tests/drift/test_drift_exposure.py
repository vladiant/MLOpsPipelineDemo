"""Drift exposure: gauge update + drift_detected log event (AC-10; FR-20)."""

from __future__ import annotations

import asyncio
import logging

from fastapi.testclient import TestClient

from mlops_pipeline.artifacts.store import LoadedArtifact
from mlops_pipeline.config import Settings
from mlops_pipeline.drift.monitor import DriftMonitor
from mlops_pipeline.serving.app import _drift_refresh_loop
from mlops_pipeline.serving.state import AppState


def _shifted(artifact: LoadedArtifact) -> dict[str, float]:
    # Push every feature well beyond its training max → concentrates in the last bin.
    return {feat.name: feat.max + 10.0 * (feat.std + 1.0) for feat in artifact.baseline.features}


def test_drift_aggregate_gauge_exposed_on_metrics(
    client: TestClient, trained_artifact: LoadedArtifact
) -> None:
    # FR-20 (metric): after significant drift is computed and pushed, the aggregate
    # PSI gauge is present on /metrics with a value in the significant band.
    state: AppState = client.app.state.app_state  # type: ignore[attr-defined]
    assert state.drift_monitor is not None and state.metrics is not None

    shifted = _shifted(trained_artifact)
    for _ in range(150):
        state.drift_monitor.record(shifted)

    report = state.drift_monitor.compute()
    assert report.status == "significant"
    assert report.aggregate >= 0.25

    state.metrics.set_drift(
        report.per_feature, report.aggregate, model_version=state.model_version or "v"
    )
    text = client.get("/metrics").text
    assert "mlops_drift_psi_aggregate" in text
    assert "mlops_feature_drift_psi" in text


def test_drift_detected_log_event_emitted(trained_artifact: LoadedArtifact, caplog) -> None:
    # FR-20 (log): a single structured drift_detected event is emitted when status
    # transitions away from "none".
    monitor = DriftMonitor(trained_artifact.baseline, window_size=500, min_samples=50)
    shifted = _shifted(trained_artifact)
    for _ in range(80):
        monitor.record(shifted)

    class _FakeMetrics:
        def set_drift(self, per_feature, aggregate, *, model_version) -> None:  # noqa: ANN001
            return None

    state = AppState()
    state.drift_monitor = monitor
    state.metrics = _FakeMetrics()  # type: ignore[assignment]
    state.model_version = "v-test"

    settings = Settings(drift_refresh_seconds=0.0)

    async def _run_once() -> None:
        task = asyncio.create_task(_drift_refresh_loop(state, settings))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    with caplog.at_level(logging.WARNING, logger="mlops_pipeline"):
        asyncio.run(_run_once())

    events = [r for r in caplog.records if getattr(r, "event", None) == "drift_detected"]
    assert events, "expected a drift_detected log event"
    assert events[-1].status == "significant"
    assert hasattr(events[-1], "aggregate_psi")
    assert events[-1].model_version == "v-test"
