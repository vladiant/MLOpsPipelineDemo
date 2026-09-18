"""Observable gauges report their current value on every collection (DESIGN §4.5).

Regression guard for the exporter defect (opentelemetry-exporter-prometheus
0.65b0) where a *synchronous* gauge value was dropped after the first
``collect()`` and read as absent/stale between scrapes. Observable gauges must
re-report their latest value on every collection instead.
"""

from __future__ import annotations

import re

from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry, generate_latest

from mlops_pipeline.telemetry.metrics import setup_metrics


def _gauge_value(text: str, metric: str) -> float | None:
    """Return the value of the first sample line for ``metric`` (or ``None``)."""
    match = re.search(rf"^{re.escape(metric)}\{{[^}}]*}} (\S+)$", text, re.MULTILINE)
    return float(match.group(1)) if match else None


def test_observable_gauges_persist_across_two_collections() -> None:
    # A dedicated registry keeps this collection isolated from the global one.
    registry = CollectorRegistry()
    provider = MeterProvider(metric_readers=[PrometheusMetricReader(registry=registry)])
    metrics = setup_metrics(provider)
    try:
        metrics.set_model_ready(True, model_version="v1")
        metrics.set_model_info(model_version="v1", dataset="breast_cancer")
        metrics.set_drift({"mean_radius": 0.42}, 0.42, model_version="v1")

        first = generate_latest(registry).decode()
        second = generate_latest(registry).decode()

        # All four gauges are present AND carry their value on BOTH scrapes: a
        # synchronous gauge would vanish (or read stale) on the second collect.
        for text in (first, second):
            assert _gauge_value(text, "mlops_model_ready") == 1.0
            assert _gauge_value(text, "mlops_model_info") == 1.0
            assert _gauge_value(text, "mlops_feature_drift_psi") == 0.42
            assert _gauge_value(text, "mlops_drift_psi_aggregate") == 0.42
    finally:
        provider.shutdown()


def test_observable_gauges_reflect_updated_values() -> None:
    # A new set_*() before the next collection is reflected on the next scrape.
    registry = CollectorRegistry()
    provider = MeterProvider(metric_readers=[PrometheusMetricReader(registry=registry)])
    metrics = setup_metrics(provider)
    try:
        metrics.set_model_ready(False, model_version="v1")
        metrics.set_drift({"mean_radius": 0.10}, 0.10, model_version="v1")
        first = generate_latest(registry).decode()
        assert _gauge_value(first, "mlops_model_ready") == 0.0
        assert _gauge_value(first, "mlops_drift_psi_aggregate") == 0.10

        metrics.set_model_ready(True, model_version="v1")
        metrics.set_drift({"mean_radius": 0.30}, 0.30, model_version="v1")
        second = generate_latest(registry).decode()
        assert _gauge_value(second, "mlops_model_ready") == 1.0
        assert _gauge_value(second, "mlops_drift_psi_aggregate") == 0.30
    finally:
        provider.shutdown()
