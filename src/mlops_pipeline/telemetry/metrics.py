"""OpenTelemetry metric instruments with the ``mlops_`` prefix (DESIGN §5).

Gauges use OTel *observable* gauges (``create_observable_gauge`` + callbacks) so
their current value is re-reported on every collection/scrape. The Prometheus
exporter (0.65b0) drops a *synchronous* gauge value after the first ``collect()``,
which made ``mlops_model_ready``/``mlops_model_info``/drift PSI read as stale
between updates; observable gauges match DESIGN §4.5 and fix that.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from opentelemetry.metrics import (
    CallbackOptions,
    Counter,
    Histogram,
    Meter,
    MeterProvider,
    Observation,
)

__all__ = ["Metrics", "setup_metrics"]

# Sub-200 ms latency buckets (seconds) for p50/p95/p99 (FR-16, NFR-2).
_LATENCY_BUCKETS_SECONDS = (
    0.001,
    0.002,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.200,
    0.500,
)


class _ObservableGaugeStore:
    """Thread-safe latest-observation store feeding the observable gauge callbacks.

    The serving lifespan / drift-refresh loop push new values via the ``set_*``
    methods; the observable-gauge callbacks read the stored observations on every
    collection, so each scrape reports the current value instead of a stale one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_ready: list[Observation] = []
        self._model_info: list[Observation] = []
        self._drift: list[Observation] = []
        self._drift_aggregate: list[Observation] = []

    def set_model_ready(self, ready: bool, *, model_version: str) -> None:
        with self._lock:
            self._model_ready = [Observation(1 if ready else 0, {"model_version": model_version})]

    def set_model_info(self, *, model_version: str, dataset: str) -> None:
        with self._lock:
            self._model_info = [
                Observation(1, {"model_version": model_version, "dataset": dataset})
            ]

    def set_drift(
        self, per_feature: Mapping[str, float], aggregate: float, *, model_version: str
    ) -> None:
        with self._lock:
            self._drift = [
                Observation(value, {"feature": feature, "model_version": model_version})
                for feature, value in per_feature.items()
            ]
            self._drift_aggregate = [Observation(aggregate, {"model_version": model_version})]

    def observe_model_ready(self, options: CallbackOptions) -> Iterable[Observation]:
        with self._lock:
            return list(self._model_ready)

    def observe_model_info(self, options: CallbackOptions) -> Iterable[Observation]:
        with self._lock:
            return list(self._model_info)

    def observe_drift(self, options: CallbackOptions) -> Iterable[Observation]:
        with self._lock:
            return list(self._drift)

    def observe_drift_aggregate(self, options: CallbackOptions) -> Iterable[Observation]:
        with self._lock:
            return list(self._drift_aggregate)


@dataclass
class Metrics:
    """Holds the OTel instruments used across the serving service."""

    meter: Meter
    request_counter: Counter
    error_counter: Counter
    latency_hist: Histogram
    predicted_class_counter: Counter
    drift_gauge: object  # ObservableGauge (mlops_feature_drift_psi)
    drift_aggregate_gauge: object  # ObservableGauge (mlops_drift_psi_aggregate)
    model_ready_gauge: object  # ObservableGauge (mlops_model_ready)
    model_info_gauge: object  # ObservableGauge (mlops_model_info)
    _store: _ObservableGaugeStore = field(default_factory=_ObservableGaugeStore, repr=False)

    def record_request(self, *, endpoint: str, status_code: int, model_version: str) -> None:
        self.request_counter.add(
            1,
            {
                "endpoint": endpoint,
                "status_code": str(status_code),
                "model_version": model_version,
            },
        )

    def record_error(self, *, endpoint: str, error_type: str) -> None:
        self.error_counter.add(1, {"endpoint": endpoint, "error_type": error_type})

    def record_latency(self, seconds: float, *, endpoint: str, model_version: str) -> None:
        self.latency_hist.record(seconds, {"endpoint": endpoint, "model_version": model_version})

    def record_predicted_class(self, *, predicted_class: str, model_version: str) -> None:
        self.predicted_class_counter.add(
            1, {"predicted_class": predicted_class, "model_version": model_version}
        )

    def set_model_ready(self, ready: bool, *, model_version: str) -> None:
        self._store.set_model_ready(ready, model_version=model_version)

    def set_model_info(self, *, model_version: str, dataset: str) -> None:
        self._store.set_model_info(model_version=model_version, dataset=dataset)

    def set_drift(
        self, per_feature: Mapping[str, float], aggregate: float, *, model_version: str
    ) -> None:
        self._store.set_drift(per_feature, aggregate, model_version=model_version)


def setup_metrics(meter_provider: MeterProvider) -> Metrics:
    """Create all ``mlops_`` instruments from a configured meter provider."""
    meter = meter_provider.get_meter("mlops_pipeline")
    store = _ObservableGaugeStore()

    request_counter = meter.create_counter(
        "mlops_prediction_requests",
        unit="1",
        description="Total prediction requests by endpoint/status/model version.",
    )
    error_counter = meter.create_counter(
        "mlops_prediction_errors",
        unit="1",
        description="Total prediction errors by endpoint/error type.",
    )
    latency_hist = meter.create_histogram(
        "mlops_inference_latency_seconds",
        unit="s",
        description="Inference request latency (p50/p95/p99).",
        explicit_bucket_boundaries_advisory=list(_LATENCY_BUCKETS_SECONDS),
    )
    predicted_class_counter = meter.create_counter(
        "mlops_predicted_class",
        unit="1",
        description="Predicted-class distribution counts.",
    )
    drift_gauge = meter.create_observable_gauge(
        "mlops_feature_drift_psi",
        callbacks=[store.observe_drift],
        unit="1",
        description="Per-feature PSI vs. training baseline.",
    )
    drift_aggregate_gauge = meter.create_observable_gauge(
        "mlops_drift_psi_aggregate",
        callbacks=[store.observe_drift_aggregate],
        unit="1",
        description="Mean PSI across features.",
    )
    model_ready_gauge = meter.create_observable_gauge(
        "mlops_model_ready",
        callbacks=[store.observe_model_ready],
        unit="1",
        description="1 when a valid artifact is loaded, else 0.",
    )
    model_info_gauge = meter.create_observable_gauge(
        "mlops_model_info",
        callbacks=[store.observe_model_info],
        unit="1",
        description="Constant 1 info metric carrying model version/dataset labels.",
    )

    return Metrics(
        meter=meter,
        request_counter=request_counter,
        error_counter=error_counter,
        latency_hist=latency_hist,
        predicted_class_counter=predicted_class_counter,
        drift_gauge=drift_gauge,
        drift_aggregate_gauge=drift_aggregate_gauge,
        model_ready_gauge=model_ready_gauge,
        model_info_gauge=model_info_gauge,
        _store=store,
    )
