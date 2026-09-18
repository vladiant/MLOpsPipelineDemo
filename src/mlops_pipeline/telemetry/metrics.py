"""OpenTelemetry metric instruments with the ``mlops_`` prefix (DESIGN §5)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from opentelemetry.metrics import Counter, Histogram, Meter, MeterProvider

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


@dataclass
class Metrics:
    """Holds the OTel instruments used across the serving service."""

    meter: Meter
    request_counter: Counter
    error_counter: Counter
    latency_hist: Histogram
    predicted_class_counter: Counter
    drift_gauge: object  # synchronous Gauge (mlops_feature_drift_psi)
    drift_aggregate_gauge: object  # synchronous Gauge (mlops_drift_psi_aggregate)
    model_ready_gauge: object  # synchronous Gauge (mlops_model_ready)
    model_info_gauge: object  # synchronous Gauge (mlops_model_info)

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
        self.model_ready_gauge.set(  # type: ignore[attr-defined]
            1 if ready else 0, {"model_version": model_version}
        )

    def set_model_info(self, *, model_version: str, dataset: str) -> None:
        self.model_info_gauge.set(  # type: ignore[attr-defined]
            1, {"model_version": model_version, "dataset": dataset}
        )

    def set_drift(
        self, per_feature: Mapping[str, float], aggregate: float, *, model_version: str
    ) -> None:
        for feature, value in per_feature.items():
            self.drift_gauge.set(  # type: ignore[attr-defined]
                value, {"feature": feature, "model_version": model_version}
            )
        self.drift_aggregate_gauge.set(  # type: ignore[attr-defined]
            aggregate, {"model_version": model_version}
        )


def setup_metrics(meter_provider: MeterProvider) -> Metrics:
    """Create all ``mlops_`` instruments from a configured meter provider."""
    meter = meter_provider.get_meter("mlops_pipeline")

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
    drift_gauge = meter.create_gauge(
        "mlops_feature_drift_psi",
        unit="1",
        description="Per-feature PSI vs. training baseline.",
    )
    drift_aggregate_gauge = meter.create_gauge(
        "mlops_drift_psi_aggregate",
        unit="1",
        description="Mean PSI across features.",
    )
    model_ready_gauge = meter.create_gauge(
        "mlops_model_ready",
        unit="1",
        description="1 when a valid artifact is loaded, else 0.",
    )
    model_info_gauge = meter.create_gauge(
        "mlops_model_info",
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
    )
