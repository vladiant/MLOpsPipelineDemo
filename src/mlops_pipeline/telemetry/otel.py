"""Wire the OTel MeterProvider + Prometheus exporter and mount ``/metrics`` (FR-15)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import Metrics, setup_metrics

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..config import Settings

__all__ = ["build_meter_provider", "setup_telemetry"]


def build_meter_provider(settings: Settings) -> MeterProvider:
    """Build an SDK MeterProvider backed by a Prometheus pull exporter.

    An OTLP push exporter is added only when ``settings.otlp_endpoint`` is set, so
    tests and local runs work without a collector (R-1: pins kept minimal/compatible).
    """
    resource = Resource.create({SERVICE_NAME: settings.service_name})
    readers: list[object] = [PrometheusMetricReader()]

    if settings.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=settings.otlp_endpoint, insecure=True)
            )
        )

    return MeterProvider(resource=resource, metric_readers=readers)  # type: ignore[arg-type]


def setup_telemetry(app: FastAPI, settings: Settings) -> Metrics:
    """Configure metrics, set the global provider, mount ``/metrics``, return instruments."""
    provider = build_meter_provider(settings)
    otel_metrics.set_meter_provider(provider)
    metrics = setup_metrics(provider)

    from fastapi import Response

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:  # noqa: D401 - simple exposition route
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.state.meter_provider = provider
    return metrics
