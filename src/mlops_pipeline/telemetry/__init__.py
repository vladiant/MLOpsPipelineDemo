"""Telemetry: OTel metrics instruments, Prometheus wiring, and JSON logging.

@defgroup telemetry Telemetry
@ingroup components
@brief OpenTelemetry metric instruments, Prometheus wiring, and structured JSON logging.
"""

from __future__ import annotations

from .logging import configure_logging, get_logger, log_prediction
from .metrics import Metrics, setup_metrics
from .otel import build_meter_provider, setup_telemetry

__all__ = [
    "Metrics",
    "build_meter_provider",
    "configure_logging",
    "get_logger",
    "log_prediction",
    "setup_metrics",
    "setup_telemetry",
]
