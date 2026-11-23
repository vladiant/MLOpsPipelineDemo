"""Structured JSON logging to stdout (FR-14).

@ingroup telemetry
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

__all__ = ["configure_logging", "get_logger", "log_prediction"]

_LOGGER_NAME = "mlops_pipeline"
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install a JSON formatter on the package logger, emitting to stdout."""
    global _CONFIGURED

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
        json_ensure_ascii=False,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    _CONFIGURED = True


def get_logger() -> logging.Logger:
    """Return the package logger, configuring it with defaults if needed."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(_LOGGER_NAME)


def log_prediction(
    *,
    request_id: str,
    endpoint: str,
    status: int,
    latency_ms: float,
    model_version: str,
) -> None:
    """Emit one structured JSON log record for a prediction request (FR-14)."""
    get_logger().info(
        "prediction",
        extra={
            "event": "prediction",
            "request_id": request_id,
            "endpoint": endpoint,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "model_version": model_version,
        },
    )
