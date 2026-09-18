"""Environment-driven configuration (12-factor) via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Runtime settings for the serving service.

    All values are overridable with ``MLOPS_``-prefixed environment variables,
    e.g. ``MLOPS_ARTIFACTS_DIR`` or ``MLOPS_ARTIFACT_VERSION``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MLOPS_",
        env_file=".env",
        extra="ignore",
    )

    artifacts_dir: Path = Field(
        default=Path("artifacts"),
        description="Root directory of the artifact store.",
    )
    artifact_version: str | None = Field(
        default=None,
        description="Specific artifact version-id to serve; None → latest pointer (FR-9).",
    )
    host: str = Field(default="0.0.0.0", description="Bind host for uvicorn.")
    port: int = Field(default=8000, description="Bind port for uvicorn.")
    log_level: str = Field(default="INFO", description="Root log level.")
    service_name: str = Field(
        default="mlops-serving",
        description="OpenTelemetry service.name resource attribute.",
    )
    otlp_endpoint: str | None = Field(
        default=None,
        description="Optional OTLP/gRPC collector endpoint; None disables the OTLP exporter.",
    )
    drift_window_size: int = Field(
        default=500, description="Rolling window size for drift (FR-19)."
    )
    drift_min_samples: int = Field(
        default=100, description="Minimum window fill before PSI is reported."
    )
    drift_refresh_seconds: float = Field(
        default=15.0, description="Interval for the background drift refresh tick."
    )


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance read from the environment."""
    return Settings()
