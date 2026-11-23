"""Pydantic v2 request/response models for the serving API (DESIGN §4.6).

@ingroup serving
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "ReadyResponse",
]


class PredictRequest(BaseModel):
    """Single-record prediction request keyed by feature name (order-independent)."""

    features: dict[str, float] = Field(
        ..., description="30 named breast-cancer features → float values."
    )


class PredictResponse(BaseModel):
    """Prediction response with the model version that produced it (FR-10/FR-12)."""

    request_id: str
    model_version: str
    predicted_class: str
    predicted_index: int
    probabilities: dict[str, float] | None = Field(
        default=None, description="Class name → probability (FR-10)."
    )


class HealthResponse(BaseModel):
    """Liveness response."""

    status: str


class ReadyResponse(BaseModel):
    """Readiness response; ``ready`` is false until a valid artifact is loaded (NFR-8)."""

    ready: bool
    model_version: str | None = None
    reason: str | None = None
