"""Serving service: FastAPI app factory, models, predictor, and routes.

@defgroup serving Serving
@ingroup components
@brief FastAPI app factory, request/response models, predictor, and route handlers.
"""

from __future__ import annotations

from .app import create_app
from .models import HealthResponse, PredictRequest, PredictResponse, ReadyResponse
from .predictor import FeatureValidationError, PredictionOutcome, Predictor

__all__ = [
    "FeatureValidationError",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
    "PredictionOutcome",
    "Predictor",
    "ReadyResponse",
    "create_app",
]
