"""Route handlers: predict / healthz / readyz (DESIGN §4.7/§4.9)."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, Response, status

from ..drift.monitor import DriftMonitor
from ..telemetry.logging import log_prediction
from ..telemetry.metrics import Metrics
from .dependencies import get_app_state, get_drift_monitor, get_metrics, get_predictor
from .models import HealthResponse, PredictRequest, PredictResponse, ReadyResponse
from .predictor import FeatureValidationError, Predictor
from .state import AppState

__all__ = ["router"]

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    predictor: Predictor = Depends(get_predictor),
    monitor: DriftMonitor = Depends(get_drift_monitor),
    metrics: Metrics = Depends(get_metrics),
) -> PredictResponse:
    """Validate, predict, record drift + metrics, and log (FR-10/11/12/14/16/17/18)."""
    request_id = str(uuid.uuid4())
    endpoint = "/predict"
    model_version = predictor.model_version
    start = time.perf_counter()

    try:
        outcome = predictor.predict(request.features)
    except FeatureValidationError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics.record_error(endpoint=endpoint, error_type="validation")
        metrics.record_request(endpoint=endpoint, status_code=422, model_version=model_version)
        metrics.record_latency(latency_ms / 1000.0, endpoint=endpoint, model_version=model_version)
        log_prediction(
            request_id=request_id,
            endpoint=endpoint,
            status=422,
            latency_ms=latency_ms,
            model_version=model_version,
        )
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    monitor.record(request.features)
    latency_ms = (time.perf_counter() - start) * 1000.0

    metrics.record_latency(latency_ms / 1000.0, endpoint=endpoint, model_version=model_version)
    metrics.record_request(endpoint=endpoint, status_code=200, model_version=model_version)
    metrics.record_predicted_class(
        predicted_class=outcome.predicted_class, model_version=model_version
    )
    log_prediction(
        request_id=request_id,
        endpoint=endpoint,
        status=200,
        latency_ms=latency_ms,
        model_version=model_version,
    )

    return PredictResponse(
        request_id=request_id,
        model_version=model_version,
        predicted_class=outcome.predicted_class,
        predicted_index=outcome.predicted_index,
        probabilities=outcome.probabilities,
    )


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness — always 200 while the process runs (FR-13)."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadyResponse)
async def readyz(
    response: Response,
    state: AppState = Depends(get_app_state),
) -> ReadyResponse:
    """Readiness — 503 until a valid artifact is loaded (FR-13/NFR-8)."""
    if not state.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(ready=False, reason=state.not_ready_reason)
    return ReadyResponse(ready=True, model_version=state.model_version)
