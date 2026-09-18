"""Input validation tests (AC-6; FR-11)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlops_pipeline.artifacts.store import LoadedArtifact
from mlops_pipeline.serving.predictor import FeatureValidationError, Predictor


def test_missing_feature_returns_422(client: TestClient, sample_features: dict[str, float]) -> None:
    payload = dict(sample_features)
    payload.pop(next(iter(payload)))
    resp = client.post("/predict", json={"features": payload})
    assert resp.status_code == 422
    assert "missing" in resp.text.lower()


def test_extra_feature_returns_422(client: TestClient, sample_features: dict[str, float]) -> None:
    payload = dict(sample_features)
    payload["not_a_feature"] = 1.0
    resp = client.post("/predict", json={"features": payload})
    assert resp.status_code == 422
    assert "unexpected" in resp.text.lower()


def test_non_numeric_feature_returns_422(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    payload = dict(sample_features)
    first = next(iter(payload))
    payload[first] = "not-a-number"  # type: ignore[assignment]
    resp = client.post("/predict", json={"features": payload})
    # Pydantic rejects the non-float at the schema boundary (still a 422).
    assert resp.status_code == 422


def test_predictor_rejects_non_finite_value(trained_artifact: LoadedArtifact) -> None:
    # FR-11 predictor path: NaN/inf slip past JSON typing but must be rejected.
    predictor = Predictor(trained_artifact)
    features = {feat.name: feat.mean for feat in trained_artifact.baseline.features}
    first = next(iter(features))
    for bad in (float("nan"), float("inf"), float("-inf")):
        features[first] = bad
        with pytest.raises(FeatureValidationError, match="finite"):
            predictor.validate(features)


def test_predictor_rejects_bool_value(trained_artifact: LoadedArtifact) -> None:
    # FR-11: bool is an int subtype but is not a valid numeric feature.
    predictor = Predictor(trained_artifact)
    features: dict[str, object] = {
        feat.name: feat.mean for feat in trained_artifact.baseline.features
    }
    first = next(iter(features))
    features[first] = True
    with pytest.raises(FeatureValidationError, match="numeric"):
        predictor.validate(features)  # type: ignore[arg-type]
