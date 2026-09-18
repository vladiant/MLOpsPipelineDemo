"""Input validation tests (AC-6; FR-11)."""

from __future__ import annotations

from fastapi.testclient import TestClient


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
