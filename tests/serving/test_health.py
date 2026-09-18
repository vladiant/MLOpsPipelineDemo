"""Health/readiness tests (AC-7; FR-13/NFR-8)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mlops_pipeline.config import Settings
from mlops_pipeline.serving.app import create_app


def test_healthz_always_ok(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_ready_with_artifact(client: TestClient) -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["model_version"]


def test_readyz_not_ready_without_artifact(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    app = create_app(Settings(artifacts_dir=empty_root, drift_refresh_seconds=3600.0))
    with TestClient(app) as client:
        resp = client.get("/readyz")
        assert resp.status_code == 503
        body = resp.json()
        assert body["ready"] is False
        assert body["reason"]

        # Prediction is refused while not ready (NFR-8 gating).
        assert client.post("/predict", json={"features": {}}).status_code == 503
