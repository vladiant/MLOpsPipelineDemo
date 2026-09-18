"""Artifact-version selection at serving time (AC-4; FR-9)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mlops_pipeline.config import Settings
from mlops_pipeline.serving.app import create_app
from mlops_pipeline.training.train import TrainConfig, train


def test_serving_loads_specified_non_latest_version(tmp_path: Path) -> None:
    # FR-9: a specific artifact version is loadable via config even when it is not
    # the latest pointer. Train two versions; latest → v2; explicitly serve v1.
    root = tmp_path / "artifacts"
    v1 = train(TrainConfig(n_estimators=25, seed=1, artifacts_dir=root)).version_id
    v2 = train(TrainConfig(n_estimators=25, seed=2, artifacts_dir=root)).version_id
    assert v1 != v2

    settings = Settings(artifacts_dir=root, artifact_version=v1, drift_refresh_seconds=3600.0)
    app = create_app(settings)
    with TestClient(app) as client:
        body = client.get("/readyz").json()
        assert body["ready"] is True
        # The served version is the explicitly-selected one, not the latest (v2).
        assert body["model_version"] == v1


def test_serving_defaults_to_latest_when_version_unspecified(tmp_path: Path) -> None:
    # FR-9: unspecified version → the latest pointer (here v2, the newest run).
    root = tmp_path / "artifacts"
    train(TrainConfig(n_estimators=25, seed=1, artifacts_dir=root))
    v2 = train(TrainConfig(n_estimators=25, seed=2, artifacts_dir=root)).version_id

    settings = Settings(artifacts_dir=root, drift_refresh_seconds=3600.0)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/readyz").json()["model_version"] == v2
