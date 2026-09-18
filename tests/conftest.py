"""Shared pytest fixtures: tmp artifact store, trained artifact, in-process client."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mlops_pipeline.artifacts.store import ArtifactStore, LoadedArtifact
from mlops_pipeline.config import Settings
from mlops_pipeline.training.train import TrainConfig, TrainResult, train


@pytest.fixture(scope="session")
def _session_train(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, TrainResult]:
    """Train once per session (small forest for speed) into a shared tmp store."""
    root = tmp_path_factory.mktemp("artifacts_session")
    config = TrainConfig(n_estimators=25, artifacts_dir=root)
    result = train(config)
    return root, result


@pytest.fixture
def tmp_artifacts_dir(tmp_path: Path) -> Path:
    """An empty artifact-store root."""
    root = tmp_path / "artifacts"
    root.mkdir()
    return root


@pytest.fixture
def train_config(tmp_artifacts_dir: Path) -> TrainConfig:
    """A fast training config writing into a fresh tmp store."""
    return TrainConfig(n_estimators=25, artifacts_dir=tmp_artifacts_dir)


@pytest.fixture
def trained_artifact(_session_train: tuple[Path, TrainResult]) -> LoadedArtifact:
    """A loaded artifact produced by the session-scoped training run."""
    root, result = _session_train
    return ArtifactStore(root).load(result.version_id)


@pytest.fixture
def artifacts_root(_session_train: tuple[Path, TrainResult]) -> Path:
    """Root of the session-scoped artifact store (has a valid latest pointer)."""
    root, _ = _session_train
    return root


@pytest.fixture
def client(artifacts_root: Path) -> Iterator[object]:
    """A TestClient bound to an app loaded against the session artifact store."""
    from fastapi.testclient import TestClient

    from mlops_pipeline.serving.app import create_app

    settings = Settings(artifacts_dir=artifacts_root, drift_refresh_seconds=3600.0)
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_features(trained_artifact: LoadedArtifact) -> dict[str, float]:
    """A valid feature map built from each feature's baseline mean."""
    return {feat.name: feat.mean for feat in trained_artifact.baseline.features}
