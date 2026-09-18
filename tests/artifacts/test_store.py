"""Artifact store round-trip tests (AC-4; FR-8/FR-9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlops_pipeline.artifacts.store import ArtifactNotFoundError, ArtifactStore
from mlops_pipeline.training.train import TrainConfig, train


def test_save_load_round_trip_and_latest(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    result = train(TrainConfig(n_estimators=25, artifacts_dir=root))
    store = ArtifactStore(root)

    by_id = store.load(result.version_id)
    by_latest = store.load(None)  # None → latest pointer (FR-9)

    assert by_id.metadata.version_id == by_latest.metadata.version_id == result.version_id
    assert by_id.metadata.feature_names == by_latest.metadata.feature_names
    assert store.resolve_latest() == result.version_id


def test_missing_version_raises(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ArtifactNotFoundError):
        store.load("does-not-exist")


def test_load_without_pointer_raises(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "empty")
    with pytest.raises(ArtifactNotFoundError):
        store.load(None)


def test_save_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    result = train(TrainConfig(n_estimators=25, artifacts_dir=root))
    loaded = store.load(result.version_id)

    with pytest.raises(FileExistsError):
        store.save(loaded.model, loaded.metadata, loaded.baseline)
