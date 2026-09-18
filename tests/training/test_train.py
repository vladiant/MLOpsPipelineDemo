"""Training → artifact tests (AC-1, AC-2; FR-1..FR-7)."""

from __future__ import annotations

from pathlib import Path

from mlops_pipeline.artifacts.store import ArtifactStore
from mlops_pipeline.training.train import TrainConfig, train


def test_train_writes_valid_versioned_artifact(train_config: TrainConfig) -> None:
    # AC-1 / FR-1..FR-6: a versioned artifact dir with all three files + full metadata.
    result = train(train_config)

    artifact_dir = train_config.artifacts_dir / result.version_id
    assert (artifact_dir / "model.joblib").is_file()
    assert (artifact_dir / "metadata.json").is_file()
    assert (artifact_dir / "baseline.json").is_file()

    loaded = ArtifactStore(train_config.artifacts_dir).load(result.version_id)
    meta = loaded.metadata
    assert meta.version_id == result.version_id
    assert meta.dataset_name == "breast_cancer"
    assert meta.random_seed == train_config.seed
    assert len(meta.feature_schema) == 30
    assert set(meta.metrics) >= {"accuracy", "f1", "precision", "recall"}
    assert meta.library_versions["scikit-learn"]
    assert meta.model_sha256

    # latest.txt points at the new version (FR-9).
    assert ArtifactStore(train_config.artifacts_dir).resolve_latest() == result.version_id


def test_baseline_persisted_with_artifact(train_config: TrainConfig) -> None:
    # AC-3 / FR-6: baseline has class balance + per-feature histograms summing to 1.
    result = train(train_config)
    baseline = ArtifactStore(train_config.artifacts_dir).load(result.version_id).baseline

    assert len(baseline.features) == 30
    assert abs(sum(baseline.class_balance.values()) - 1.0) < 1e-9
    for feat in baseline.features:
        assert abs(sum(feat.bin_proportions) - 1.0) < 1e-9
        assert feat.bin_edges == sorted(feat.bin_edges)


def test_reproducible_metrics_and_no_overwrite(tmp_path: Path) -> None:
    # AC-2 / FR-4, FR-7: same seed → exact-equal metrics; two runs → two artifacts.
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    result_a = train(TrainConfig(n_estimators=25, artifacts_dir=root_a))
    result_b = train(TrainConfig(n_estimators=25, artifacts_dir=root_b))

    # R-5: fixed seed + pinned sklearn → exact equality (no tolerance needed).
    assert result_a.metrics == result_b.metrics

    # Two runs into the same store keep both artifacts (immutability, FR-4).
    shared = tmp_path / "shared"
    v1 = train(TrainConfig(n_estimators=25, seed=1, artifacts_dir=shared)).version_id
    v2 = train(TrainConfig(n_estimators=25, seed=2, artifacts_dir=shared)).version_id
    assert v1 != v2
    versions = ArtifactStore(shared).list_versions()
    assert v1 in versions and v2 in versions
