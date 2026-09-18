"""Drift monitor tests (AC-10; FR-19/FR-20)."""

from __future__ import annotations

from mlops_pipeline.artifacts.store import LoadedArtifact
from mlops_pipeline.drift.monitor import DriftMonitor


def _mean_features(artifact: LoadedArtifact) -> dict[str, float]:
    return {feat.name: feat.mean for feat in artifact.baseline.features}


def test_below_min_samples_reports_none(trained_artifact: LoadedArtifact) -> None:
    monitor = DriftMonitor(trained_artifact.baseline, window_size=500, min_samples=100)
    for _ in range(10):
        monitor.record(_mean_features(trained_artifact))
    report = monitor.compute()
    assert report.status == "none"
    assert report.aggregate == 0.0
    assert report.n_samples == 10


def test_window_is_bounded(trained_artifact: LoadedArtifact) -> None:
    monitor = DriftMonitor(trained_artifact.baseline, window_size=50, min_samples=10)
    for _ in range(200):
        monitor.record(_mean_features(trained_artifact))
    assert monitor.sample_count == 50


def test_shifted_inputs_raise_aggregate(trained_artifact: LoadedArtifact) -> None:
    baseline = trained_artifact.baseline
    monitor = DriftMonitor(baseline, window_size=500, min_samples=50)

    # Feed values pushed well above each feature's max → concentrates in the last bin.
    shifted = {feat.name: feat.max + 10.0 * (feat.std + 1.0) for feat in baseline.features}
    for _ in range(200):
        monitor.record(shifted)

    report = monitor.compute()
    assert report.n_samples == 200
    assert report.aggregate > 0.25
    assert report.status == "significant"
