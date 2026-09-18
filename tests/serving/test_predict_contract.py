"""Prediction contract + metrics surface tests (AC-5, AC-9; FR-10/12/15..18)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mlops_pipeline.artifacts.store import LoadedArtifact


def test_valid_prediction_returns_200_with_contract(
    client: TestClient, sample_features: dict[str, float], artifacts_root
) -> None:
    resp = client.post("/predict", json={"features": sample_features})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in {"malignant", "benign"}
    assert isinstance(body["probabilities"], dict)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert body["model_version"]
    assert body["request_id"]


def test_metrics_endpoint_exposes_mlops_metrics(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    # Drive one prediction so counters/histograms are populated.
    client.post("/predict", json={"features": sample_features})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "mlops_inference_latency_seconds" in text
    assert "mlops_prediction_requests" in text
    assert "mlops_predicted_class" in text


def test_predict_proba_column_order_matches_metadata(
    trained_artifact: LoadedArtifact,
) -> None:
    # R-3 invariant: model.classes_ order == metadata.target_classes order.
    model = trained_artifact.model
    target_names = ["malignant", "benign"]  # sklearn breast_cancer target order
    expected = [target_names[int(c)] for c in model.classes_]
    assert trained_artifact.metadata.target_classes == expected


def test_prediction_log_has_required_fields(
    client: TestClient, sample_features: dict[str, float], caplog
) -> None:
    # AC-8 / FR-14: one structured record with the required keys.
    import logging

    with caplog.at_level(logging.INFO, logger="mlops_pipeline"):
        client.post("/predict", json={"features": sample_features})

    records = [r for r in caplog.records if getattr(r, "event", None) == "prediction"]
    assert records, "expected a prediction log record"
    rec = records[-1]
    for field in ("request_id", "endpoint", "status", "latency_ms", "model_version"):
        assert hasattr(rec, field)
