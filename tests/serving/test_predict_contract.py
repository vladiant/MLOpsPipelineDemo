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


def test_metrics_expose_error_counter_and_model_info(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    # AC-9 / FR-17/FR-12: a bad request increments the error counter; model_info
    # and the success/error request counters are all exposed with the mlops_ prefix.
    client.post("/predict", json={"features": sample_features})  # 200 path
    bad = dict(sample_features)
    bad.pop(next(iter(bad)))
    client.post("/predict", json={"features": bad})  # 422 path → error counter

    text = client.get("/metrics").text
    assert "mlops_prediction_errors" in text
    assert 'error_type="validation"' in text
    assert "mlops_model_info" in text
    assert "mlops_model_ready" in text
    # Every project metric carries the mlops_ namespace prefix (NFR-1).
    metric_lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert any(line.startswith("mlops_") for line in metric_lines)
    assert all(
        not (line.startswith("commerce_") or line.startswith("app_")) for line in metric_lines
    )


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


def test_prediction_log_serializes_to_json_with_timestamp(capsys) -> None:
    # AC-8 / FR-14: the emitted record is valid JSON carrying timestamp + all
    # required fields (the JSON formatter path, not just LogRecord attributes).
    import json

    from mlops_pipeline.telemetry.logging import configure_logging, log_prediction

    configure_logging("INFO")  # binds the JSON handler to the captured stdout
    log_prediction(
        request_id="req-123",
        endpoint="/predict",
        status=200,
        latency_ms=1.25,
        model_version="v-test",
    )
    line = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(line)  # must be valid JSON
    for field in (
        "timestamp",
        "request_id",
        "endpoint",
        "status",
        "latency_ms",
        "model_version",
    ):
        assert field in record, f"missing {field} in JSON log record"
    assert record["request_id"] == "req-123"
    assert record["status"] == 200
    assert record["model_version"] == "v-test"
