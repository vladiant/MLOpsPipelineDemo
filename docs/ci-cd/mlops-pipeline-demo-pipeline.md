# MLOpsPipelineDemo — CI/CD & Local Observability Pipeline

This document describes the deployment and CI layer added at the Release stage:
the container image, the local `docker-compose` observability stack, and the
GitHub Actions pipeline. It is the reproduce-it-locally reference for reviewers.

- **Reference pattern:** `CommerceSystemDemo` (metrics-first, provisioned-as-code Grafana).
- **Scope decisions:** metrics only — no Tempo/Loki, no alert rules (PM OQ-6, OQ-8).

---

## 1. Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage, slim, non-root image for the serving service. |
| `docker-entrypoint.sh` | Trains a model on first start if the store is empty, then serves. |
| `.dockerignore` | Keeps the build context small/reproducible. |
| `docker-compose.yaml` | Brings up serving + OTel Collector + Prometheus + Grafana. |
| `configs/prometheus/prometheus.yml` | Scrape config (service `/metrics` + collector). |
| `configs/otel-collector-config.yaml` | Minimal OTLP→Prometheus metrics pipeline. |
| `configs/grafana/provisioning/datasources/datasource.yml` | Prometheus datasource. |
| `configs/grafana/provisioning/dashboards/dashboards.yml` | Dashboard provider. |
| `configs/grafana/provisioning/dashboards/mlops-overview.json` | "MLOps Overview" dashboard. |
| `constraints.txt` | QA-validated version pins (R-1, NFR-3). |
| `.github/workflows/ci.yml` | Lint / type-check / test / build pipeline. |

---

## 2. Container image (`Dockerfile`)

- **Multi-stage:** stage 1 builds an isolated `/opt/venv` (package installed against
  `constraints.txt`); stage 2 is a `python:3.12-slim` runtime that copies only the venv.
- **Non-root:** runs as uid/gid `10001` (`app`). `/app/artifacts` is owned by that user
  so a Docker **named volume** mounted there is writable with no manual `chown`.
- **First-run training:** `docker-entrypoint.sh` runs `mlops-train --artifacts-dir
  $MLOPS_ARTIFACTS_DIR` only when `latest.txt` is absent, then `exec mlops-serve`.
  This is the chosen "bake vs. mount" answer: **the image bakes no model; it trains
  one on first start into the mounted volume.** Rationale: keeps the image small,
  reproducible from a clean checkout, and lets a reviewer keep the trained model
  across restarts via the `mlops-artifacts` volume.
- **Healthcheck:** hits `GET /readyz` with stdlib `urllib` (no `curl` in the slim image);
  readiness only flips true once a valid artifact is loaded (NFR-8).

### Build / run standalone

```bash
docker build -t mlops-pipeline-demo:local .
docker run -p 8000:8000 mlops-pipeline-demo:local     # trains on first start, then serves
```

### Point at a pre-trained artifact instead of first-run training

Train on the host, then bind-mount the store (uid must be able to read it):

```bash
mlops-train                                            # writes ./artifacts/<version>/
docker run -p 8000:8000 \
  -e MLOPS_ARTIFACTS_DIR=/app/artifacts \
  -e MLOPS_ARTIFACT_VERSION=<version-id> \             # optional; omit → latest pointer
  -v "$PWD/artifacts:/app/artifacts" \
  mlops-pipeline-demo:local
```

---

## 3. Local stack (`docker-compose up`)

```bash
docker compose up --build          # one command, no manual post-setup (AC-12, NFR-4)
docker compose down                # stop
docker compose down -v             # stop and wipe the trained-model volume
```

| Service | Image | Host port | Role |
|---------|-------|-----------|------|
| `serving` | built from `Dockerfile` | `8000` | FastAPI app + `/metrics` |
| `otel-collector` | `otel/opentelemetry-collector-contrib:0.111.0` | `4317`, `8889` | OTLP ingest → re-expose metrics |
| `prometheus` | `prom/prometheus:v2.55.1` | `9090` | scrapes service + collector |
| `grafana` | `grafana/grafana:11.3.0` | `3000` | provisioned dashboard |

All services share the `mlops` bridge network. The service pushes OTLP metrics to the
collector (`MLOPS_OTLP_ENDPOINT=otel-collector:4317`) **and** exposes the authoritative
pull `/metrics`; Prometheus scrapes both targets every 5 s.

### URLs

- Service: <http://localhost:8000> — `/predict`, `/healthz`, `/readyz`, `/metrics`, `/docs`
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> — anonymous admin, dashboard **MLOps Overview**
  (uid `mlops-overview`, direct link <http://localhost:3000/d/mlops-overview>)

### Send a prediction (drives the dashboards)

```bash
curl -s http://localhost:8000/readyz
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"features": {"mean radius": 14.1, "mean texture": 20.2, ... }}'   # all 30 named features
```

---

## 4. Metrics → dashboard mapping

Rendered Prometheus names (OTel exporter appends `_total` to counters and
`_bucket/_sum/_count` to histograms):

| Metric | Dashboard panel | Requirement |
|--------|-----------------|-------------|
| `mlops_inference_latency_seconds_bucket` | Inference latency p95 / p50 | FR-16, NFR-2, AC-14 |
| `mlops_prediction_errors_total`, `mlops_prediction_requests_total` | Error rate | FR-17 |
| `mlops_predicted_class_total` | Predicted-class distribution | FR-18, FR-22 |
| `mlops_drift_psi_aggregate`, `mlops_feature_drift_psi` | Drift — PSI | FR-19, FR-20, FR-22 |
| `mlops_model_ready` | Model readiness | FR-13, NFR-8 |
| `mlops_prediction_requests_total` | Throughput by status | FR-17 |

p95 query:
`histogram_quantile(0.95, sum(rate(mlops_inference_latency_seconds_bucket{endpoint="/predict"}[$__rate_interval])) by (le))`

> Drift gauges (`mlops_feature_drift_psi`, `mlops_drift_psi_aggregate`) only appear once
> the rolling window reaches `MLOPS_DRIFT_MIN_SAMPLES` (default 100) predictions and the
> background refresh tick has run, so the drift panel is empty on a cold service.

### 4.1 Known limitation — synchronous gauges (defect flagged for the Developer/QA)

The app instruments `mlops_model_ready`, `mlops_model_info`, `mlops_drift_psi_aggregate`,
and `mlops_feature_drift_psi` with OpenTelemetry **synchronous** `Gauge` instruments
(`create_gauge`, `telemetry/metrics.py`). Verified live behavior with
`opentelemetry-exporter-prometheus 0.65b0`: a synchronous-gauge LastValue data point is
**dropped after the first `collect()`** and only reappears if `set()` runs again before the
next collect. Because Prometheus scrapes every 5 s (each scrape is a `collect()`), these
gauges are consumed between updates. Observed impact:

- Counter/histogram panels (latency p95, error rate, throughput, predicted-class) — **fully correct**.
- Drift PSI panels — populate at the refresh cadence (~15 s); one scrape per refresh catches the value.
- `mlops_model_ready` / `mlops_model_info` — set only once at startup, so Prometheus captures a single sample.

**Mitigation applied here (config-only, no product change):** the drift and readiness panels
wrap their queries in `last_over_time(...)` so the last captured value is carried forward
across the gaps. This keeps the dashboard readable without touching `src/`.

**Recommended upstream fix (product code — out of Release scope):** switch these four
instruments from synchronous `create_gauge` to **`ObservableGauge`** with observer callbacks
(as DESIGN §4.5 already anticipated: `ObservableGauge|Gauge`). Observable gauges are re-read
on every `collect()`, so values persist across scrapes with no `last_over_time` workaround.
This is a small, well-scoped change for the Python Developer / QA to make and re-verify.

> Metrics flow via the pull path only. OTLP metric push to the collector is intentionally
> disabled in compose so the same `mlops_` counters are scraped from a single target
> (the service); scraping them again off the collector's `:8889` exposition would
> double-count them in dashboard `sum()`s. The collector stays up for reference parity
> and future OTLP traces.


---

## 5. GitHub Actions (`.github/workflows/ci.yml`)

Triggered on push/PR to `main`.

- **`quality`** (matrix Python **3.11** and **3.12**):
  `actions/setup-python` with `cache: pip` keyed on `pyproject.toml` + `constraints.txt`
  → `pip install -e ".[dev]" -c constraints.txt` → `ruff check .` → `ruff format --check .`
  → `mypy src` → `pytest` (coverage XML + `--cov-report=term-missing` + JUnit XML).
  Reports uploaded as build artifacts.
- **`docker-build`** (after `quality`): builds the image with Buildx + GHA layer cache
  (build artifacts are **not** cached across runs — each release build is from a clean
  checkout), then smoke-tests it: run the container, poll `/readyz`, count `mlops_` metrics.

### Reproduce CI locally

```bash
pip install -e ".[dev]" -c constraints.txt
ruff check . && ruff format --check . && mypy src
pytest --cov=mlops_pipeline --cov-report=term-missing
```

---

## 6. Version reconciliation (R-1)

`constraints.txt` pins the exact versions QA validated (Stage 4). These match the
Stage-3 developer pins recorded in `docs/status.md`; the OTel set is internally
consistent (`opentelemetry-sdk 1.44.0` ↔ `opentelemetry-exporter-prometheus 0.65b0`).
The `pyproject.toml` ranges already admit these versions, so no dependency-range change
was needed — the constraints file is the single reproducibility lock used by both CI and
the Docker build. Container image tags are pinned for reproducibility.

---

## 7. Load / latency check (AC-14, NFR-2)

Not part of the unit suite. With the stack up, drive load and read p95 off the dashboard
(or Prometheus directly), e.g.:

```bash
# simple loop, or use hey/locust
for i in $(seq 1 2000); do curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' -d @sample_request.json >/dev/null; done
```

Then evaluate the p95 query above; target is well under 200 ms (stretch p95 ≤ 50 ms).
