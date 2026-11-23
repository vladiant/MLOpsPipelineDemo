# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** the `[0.1.0]` heading and date below are placeholders for the first
> tagged release; the project maintainer finalizes the version and date when the
> release tag is cut.

## [Unreleased]

### Added

- **API documentation (Doxygen)**: root `Doxyfile` renders an HTML API reference
  for the `mlops_pipeline` package (with `README.md` as the main page), including
  Graphviz/`dot` class, collaboration, include, call, and directory-dependency
  diagrams. `@defgroup`/`@ingroup` comments across `src/mlops_pipeline/**` group
  the code under a top-level `components` group nesting seven logical components
  (training, artifacts, serving, telemetry, drift, config, cli). Comment-only —
  no runtime behavior changed. Generated output (`docs/doxygen/`, and the GitLab
  `public/` dir) is git-ignored.
- **Docs publishing to GitHub Pages and GitLab Pages**: `.github/workflows/docs.yml`
  builds the Doxygen HTML (installing doxygen + graphviz) and deploys to GitHub
  Pages via `actions/upload-pages-artifact` + `actions/deploy-pages` (OIDC,
  `github-pages` environment), triggered on push to `main` touching `src/**`,
  `Doxyfile`, `README.md`, or the workflow, plus manual `workflow_dispatch`.
  `.gitlab-ci.yml` runs an equivalent single `pages` job on the GitLab mirror.
  Published at <https://vladiant.github.io/MLOpsPipelineDemo/> and
  <https://vladiant.gitlab.io/MLOpsPipelineDemo/>. One-time enablement: GitHub
  Settings → Pages → Source = "GitHub Actions"; Pages enabled on the GitLab
  mirror. Additive tooling only (no public API change) → PATCH-level; **PM to
  finalize the version bump, date, and tag.**

## [0.1.0] - 2026-09-19

First complete, runnable MLOps loop: train → package → serve → observe → drift-monitor.

### Added

- **Training pipeline** (`mlops-train`): trains a `RandomForestClassifier` (fixed
  seed) on the scikit-learn Breast-Cancer (Wisconsin) dataset and reports
  accuracy / f1 / precision / recall.
- **Immutable versioned artifacts**: each run writes
  `artifacts/<UTC-timestamp>-<short-content-hash>/` containing `model.joblib`,
  `metadata.json`, and `baseline.json`, with a `latest.txt` pointer. Artifact
  version ids are distinct from the project's semantic version.
- **Serving API** (`mlops-serve`, FastAPI): single-record `POST /predict` with
  Pydantic v2 validation (HTTP 422 on schema-invalid input) returning label,
  class index, probabilities, and the producing `model_version`; `GET /healthz`
  liveness; `GET /readyz` readiness (503 until a valid artifact is loaded);
  structured JSON request logging.
- **Observability**: `GET /metrics` Prometheus exposition under the `mlops_`
  prefix — inference-latency histogram, request/error counters, predicted-class
  counter, and gauges for model readiness, model info, and PSI drift (per-feature
  and aggregate).
- **Drift monitoring**: rolling-window PSI vs. the training baseline, computed on
  a background refresh tick once a minimum sample count is reached.
- **Local observability stack** (`docker-compose.yaml`): serving + OpenTelemetry
  Collector + Prometheus + Grafana, brought up with a single
  `docker compose up --build` and no manual post-setup. Grafana ships a
  provisioned Prometheus datasource and the **MLOps Overview** dashboard (p95/p50
  latency, error rate, predicted-class distribution, PSI drift, model readiness,
  throughput by status).
- **Container image** (`Dockerfile`, `docker-entrypoint.sh`): multi-stage,
  slim, non-root; trains a model into a mounted named volume on first start when
  none exists, then serves.
- **CI** (`.github/workflows/ci.yml`): quality gate (ruff, ruff-format, mypy,
  pytest + coverage) on a Python 3.11/3.12 matrix, followed by a Docker build and
  smoke test.
- **Reproducibility**: `constraints.txt` pins the QA-validated dependency
  versions used by both CI and the Docker build.
- **Documentation**: SRS, DESIGN, and CI/CD pipeline references under `docs/`.

### Changed

- Dashboard drift and readiness panels now use direct gauge queries
  (`max(mlops_drift_psi_aggregate)`, `topk(5, max by (feature) (mlops_feature_drift_psi))`,
  `max by (model_version) (mlops_model_ready)`), replacing the earlier
  `last_over_time(...)` workaround that was needed while the gauges were stale
  between scrapes.

### Fixed

- **Observable-gauge staleness across scrapes**: `mlops_model_ready`,
  `mlops_model_info`, `mlops_drift_psi_aggregate`, and `mlops_feature_drift_psi`
  were instrumented with OpenTelemetry *synchronous* gauges, whose value the
  Prometheus exporter (0.65b0) dropped after the first `collect()`. Prometheus
  scrapes therefore consumed these gauges between updates, so readiness/model-info
  showed a single sample and drift PSI only populated on the scrape that coincided
  with a refresh tick. The four instruments were switched to **observable gauges**
  with observer callbacks, so each value re-reports at its current level on every
  scrape.

[Unreleased]: https://example.com/compare/v0.1.0...HEAD
[0.1.0]: https://example.com/releases/tag/v0.1.0
