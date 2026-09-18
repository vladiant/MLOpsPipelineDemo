# Software Requirements Specification — MLOpsPipelineDemo

- **Project:** MLOpsPipelineDemo
- **Stage:** Requirement Analysis (SDLC stage 1)
- **Author:** Requirements Analyst
- **Status:** Draft for System Architect review
- **Date:** 2026-09-19
- **Reference project:** `CommerceSystemDemo` (https://github.com/vlantonov/CommerceSystemDemo) — reuse its OpenTelemetry / Prometheus / Grafana observability pattern.

---

## 1. Overview

MLOpsPipelineDemo is a portfolio project that demonstrates a **minimal but complete MLOps loop**:

1. Train a small classifier on a public dataset and produce a **versioned model artifact**.
2. **Package** the artifact so it can be loaded deterministically at serving time.
3. **Serve** predictions behind a REST endpoint with **health checks** and **structured request logging**.
4. **Observe** the service by reusing the `CommerceSystemDemo` OpenTelemetry/Prometheus/Grafana stack for **inference latency** and **error-rate** dashboards.
5. **Detect data/prediction drift** against a reference baseline captured at training time.

The goal is to show an end-to-end, reproducible ML lifecycle at the smallest viable scale — one dataset, one model, one service — not to build a production-grade platform. It exists to demonstrate MLOps competency (training reproducibility, artifact versioning, model serving, observability, drift monitoring) for a portfolio audience.

### 1.1 Definitions

| Term | Meaning |
|------|---------|
| Model artifact | Serialized trained model plus metadata (version, metrics, feature schema, training-data baseline). |
| Serving service | The long-running process exposing the prediction, health, and metrics endpoints. |
| Drift | A statistically detectable difference between live inference input (or prediction) distributions and the training baseline. |
| Baseline | Feature/label distribution summary captured at training time and stored with the artifact. |
| Observability stack | The Prometheus + Grafana + OpenTelemetry Collector (+ Tempo/Loki optional) compose stack reused from `CommerceSystemDemo`. |

---

## 2. Scope & Objectives

### 2.1 In Scope

- A **training script/CLI** that loads one public scikit-learn dataset, trains one classifier, evaluates it, and writes a versioned artifact plus a training baseline.
- **Artifact versioning**: each training run produces an immutable, uniquely identified artifact with reproducible metadata.
- A **serving service** (REST as primary transport) that loads a selected artifact and serves predictions.
- **Health/readiness endpoints** and **structured request logging** on the serving service.
- **Prometheus metrics exposure** for inference latency, throughput, and error rate, following the `CommerceSystemDemo` instrumentation pattern.
- **Grafana dashboards** for inference latency and error rate.
- **Drift detection** comparing live inputs against the stored training baseline, exposed as a metric and/or logged event.
- A **local `docker-compose` stack** that runs the service plus the observability stack.

### 2.2 Objectives

- Demonstrate a reproducible train → package → serve → monitor loop that a reviewer can run locally with a single command.
- Reuse (not reinvent) the reference observability approach to show consistency across portfolio projects.

### 2.3 Out of Scope (this iteration)

- Cloud deployment, managed model registries (MLflow, SageMaker, Vertex, etc.), and hosted feature stores.
- Distributed/large-scale training, GPU training, deep learning, or hyperparameter search at scale.
- Automated retraining / continuous-training triggers and CI-driven model promotion gates.
- Authentication, authorization, multi-tenancy, and rate limiting on the serving API.
- A/B testing, canary rollout, or shadow deployment of multiple model versions simultaneously.
- Real-time streaming ingestion (Kafka, etc.).
- Front-end UI beyond the auto-generated API docs and Grafana.
- Persistent long-term storage of every inference request (only rolling in-memory/windowed stats required for drift).

---

## 3. Functional Requirements

Each requirement is independently testable. IDs are stable references for downstream stages.

### 3.1 Training & Versioned Artifacts

- **FR-1 — Dataset loading.** The training component shall load exactly one public dataset bundled with or downloadable via a standard library (recommended: a scikit-learn toy dataset such as Iris, Wine, or Breast-Cancer). No proprietary or private data.
- **FR-2 — Model training.** The training component shall train a single classifier on the loaded dataset using a fixed, configurable random seed.
- **FR-3 — Evaluation.** The training component shall evaluate the trained model on a held-out split and record at least accuracy and one additional classification metric (e.g., F1 or per-class precision/recall).
- **FR-4 — Versioned artifact production.** Each successful training run shall produce an immutable artifact identified by a unique version (e.g., semantic or timestamp/content-hash based). Re-running training shall not overwrite a prior artifact of a different version.
- **FR-5 — Artifact metadata.** The artifact shall embed or accompany metadata including: version identifier, training timestamp, dataset name, library versions, random seed, evaluation metrics, and the input feature schema (feature names, order, and expected types/ranges).
- **FR-6 — Training baseline capture.** The training run shall persist a statistical baseline of the training feature distributions (and class balance) alongside the artifact, for later drift comparison (per FR-13).
- **FR-7 — Reproducibility.** Given the same seed, dataset, and library versions, re-running training shall yield an artifact with equivalent evaluation metrics (within a documented tolerance).

### 3.2 Packaging

- **FR-8 — Deterministic packaging.** The system shall package a trained artifact into a self-contained, loadable form (serialized model + metadata + baseline) that the serving service can load without access to the training code path.
- **FR-9 — Artifact selection.** The serving service shall load a specific artifact version via configuration (e.g., an env var or path), defaulting to a designated "current"/latest artifact when unspecified.

### 3.3 Serving Endpoint

- **FR-10 — Prediction API.** The service shall expose a REST prediction endpoint that accepts a JSON request containing the model's input features and returns a JSON response containing at least the predicted class label; class probabilities/confidence shall be returned when the model supports them.
- **FR-11 — Input validation.** The service shall validate incoming prediction requests against the artifact's feature schema and return a client error (HTTP 4xx) with a descriptive message for malformed or schema-violating input.
- **FR-12 — Model/version transparency.** Prediction responses (or response headers/metadata) shall include the model version that produced the prediction.
- **FR-13 — Health & readiness checks.** The service shall expose a health/readiness endpoint that reports service liveness and whether a valid model artifact is loaded and ready to serve.
- **FR-14 — Structured request logging.** The service shall emit one structured (JSON) log record per prediction request including at minimum: timestamp, request identifier, endpoint, HTTP status, latency, and model version; logs shall support trace correlation fields consistent with the `CommerceSystemDemo` logging pattern.

### 3.4 Metrics & Observability

- **FR-15 — Prometheus metrics endpoint.** The service shall expose a `/metrics` endpoint in Prometheus text format, using the OpenTelemetry/Prometheus exporter approach reused from `CommerceSystemDemo`.
- **FR-16 — Inference latency metric.** The service shall record inference request latency as a histogram suitable for computing p50/p95/p99.
- **FR-17 — Throughput & error metrics.** The service shall expose request throughput (total requests) and error counts/rates by status class, enabling an error-rate dashboard.
- **FR-18 — Prediction metrics.** The service shall expose at least one ML-domain metric, such as predicted-class distribution counts, to support drift and behavior monitoring.

### 3.5 Drift Detection

- **FR-19 — Drift computation.** The service (or a companion component) shall compare a rolling window of live inference inputs (and/or predicted-class distribution) against the training baseline captured in FR-6, using a documented drift measure (e.g., population stability index, Kolmogorov–Smirnov, or Jensen–Shannon divergence).
- **FR-20 — Drift exposure.** Drift results shall be exposed as a Prometheus metric and/or a structured log event so they can be visualized and alerted on.

### 3.6 Dashboards

- **FR-21 — Latency & error dashboards.** A provisioned Grafana dashboard shall visualize inference latency (p95 at minimum) and error rate from the metrics in FR-16/FR-17.
- **FR-22 — Drift/behavior panel.** A Grafana panel shall visualize the drift metric and/or predicted-class distribution from FR-18/FR-20.
- **FR-23 — Reproducible provisioning.** Dashboards and datasources shall be provisioned as code so the stack comes up ready-to-use via `docker-compose`, mirroring the `CommerceSystemDemo` provisioning pattern.

---

## 4. Non-Functional Requirements

- **NFR-1 — Observability reuse.** Instrumentation shall follow the `CommerceSystemDemo` OpenTelemetry + Prometheus pattern (metrics via a `/metrics` exporter, structured JSON logs, optional OTLP traces to an OpenTelemetry Collector). The metric namespace shall use a project-specific prefix (e.g., `mlops_`) analogous to the reference `commerce_` prefix.
- **NFR-2 — Latency target.** For the single-record prediction endpoint on the demo dataset, served locally, p95 inference latency should be well under 200 ms (target p95 ≤ 50 ms as a stretch goal). Any endpoint that cannot meet the target shall have the reason documented, consistent with the reference project's approach.
- **NFR-3 — Reproducible training.** Training shall be deterministic given pinned dependencies and a fixed seed (supports FR-7); dependency versions shall be pinned.
- **NFR-4 — Portability.** The full loop (train, serve, observe) shall run locally on Linux via `docker-compose` with no cloud dependency and no manual post-setup steps beyond a documented command.
- **NFR-5 — Supported runtime.** The project shall target a modern Python version (recommend 3.11+; minimum 3.10) and declare it in `pyproject.toml`.
- **NFR-6 — Maintainability.** The codebase shall be a `pyproject.toml`-based package with clear separation between training, packaging, serving, and observability concerns (exact layout is the System Architect's decision).
- **NFR-7 — Testability.** Core behaviors (training produces a valid versioned artifact, prediction contract, health check, drift computation) shall be covered by `pytest` tests runnable without the full docker stack (e.g., against an in-process app / loaded artifact).
- **NFR-8 — Startup readiness.** The service shall not report "ready" until a valid artifact is loaded (supports FR-13), so orchestration can gate traffic correctly.
- **NFR-9 — Licensing.** The project shall retain an OSI-approved open-source license (existing `LICENSE`) and use only permissively licensed dependencies and public datasets.
- **NFR-10 — Resource footprint.** The demo shall run comfortably on a typical developer laptop (no GPU, modest RAM), consistent with the toy-dataset scope.

---

## 5. Assumptions & Constraints

- **A-1** Implementation language is Python, packaged via `pyproject.toml`, tested with `pytest`.
- **A-2** Only a public dataset is used; recommended default is a scikit-learn toy classification dataset (Iris/Wine/Breast-Cancer) requiring no download credentials.
- **A-3** A single classifier and a single model artifact version are served at a time (no simultaneous multi-version serving in this iteration).
- **A-4** The observability approach is reused from `CommerceSystemDemo`: Prometheus scrapes the service `/metrics`; Grafana reads Prometheus; structured JSON logs to stdout; optional OTLP traces → OpenTelemetry Collector → Tempo; optional logs → Promtail → Loki.
- **A-5** The entire stack runs locally via `docker-compose`; there is no cloud dependency and no external managed services.
- **A-6** REST is the **primary and required** transport. gRPC is **optional/stretch** and out of scope unless explicitly prioritized by the PM.
- **A-7** No authentication/authorization is required, matching the reference project and the portfolio-demo scope.
- **A-8** Drift is evaluated over a rolling in-memory window of recent requests; long-term inference persistence is not required.
- **A-9** Web framework and serialization/library choices are deferred to the System Architect; the reference project uses FastAPI + Pydantic, which is a natural fit but not mandated here.

---

## 6. Out of Scope

See §2.3. Summarized: no cloud/managed registries, no distributed/GPU/deep-learning training, no automated retraining or promotion gates, no auth, no multi-version/canary serving, no streaming ingestion, no bespoke UI, and no long-term per-request storage.

---

## 7. Open Questions (for the PM)

- **OQ-1 — Dataset choice.** Confirm the public dataset. Recommendation: **Iris** (smallest, fastest, easiest to demo) or **Breast-Cancer** (binary, slightly more "real-world" framing). Which do you prefer?
- **OQ-2 — Model choice.** Any preference for the classifier (e.g., `LogisticRegression` vs `RandomForestClassifier`)? Recommendation: a simple, deterministic model for reproducibility.
- **OQ-3 — Transport scope.** Confirm **REST-only** for this iteration, with gRPC explicitly deferred. Is gRPC required for the portfolio narrative, or acceptable as a future enhancement?
- **OQ-4 — Artifact versioning scheme.** Preference between semantic version, timestamp-based, or content-hash identifiers for artifacts? (This also interacts with the `semver-version-publish` project tagging flow.)
- **OQ-5 — Drift measure.** Any preferred drift statistic (PSI, KS, Jensen–Shannon)? Recommendation: PSI for interpretability, or KS for simplicity.
- **OQ-6 — Traces & logs depth.** Is full trace (Tempo) and log-aggregation (Loki/Promtail) parity with `CommerceSystemDemo` required, or is metrics-only (Prometheus + Grafana) sufficient for the minimal demo? This affects compose-stack size.
- **OQ-7 — Batch prediction.** Should the prediction endpoint support batch (list of records) in addition to single-record, or is single-record sufficient for v1?
- **OQ-8 — Alerting.** Are Prometheus alert rules (e.g., high error rate, high p95, drift threshold breached) in scope for this iteration, or dashboards-only?

Where these are unresolved, the SRS states a recommended default; the recommendations are not commitments and should be confirmed before design freeze.

---

## 8. Acceptance Criteria

A QA engineer shall be able to verify the following checklist:

- [ ] **AC-1** Running the training command produces a versioned artifact file with embedded/accompanying metadata (version, timestamp, dataset, seed, metrics, feature schema). *(FR-1..FR-5)*
- [ ] **AC-2** Two training runs with the same seed yield equivalent evaluation metrics within the documented tolerance; two runs without version overwrite each other's artifacts. *(FR-4, FR-7, NFR-3)*
- [ ] **AC-3** A training baseline (feature distributions + class balance) is persisted with the artifact. *(FR-6)*
- [ ] **AC-4** The serving service loads a specified artifact version and reports it via health metadata and prediction responses. *(FR-9, FR-12, FR-13)*
- [ ] **AC-5** A valid prediction request returns a predicted label (and probabilities when supported) with HTTP 200. *(FR-10)*
- [ ] **AC-6** A malformed/schema-violating prediction request returns a descriptive HTTP 4xx. *(FR-11)*
- [ ] **AC-7** The health/readiness endpoint reports "not ready" when no valid artifact is loaded and "ready" once one is. *(FR-13, NFR-8)*
- [ ] **AC-8** Each prediction produces one structured JSON log record containing timestamp, request id, endpoint, status, latency, and model version. *(FR-14)*
- [ ] **AC-9** `GET /metrics` returns Prometheus-format output including inference latency histogram, request throughput, error counters, and a predicted-class metric using the project metric prefix. *(FR-15..FR-18, NFR-1)*
- [ ] **AC-10** A drift metric/log is produced comparing live inputs/predictions to the training baseline using the chosen measure. *(FR-19, FR-20)*
- [ ] **AC-11** Grafana comes up pre-provisioned with a dashboard showing p95 inference latency, error rate, and a drift/prediction-distribution panel. *(FR-21..FR-23)*
- [ ] **AC-12** `docker-compose up` brings up the service plus observability stack locally with no manual post-setup, on Linux, without cloud dependencies. *(NFR-4)*
- [ ] **AC-13** `pytest` passes for training, prediction contract, health, and drift logic without requiring the full docker stack. *(NFR-7)*
- [ ] **AC-14** Under a simple local load test, single-record prediction p95 latency meets NFR-2 (or the deviation is documented). *(NFR-2)*

---

## 9. Handoff

Requirements are ready for the System Architect agent to design against. Open questions OQ-1..OQ-8 should ideally be confirmed by the PM before design freeze; recommended defaults are stated so design can proceed provisionally if answers are pending.
