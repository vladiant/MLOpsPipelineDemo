# Project Status — MLOpsPipelineDemo

**Goal:** Minimal but complete MLOps loop — train a small classifier on a public dataset, package it, serve it behind a REST/gRPC endpoint, and monitor drift/latency with the same OpenTelemetry/Prometheus/Grafana pattern used in `CommerceSystemDemo`.

## Scope (agreed with stakeholder)
- Model training script producing **versioned artifacts**.
- Serving endpoint with **health checks** and **request logging**.
- Reuse existing **OpenTelemetry/Prometheus** pattern for **inference latency** and **error-rate** dashboards.
- Drift monitoring.
- After each stage: commit via `semver-commit-description`.
- After work is done: set version + tag via `semver-version-publish`.

## Pipeline (new-project chain)
requirements-analyst → system-architect → python-developer → qa-engineer → release-engineer → technical-writer

## Stage Tracker
| Stage | Owner | Status | Notes |
|-------|-------|--------|-------|
| 1. Requirements | Requirements Analyst | ✅ Done | SRS at docs/requirements/SRS.md; 8 OQs resolved by PM |
| 2. Design | System Architect | ✅ Done | Design at docs/design/DESIGN.md; FastAPI+Pydantic, joblib, src-layout `mlops_pipeline` |
| 3. Implementation | Python Developer | ✅ Done | src/mlops_pipeline; 25 pytest pass; ruff+mypy clean; CLI verified |
| 4. QA | QA Engineer | ✅ Done | 33 pytest pass, 90% cov, 0 defects; APPROVED (AC-11/12/14 deferred to post-Release) |
| 5. Release/CI | Release Engineer | ✅ Done | Dockerfile + docker-compose (service/OTel/Prometheus/Grafana) verified live; GitHub Actions CI; pins reconciled via constraints.txt. AC-11/12 met; AC-14 p95 ≈ 25 ms observed |
| 5b. Defect fix loop | Python Developer → QA | 🔄 In progress | PM routed gauge defect back: switch drift/readiness/model_info gauges to ObservableGauge (design §4.5) |
| 6. Documentation | Technical Writer | ⬜ Not started | After fix loop |

## PM Decisions (open questions resolved)
- **OQ-1 Dataset:** Breast-Cancer (Wisconsin) — binary, 30 continuous features.
- **OQ-2 Model:** `RandomForestClassifier`, fixed seed, `predict_proba` enabled.
- **OQ-3 Transport:** REST-only; gRPC deferred.
- **OQ-4 Artifact versioning:** per-run id `<UTC-timestamp>-<short-content-hash>` + `latest` pointer; distinct from project semver (`VERSION`).
- **OQ-5 Drift measure:** PSI (per-feature + aggregate).
- **OQ-6 Trace/log depth:** metrics-first — OTel + Collector + Prometheus + Grafana + structured JSON logs; Tempo/Loki deferred.
- **OQ-7 Batch prediction:** single-record required; batch optional stretch.
- **OQ-8 Alerting:** dashboards-only; alert rules deferred.

## Blockers / Open Questions
- None open (all 8 SRS OQs resolved above).
- Design risks to carry into implementation: R-1 (reconcile OTel/Prometheus pins with CommerceSystemDemo), R-3 (`predict_proba` class ordering must match metadata), R-5 (finalize reproducibility tolerance — recommend exact-equal for fixed seed).

## Release Engineer notes (Stage 5)
- **R-1 reconciled:** `constraints.txt` pins the exact QA-validated versions (fastapi 0.115.14, scikit-learn 1.5.2, pydantic 2.13.5, numpy 2.0.2, joblib 1.6.0, uvicorn 0.30.6, opentelemetry-sdk/api 1.44.0, opentelemetry-exporter-prometheus 0.65b0, plus dev tools). `pyproject.toml` ranges already admit these; no range change needed. CI + Docker build both use the constraints file. Container image tags pinned.
- **Stack verified live:** `docker compose up --build` → service healthy on first-run training; Prometheus scrapes service `/metrics` (target up); Grafana auto-provisions the `MLOps Overview` dashboard + Prometheus datasource. AC-11 ✅, AC-12 ✅. AC-14: p95 ≈ 24–25 ms live (well under NFR-2 200 ms; beats 50 ms stretch).
- **Defect flagged for Developer/QA (product code, out of Release scope):** `model_ready`/`model_info`/drift PSI use synchronous OTel `create_gauge`; with exporter-prometheus 0.65b0 a synchronous-gauge value is dropped after the first `collect()`, so Prometheus scrapes consume it between updates. Dashboard mitigated with `last_over_time(...)` (config-only). Recommended fix: switch to `ObservableGauge` with callbacks (DESIGN §4.5 anticipated this). See docs/ci-cd/mlops-pipeline-demo-pipeline.md §4.1.

## Commit / Release Log
- `5123303` semver(patch): add SRS and project status tracker for MLOps loop (Stage 1)
- `a1fabee` semver(patch): add architecture and design document (Stage 2)
- Stage 3 pinned versions (R-1, for Release Engineer to reconcile): fastapi 0.115.14, scikit-learn 1.5.2, pydantic 2.13.5, numpy 2.0.2, joblib 1.6.0, uvicorn 0.30.6, opentelemetry-sdk 1.44.0, opentelemetry-exporter-prometheus 0.65b0.
- `157f4a9` semver(minor): implement train-package-serve-monitor MLOps pipeline (Stage 3)
- QA (Stage 4): 33 tests pass, 90% coverage, 0 defects; verdict APPROVED with AC-11/12/14 deferred to post-Release QA.
- `7c6c23e` semver(patch): extend QA test coverage for acceptance criteria (Stage 4)
