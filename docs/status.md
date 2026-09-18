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
| 5b. Defect fix loop | Python Developer → QA | ✅ Done | Gauges → ObservableGauge; 35 tests pass, 91% cov; QA APPROVED, no regression |
| 5c. Dashboard cleanup | Release Engineer | ✅ Done | Removed last_over_time workaround; live-verified gauges persist per scrape; defect RESOLVED |
| 6. Documentation | Technical Writer | ✅ Done | Portfolio README (architecture, quickstarts, API, observability, CI, layout); CHANGELOG.md (Keep a Changelog, observable-gauge fix under Fixed); status updated. Awaiting PM version tag |

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
- `d3ca832` semver(minor): add container, observability stack, and CI pipeline (Stage 5)
- Fix loop (5b): Release found synchronous-gauge staleness defect; PM routed to Developer → converted to ObservableGauge; QA regression APPROVED (35 tests, 91% cov).
- `1e80d65` semver(patch): report gauges via OpenTelemetry observable gauges (Stage 5b)
- `e1a0c61` semver(patch): simplify Grafana gauge panels after observable-gauge fix (Stage 5c)
- OTLP doc discrepancy (writer-flagged) reconciled by Technical Writer: CI-CD doc §3 corrected to match compose (OTLP push disabled by default; Prometheus pull is authoritative).

## Technical Writer notes (Stage 6)
- **Docs shipped:** rewrote `README.md` as the portfolio front door (project summary + value, Mermaid architecture of the train→serve→observe→drift loop, local + Docker Compose quickstarts, API reference with sample `/predict` request/response, observability section with `mlops_` metric names + dashboard-panel + PSI explanation, testing/CI, project layout, config env vars, explicit non-goals). Ports, Grafana dashboard name/uid, and compose commands were taken from `docker-compose.yaml`, `configs/prometheus/prometheus.yml`, `configs/grafana/provisioning/dashboards/mlops-overview.json`, and `docs/ci-cd/mlops-pipeline-demo-pipeline.md`.
- **CHANGELOG:** added `CHANGELOG.md` (Keep a Changelog style) with all shipped work under `[0.1.0] - 2026-09-19` (placeholder — **PM to finalize version/date and attach the tag**). Observable-gauge staleness recorded under **Fixed**; `last_over_time` removal recorded under **Changed**.
- **`last_over_time` audit:** confirmed no doc presents the removed `last_over_time` dashboard workaround as current behavior. `docs/ci-cd/...md §4.1` frames it as RESOLVED/removed; `status.md` row 5c is historical. README/CHANGELOG describe the direct-query gauges.
- **⚠ Discrepancy for PM (do NOT fix code — doc-vs-doc):** `docs/ci-cd/mlops-pipeline-demo-pipeline.md §3` states the service "pushes OTLP metrics to the collector (`MLOPS_OTLP_ENDPOINT=otel-collector:4317`)", but the actual `docker-compose.yaml` intentionally leaves `MLOPS_OTLP_ENDPOINT` **unset** (OTLP push disabled to avoid double-counting; documented in the compose comments and §4.1 of the same file). §3 contradicts §4.1 and the compose file. README was written to match the real compose (OTLP push disabled, single-target scrape). Recommend system-architect/release-engineer reconcile §3 of the CI-CD doc.
