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
| 2. Design | System Architect | 🔄 In progress | Designing against SRS + PM decisions |
| 3. Implementation | Python Developer | ⬜ Not started | |
| 4. QA | QA Engineer | ⬜ Not started | |
| 5. Release/CI | Release Engineer | ⬜ Not started | |
| 6. Documentation | Technical Writer | ⬜ Not started | |

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

## Commit / Release Log
- (pending)
