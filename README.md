# MLOpsPipelineDemo

A minimal but complete MLOps loop: **train** a classifier on the scikit-learn
Breast-Cancer dataset, produce an **immutable versioned artifact**, **serve**
predictions behind a FastAPI service with health/readiness checks and structured
JSON logging, and **observe** the service with OpenTelemetry → Prometheus metrics
(latency, throughput, errors, predicted-class distribution, and PSI drift).

See [docs/design/DESIGN.md](docs/design/DESIGN.md) for the architecture and
[docs/requirements/SRS.md](docs/requirements/SRS.md) for requirements.

## Install (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Train

```bash
mlops-train                 # writes artifacts/<version-id>/ and updates latest.txt
```

## Serve

```bash
mlops-serve                 # launches uvicorn against the latest artifact
```

Endpoints:

| Method / path   | Purpose                          |
| --------------- | -------------------------------- |
| `POST /predict` | single-record prediction         |
| `GET /healthz`  | liveness                         |
| `GET /readyz`   | readiness (artifact loaded)      |
| `GET /metrics`  | Prometheus exposition (`mlops_`) |

## Test

```bash
pytest
```
