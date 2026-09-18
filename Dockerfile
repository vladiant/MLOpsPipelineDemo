# syntax=docker/dockerfile:1

# Multi-stage build for the MLOpsPipelineDemo serving service.
# Stage 1 builds a self-contained virtualenv from pinned deps; stage 2 is a slim,
# non-root runtime that trains a model on first start (if none is mounted) and serves it.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1 — builder: create a venv with the package + reproducible pins
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Isolated venv we can copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what the build needs first so layer caching works on source-only edits.
COPY pyproject.toml README.md LICENSE constraints.txt ./
COPY src ./src

# Install the package (runtime deps only) against the QA-validated constraints.
RUN pip install --no-cache-dir . -c constraints.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim, non-root, ships the venv only
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MLOPS_ARTIFACTS_DIR=/app/artifacts \
    MLOPS_HOST=0.0.0.0 \
    MLOPS_PORT=8000

# Fixed non-root uid/gid so a named volume mounted at /app/artifacts inherits
# writable ownership on first initialization (no manual chown needed).
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --home-dir /app --create-home app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/artifacts \
    && chown -R app:app /app

USER app

EXPOSE 8000

# Container-native readiness probe (no curl in the slim image).
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=6 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz', timeout=3).status==200 else 1)"]

ENTRYPOINT ["docker-entrypoint.sh"]
