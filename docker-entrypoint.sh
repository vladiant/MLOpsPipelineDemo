#!/usr/bin/env sh
# Entrypoint for the serving container.
# Trains a model on first start when the mounted artifact store is empty, so the
# service can reach readiness with a single `docker compose up` (AC-12, NFR-4).
set -eu

ARTIFACTS_DIR="${MLOPS_ARTIFACTS_DIR:-/app/artifacts}"

if [ ! -f "${ARTIFACTS_DIR}/latest.txt" ]; then
    echo "[entrypoint] No artifact in ${ARTIFACTS_DIR}; training an initial model..."
    mlops-train --artifacts-dir "${ARTIFACTS_DIR}"
else
    echo "[entrypoint] Existing artifact found in ${ARTIFACTS_DIR}; skipping training."
fi

echo "[entrypoint] Starting serving service on ${MLOPS_HOST:-0.0.0.0}:${MLOPS_PORT:-8000}"
exec mlops-serve
