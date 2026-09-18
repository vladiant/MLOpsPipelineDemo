"""Console-script entry points: ``mlops-train`` and ``mlops-serve`` (DESIGN §2.2)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_settings
from .telemetry.logging import configure_logging, get_logger

__all__ = ["serve_command", "train_command"]


def train_command(argv: list[str] | None = None) -> int:
    """Run a training run and write a versioned artifact + latest pointer."""
    from .training.train import TrainConfig, train

    parser = argparse.ArgumentParser(prog="mlops-train", description="Train and version a model.")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    configure_logging()
    logger = get_logger()

    config = TrainConfig(
        dataset=args.dataset,
        seed=args.seed,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        n_bins=args.n_bins,
        artifacts_dir=args.artifacts_dir,
    )
    result = train(config)
    logger.info(
        "training_complete",
        extra={
            "event": "training_complete",
            "version_id": result.version_id,
            "artifact_dir": str(result.artifact_dir),
            "metrics": result.metrics,
        },
    )
    print(f"Wrote artifact {result.version_id} → {result.artifact_dir}")
    print(f"Metrics: {result.metrics}")
    return 0


def serve_command(argv: list[str] | None = None) -> int:
    """Launch uvicorn against ``create_app()`` (thin wrapper)."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="mlops-serve", description="Serve predictions.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    uvicorn.run(
        "mlops_pipeline.serving.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )
    return 0
