"""Training orchestrator: load → train → evaluate → persist artifact (FR-1..FR-7).

@ingroup training
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..artifacts.schema import SCHEMA_VERSION, ArtifactMetadata, FeatureSpec
from ..artifacts.store import ArtifactStore
from ..artifacts.versioning import build_version_id, content_hash
from .baseline import compute_baseline
from .dataset import load_dataset

__all__ = ["TrainConfig", "TrainResult", "train"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Configuration for a single training run."""

    dataset: str = "breast_cancer"
    seed: int = 42
    test_size: float = 0.2
    n_estimators: int = 200
    n_bins: int = 10
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Outcome of a training run."""

    version_id: str
    artifact_dir: Path
    metrics: dict[str, float]


def train(config: TrainConfig) -> TrainResult:
    """Run the full training pipeline and persist an immutable versioned artifact."""
    data = load_dataset(config.dataset, seed=config.seed, test_size=config.test_size)

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.seed,
    )
    model.fit(data.X_train, data.y_train)

    metrics = _evaluate(model, data.X_test, data.y_test)

    # R-3 invariant: probability column order must follow model.classes_.
    target_classes = [data.target_names[int(c)] for c in model.classes_]

    created_dt = datetime.now(UTC)
    created_at = created_dt.strftime("%Y%m%dT%H%M%SZ")
    model_bytes = _serialize(model)
    version_id = build_version_id(model_bytes, created_at=created_at)

    metadata = ArtifactMetadata(
        schema_version=SCHEMA_VERSION,
        version_id=version_id,
        created_at=created_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dataset_name=config.dataset,
        model_type="sklearn.ensemble.RandomForestClassifier",
        random_seed=config.seed,
        library_versions={
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        hyperparameters={
            "n_estimators": config.n_estimators,
            "max_depth": None,
        },
        metrics=metrics,
        target_classes=target_classes,
        feature_schema=_feature_schema(data.X_train, data.feature_names),
        model_file="model.joblib",
        model_sha256=content_hash(model_bytes),
    )

    baseline = compute_baseline(
        data.X_train,
        data.y_train,
        data.feature_names,
        data.target_names,
        version_id,
        n_bins=config.n_bins,
    )

    store = ArtifactStore(config.artifacts_dir)
    store.save(model, metadata, baseline)

    return TrainResult(
        version_id=version_id,
        artifact_dir=config.artifacts_dir / version_id,
        metrics=metrics,
    )


def _evaluate(
    model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }


def _feature_schema(X_train: np.ndarray, feature_names: list[str]) -> list[FeatureSpec]:
    X = np.asarray(X_train, dtype=np.float64)
    return [
        FeatureSpec(
            name=name,
            dtype="float64",
            min=float(np.min(X[:, col])),
            max=float(np.max(X[:, col])),
        )
        for col, name in enumerate(feature_names)
    ]


def _serialize(model: RandomForestClassifier) -> bytes:
    import io

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()
