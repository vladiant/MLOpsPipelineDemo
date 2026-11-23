"""Shared, dependency-free artifact contract layer (frozen dataclasses).

This module has no project-internal dependencies; ``training``, ``serving`` and
``drift`` all depend on it. Serialization to/from plain JSON dicts is explicit so
metadata and baseline files are stable and human-readable (FR-5, FR-6).

@ingroup artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ArtifactMetadata",
    "Baseline",
    "FeatureBaseline",
    "FeatureSpec",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One input feature's name, dtype and observed training range."""

    name: str
    dtype: str
    min: float
    max: float

    def to_json(self) -> dict[str, object]:
        return {"name": self.name, "dtype": self.dtype, "min": self.min, "max": self.max}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FeatureSpec:
        return cls(
            name=str(data["name"]),
            dtype=str(data["dtype"]),
            min=float(data["min"]),
            max=float(data["max"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Immutable metadata accompanying a serialized model (see DESIGN §3.3)."""

    schema_version: int
    version_id: str
    created_at: str
    dataset_name: str
    model_type: str
    random_seed: int
    library_versions: dict[str, str]
    hyperparameters: dict[str, object]
    metrics: dict[str, float]
    target_classes: list[str]
    feature_schema: list[FeatureSpec]
    model_file: str
    model_sha256: str

    @property
    def feature_names(self) -> list[str]:
        """Feature names in the order the model expects them."""
        return [spec.name for spec in self.feature_schema]

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version_id": self.version_id,
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "model_type": self.model_type,
            "random_seed": self.random_seed,
            "library_versions": dict(self.library_versions),
            "hyperparameters": dict(self.hyperparameters),
            "metrics": dict(self.metrics),
            "target_classes": list(self.target_classes),
            "feature_schema": [spec.to_json() for spec in self.feature_schema],
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ArtifactMetadata:
        raw_schema = data["feature_schema"]
        assert isinstance(raw_schema, list)
        return cls(
            schema_version=int(data["schema_version"]),
            version_id=str(data["version_id"]),
            created_at=str(data["created_at"]),
            dataset_name=str(data["dataset_name"]),
            model_type=str(data["model_type"]),
            random_seed=int(data["random_seed"]),
            library_versions={str(k): str(v) for k, v in dict(data["library_versions"]).items()},
            hyperparameters=dict(data["hyperparameters"]),
            metrics={str(k): float(v) for k, v in dict(data["metrics"]).items()},
            target_classes=[str(c) for c in data["target_classes"]],
            feature_schema=[FeatureSpec.from_json(dict(s)) for s in raw_schema],
            model_file=str(data["model_file"]),
            model_sha256=str(data["model_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class FeatureBaseline:
    """Per-feature training histogram used as the expected PSI distribution."""

    name: str
    bin_edges: list[float]
    bin_proportions: list[float]
    min: float
    max: float
    mean: float
    std: float

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bin_edges": list(self.bin_edges),
            "bin_proportions": list(self.bin_proportions),
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FeatureBaseline:
        return cls(
            name=str(data["name"]),
            bin_edges=[float(e) for e in data["bin_edges"]],
            bin_proportions=[float(p) for p in data["bin_proportions"]],
            min=float(data["min"]),
            max=float(data["max"]),
            mean=float(data["mean"]),
            std=float(data["std"]),
        )


@dataclass(frozen=True, slots=True)
class Baseline:
    """Training baseline for drift comparison (see DESIGN §3.4)."""

    schema_version: int
    version_id: str
    n_samples: int
    class_balance: dict[str, float]
    features: list[FeatureBaseline]

    def feature(self, name: str) -> FeatureBaseline:
        """Return the baseline for ``name`` or raise ``KeyError``."""
        for feat in self.features:
            if feat.name == name:
                return feat
        raise KeyError(name)

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version_id": self.version_id,
            "n_samples": self.n_samples,
            "class_balance": dict(self.class_balance),
            "features": [feat.to_json() for feat in self.features],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Baseline:
        raw_features = data["features"]
        assert isinstance(raw_features, list)
        return cls(
            schema_version=int(data["schema_version"]),
            version_id=str(data["version_id"]),
            n_samples=int(data["n_samples"]),
            class_balance={str(k): float(v) for k, v in dict(data["class_balance"]).items()},
            features=[FeatureBaseline.from_json(dict(f)) for f in raw_features],
        )
