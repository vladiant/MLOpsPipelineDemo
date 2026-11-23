"""Predictor wrapping a loaded estimator with schema validation (DESIGN §4.8).

@ingroup serving
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..artifacts.store import LoadedArtifact

__all__ = ["FeatureValidationError", "PredictionOutcome", "Predictor"]


class FeatureValidationError(ValueError):
    """Raised on schema-violating input; mapped to HTTP 422 (FR-11)."""


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """Result of a single prediction."""

    predicted_index: int
    predicted_class: str
    probabilities: dict[str, float] | None


class Predictor:
    """Wraps a loaded estimator and enforces the artifact feature schema."""

    def __init__(self, artifact: LoadedArtifact) -> None:
        self._model = artifact.model
        self._metadata = artifact.metadata
        self._feature_names = artifact.metadata.feature_names
        self._target_classes = artifact.metadata.target_classes

    @property
    def model_version(self) -> str:
        return self._metadata.version_id

    def validate(self, features: Mapping[str, float]) -> np.ndarray:
        """Validate presence/absence/type of features; return an ordered vector (FR-11)."""
        provided = set(features)
        expected = set(self._feature_names)

        missing = expected - provided
        extra = provided - expected
        if missing:
            raise FeatureValidationError(f"missing required features: {sorted(missing)}")
        if extra:
            raise FeatureValidationError(f"unexpected features: {sorted(extra)}")

        vector = np.empty(len(self._feature_names), dtype=np.float64)
        for i, name in enumerate(self._feature_names):
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FeatureValidationError(
                    f"feature {name!r} must be numeric, got {type(value).__name__}"
                )
            numeric = float(value)
            if not np.isfinite(numeric):
                raise FeatureValidationError(f"feature {name!r} must be finite")
            vector[i] = numeric
        return vector

    def predict(self, features: Mapping[str, float]) -> PredictionOutcome:
        """Validate and run the model, returning label + probabilities.

        R-3 invariant: probability columns are labeled by ``model.classes_`` order,
        which equals ``metadata.target_classes`` (asserted in tests).
        """
        vector = self.validate(features).reshape(1, -1)
        index = int(self._model.predict(vector)[0])

        probabilities: dict[str, float] | None = None
        if hasattr(self._model, "predict_proba"):
            proba_row = self._model.predict_proba(vector)[0]
            probabilities = {
                self._target_classes[col]: float(prob) for col, prob in enumerate(proba_row)
            }

        class_position = list(self._model.classes_).index(index)
        predicted_class = self._target_classes[class_position]
        return PredictionOutcome(
            predicted_index=index,
            predicted_class=predicted_class,
            probabilities=probabilities,
        )
