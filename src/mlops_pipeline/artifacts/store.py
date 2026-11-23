"""Filesystem-backed artifact store: immutable save + load (DESIGN §4.3).

@ingroup artifacts
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from .schema import ArtifactMetadata, Baseline
from .versioning import read_pointer, write_pointer

__all__ = [
    "ArtifactNotFoundError",
    "ArtifactStore",
    "LoadedArtifact",
]

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
BASELINE_FILENAME = "baseline.json"


class ArtifactNotFoundError(RuntimeError):
    """Raised when a requested (or latest) artifact version cannot be resolved."""


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """A fully loaded artifact: estimator + metadata + baseline."""

    model: BaseEstimator
    metadata: ArtifactMetadata
    baseline: Baseline

    @property
    def version_id(self) -> str:
        return self.metadata.version_id


class ArtifactStore:
    """Manages ``artifacts/<version-id>/`` directories and the ``latest.txt`` pointer."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(self, model: BaseEstimator, metadata: ArtifactMetadata, baseline: Baseline) -> str:
        """Atomically persist an immutable artifact directory; return its version-id.

        Writes to a temp directory then renames into place. A version-id collision
        aborts rather than overwriting a prior artifact (FR-4).
        """
        version_id = metadata.version_id
        if version_id != baseline.version_id:
            raise ValueError(
                f"metadata/baseline version_id mismatch: {version_id!r} != {baseline.version_id!r}"
            )
        final_dir = self._root / version_id
        if final_dir.exists():
            raise FileExistsError(f"artifact {version_id!r} already exists; refusing to overwrite")

        self._root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=self._root, prefix=f".{version_id}-"))
        try:
            joblib.dump(model, tmp_dir / MODEL_FILENAME)
            _write_json(tmp_dir / METADATA_FILENAME, metadata.to_json())
            _write_json(tmp_dir / BASELINE_FILENAME, baseline.to_json())
            os.replace(tmp_dir, final_dir)
        except BaseException:
            _rmtree(tmp_dir)
            raise

        write_pointer(self._root, version_id)
        return version_id

    def load(self, version_id: str | None = None) -> LoadedArtifact:
        """Load an artifact; ``None`` resolves the ``latest.txt`` pointer (FR-9)."""
        resolved = version_id or self.resolve_latest()
        if resolved is None:
            raise ArtifactNotFoundError("no artifact version specified and no latest pointer found")
        artifact_dir = self._root / resolved
        if not artifact_dir.is_dir():
            raise ArtifactNotFoundError(f"artifact {resolved!r} not found under {self._root}")

        model = joblib.load(artifact_dir / MODEL_FILENAME)
        metadata = ArtifactMetadata.from_json(_read_json(artifact_dir / METADATA_FILENAME))
        baseline = Baseline.from_json(_read_json(artifact_dir / BASELINE_FILENAME))
        return LoadedArtifact(model=model, metadata=metadata, baseline=baseline)

    def resolve_latest(self) -> str | None:
        """Return the version-id from ``latest.txt`` or ``None`` when absent."""
        return read_pointer(self._root)

    def list_versions(self) -> list[str]:
        """Return known version-ids, sorted newest-first (lexicographic == chronological)."""
        if not self._root.is_dir():
            return []
        versions = [
            child.name
            for child in self._root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]
        return sorted(versions, reverse=True)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink(missing_ok=True)
    path.rmdir()
