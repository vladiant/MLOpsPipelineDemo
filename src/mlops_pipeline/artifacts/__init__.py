"""Artifact packaging: shared schema, versioning, and the filesystem store."""

from __future__ import annotations

from .schema import (
    ArtifactMetadata,
    Baseline,
    FeatureBaseline,
    FeatureSpec,
)
from .store import ArtifactNotFoundError, ArtifactStore, LoadedArtifact
from .versioning import build_version_id, read_pointer, write_pointer

__all__ = [
    "ArtifactMetadata",
    "ArtifactNotFoundError",
    "ArtifactStore",
    "Baseline",
    "FeatureBaseline",
    "FeatureSpec",
    "LoadedArtifact",
    "build_version_id",
    "read_pointer",
    "write_pointer",
]
