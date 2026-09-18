"""Artifact version-id construction and the atomic ``latest.txt`` pointer (FR-4/FR-9)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

__all__ = [
    "POINTER_FILENAME",
    "build_version_id",
    "content_hash",
    "read_pointer",
    "short_hash",
    "write_pointer",
]

POINTER_FILENAME = "latest.txt"


def content_hash(model_bytes: bytes) -> str:
    """Full sha256 hex digest of the serialized model bytes."""
    return hashlib.sha256(model_bytes).hexdigest()


def short_hash(model_bytes: bytes, *, length: int = 8) -> str:
    """First ``length`` hex chars of the model content hash."""
    return content_hash(model_bytes)[:length]


def build_version_id(model_bytes: bytes, *, created_at: str) -> str:
    """Build ``<UTC-timestamp>-<short-content-hash>`` (DESIGN §3.2).

    Args:
        model_bytes: The serialized ``model.joblib`` bytes.
        created_at: Compact UTC timestamp, e.g. ``20260919T141530Z``.
    """
    return f"{created_at}-{short_hash(model_bytes)}"


def read_pointer(root: Path) -> str | None:
    """Return the version-id in ``latest.txt`` or ``None`` if missing/empty."""
    pointer = root / POINTER_FILENAME
    if not pointer.is_file():
        return None
    value = pointer.read_text(encoding="utf-8").strip()
    return value or None


def write_pointer(root: Path, version_id: str) -> None:
    """Atomically write ``latest.txt`` (temp file + rename)."""
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=root, prefix=".latest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{version_id}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, root / POINTER_FILENAME)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
