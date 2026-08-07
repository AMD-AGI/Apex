"""Canonical hashing helpers used for identities, events, and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data deterministically as UTF-8 bytes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest without an algorithm prefix."""

    return hashlib.sha256(content).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using :func:`canonical_json_bytes`."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
