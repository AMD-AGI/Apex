"""Confined filesystem reads for targeted trace validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError

from .targeted_trace_models import nonempty_text


def resolve_workspace_path(
    value: object, workspace: Path, *, expected_name: str
) -> Path:
    raw = nonempty_text(value, expected_name)
    candidate = Path(raw)
    candidate = candidate if candidate.is_absolute() else workspace / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(workspace)
    except (OSError, ValueError) as error:
        raise IntegrityError(
            "Diagnostic artifact escapes or is missing from workspace",
            "invalid_artifact_path",
        ) from error
    if not resolved.is_file() or resolved.name != expected_name:
        raise IntegrityError(
            "Diagnostic artifact has an unexpected path", "invalid_artifact_path"
        )
    return resolved


def resolve_trace_path(value: object, *, workspace: Path, trace_dir: Path) -> Path:
    raw = nonempty_text(value, "shard path")
    candidate = Path(raw)
    candidate = candidate if candidate.is_absolute() else trace_dir / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(workspace)
        resolved.relative_to(trace_dir)
    except (OSError, ValueError) as error:
        raise IntegrityError(
            "Shard path escapes or is missing from trace directory",
            "invalid_shard_path",
        ) from error
    if not resolved.is_file() or resolved.suffix != ".jsonl":
        raise IntegrityError(
            "Shard receipt does not name a JSONL file", "invalid_shard_path"
        )
    return resolved


def read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            f"Cannot read {label}", "invalid_targeted_trace"
        ) from error
    if not isinstance(value, Mapping):
        raise IntegrityError(
            f"{label} is not an object", "invalid_targeted_trace"
        )
    return value


__all__ = ["read_object", "resolve_trace_path", "resolve_workspace_path"]
