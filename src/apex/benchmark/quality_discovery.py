"""Non-formal discovery of public lm-eval artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apex.core import IntegrityError, sha256_file


def discover_quality_artifacts(workspace: Path) -> tuple[Path, ...]:
    """Discover public artifacts only when no formal authority is expected."""

    quality_root = workspace.resolve() / "lm_eval"
    if quality_root.is_symlink() or not quality_root.is_dir():
        return ()
    files: list[Path] = []
    for path in sorted(quality_root.rglob("*")):
        resolved = path.resolve()
        try:
            resolved.relative_to(quality_root)
        except ValueError as error:
            raise IntegrityError(
                f"Quality artifact escapes benchmark workspace: {path}",
                "unsafe_quality_artifact",
            ) from error
        if path.is_symlink():
            raise IntegrityError(
                f"Quality artifact must be a regular file: {path}",
                "unsafe_quality_artifact",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise IntegrityError(
                f"Quality artifact must be a regular file: {path}",
                "unsafe_quality_artifact",
            )
        if _is_quality_artifact(path):
            files.append(resolved)
    return tuple(files)


def artifact_receipt(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(workspace.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _is_quality_artifact(path: Path) -> bool:
    if path.name.startswith("results") and path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
            return False
    return path.name.startswith(("results", "samples"))


__all__ = ["artifact_receipt", "discover_quality_artifacts"]
