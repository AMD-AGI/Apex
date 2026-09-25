"""Path-free identity for the Apex bytes that execute one experiment."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)

from .workspace_identity import WorkspaceGitIdentityResolver


SCHEMA = "apex.execution-identity/v1"
_DISTRIBUTION = "amd-apex-optimizer"
_IGNORED_PARTS = {"__pycache__"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True, slots=True)
class ApexExecutionIdentity:
    """Immutable observation of executed Apex package bytes and lock identity."""

    canonical_payload: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        if sha256_bytes(self.canonical_payload) != self.receipt_sha256:
            raise IntegrityError(
                "Apex execution identity digest differs",
                "execution_identity_tampered",
            )
        _validate_payload(self.document)

    @property
    def document(self) -> Mapping[str, Any]:
        try:
            value = json.loads(self.canonical_payload)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "Apex execution identity is invalid JSON",
                "execution_identity_tampered",
            ) from error
        if not isinstance(value, Mapping):
            raise IntegrityError(
                "Apex execution identity root differs",
                "execution_identity_tampered",
            )
        return value

    @property
    def apex_tree(self) -> str | None:
        value = self.document["repository"]["tree"]
        return str(value) if value is not None else None

    @property
    def source_manifest_sha256(self) -> str:
        return str(self.document["package"]["source_manifest_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return {**self.document, "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ApexExecutionIdentity":
        if set(value) != {
            "schema",
            "repository",
            "package",
            "dependency_lock_sha256",
            "receipt_sha256",
        }:
            raise IntegrityError(
                "Apex execution identity field set differs",
                "execution_identity_tampered",
            )
        digest = value.get("receipt_sha256")
        payload = {key: value[key] for key in value if key != "receipt_sha256"}
        if not isinstance(digest, str):
            raise IntegrityError(
                "Apex execution identity digest is missing",
                "execution_identity_tampered",
            )
        return cls(canonical_json_bytes(payload), digest)


def collect_apex_execution_identity(
    apex_root: Path,
    *,
    dependency_lock_sha256: str | None = None,
    package_root: Path | None = None,
) -> ApexExecutionIdentity:
    """Observe exact package bytes without asserting release readiness."""

    root = apex_root.expanduser().resolve(strict=True)
    source = (package_root or Path(__file__).resolve().parents[1]).resolve(strict=True)
    if not source.is_dir():
        raise ContractError(
            "Apex package root is not a directory",
            "execution_identity_unavailable",
        )
    files = _source_manifest(source)
    repository = WorkspaceGitIdentityResolver().inspect(root).to_dict()
    payload = {
        "schema": SCHEMA,
        "repository": repository,
        "package": {
            "distribution": _DISTRIBUTION,
            "version": _distribution_version(),
            "source_manifest_sha256": sha256_json(
                {"schema": "apex.package-source-manifest/v1", "files": files}
            ),
            "file_count": len(files),
        },
        "dependency_lock_sha256": dependency_lock_sha256,
    }
    canonical = canonical_json_bytes(payload)
    return ApexExecutionIdentity(canonical, sha256_bytes(canonical))


def _source_manifest(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ContractError(
                "Apex package source cannot contain symlinks",
                "execution_identity_unavailable",
            )
        if not path.is_file() or path.suffix in _IGNORED_SUFFIXES:
            continue
        content = path.read_bytes()
        entries.append(
            {
                "path": relative.as_posix(),
                "size": len(content),
                "sha256": sha256_bytes(content),
            }
        )
    if not entries:
        raise ContractError(
            "Apex package source is empty",
            "execution_identity_unavailable",
        )
    return entries


def _distribution_version() -> str | None:
    try:
        return importlib.metadata.version(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_payload(value: Mapping[str, Any]) -> None:
    if set(value) != {"schema", "repository", "package", "dependency_lock_sha256"}:
        raise IntegrityError(
            "Apex execution identity payload fields differ",
            "execution_identity_tampered",
        )
    repository = value.get("repository")
    package = value.get("package")
    lock = value.get("dependency_lock_sha256")
    valid = (
        value.get("schema") == SCHEMA
        and isinstance(repository, Mapping)
        and _valid_repository(repository)
        and isinstance(package, Mapping)
        and set(package) == {
            "distribution",
            "version",
            "source_manifest_sha256",
            "file_count",
        }
        and package.get("distribution") == _DISTRIBUTION
        and (
            package.get("version") is None
            or isinstance(package.get("version"), str)
        )
        and _digest(package.get("source_manifest_sha256"))
        and isinstance(package.get("file_count"), int)
        and package.get("file_count", 0) > 0
        and (lock is None or _digest(lock))
    )
    if not valid:
        raise IntegrityError(
            "Apex execution identity payload is invalid",
            "execution_identity_tampered",
        )


def _valid_repository(value: Mapping[str, Any]) -> bool:
    if set(value) != {
        "root_sha256",
        "status",
        "remote",
        "commit",
        "tree",
        "dirty_paths",
        "unavailable_reason",
    }:
        return False
    resolved = value.get("status") == "resolved"
    identity = (
        isinstance(value.get("remote"), str)
        and _git_object(value.get("commit"))
        and _git_object(value.get("tree"))
    )
    empty_identity = all(
        value.get(field) is None for field in ("remote", "commit", "tree")
    )
    reason = value.get("unavailable_reason")
    return bool(
        _digest(value.get("root_sha256"))
        and value.get("status") in {"resolved", "unresolved"}
        and isinstance(value.get("dirty_paths"), list)
        and all(isinstance(item, str) for item in value["dirty_paths"])
        and (identity if resolved else empty_identity)
        and (reason is None if resolved else isinstance(reason, str) and bool(reason))
    )


def _git_object(value: object) -> bool:
    return (
        isinstance(value, str)
        and 40 <= len(value) <= 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = ["ApexExecutionIdentity", "collect_apex_execution_identity"]
