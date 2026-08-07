"""Verification and receipt contracts for an immutable lm-eval runtime CAS."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .lm_eval_lock import LmEvalRuntimeLock, RUNTIME_SCHEMA
from .repositories import BootstrapError


_MANIFEST = "lm_eval_runtime_manifest.json"
_SITE_PACKAGES = "site-packages"


@dataclass(frozen=True, slots=True)
class LmEvalRuntimeReceipt:
    """Identity of one fully revalidated read-only runtime."""

    root: Path
    runtime_sha256: str
    manifest_sha256: str
    identity: Mapping[str, str]
    file_count: int
    lock_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_SCHEMA,
            "path": str(self.root),
            "sha256": self.runtime_sha256,
            "manifest_sha256": self.manifest_sha256,
            "identity": dict(self.identity),
            "file_count": self.file_count,
            "lock_sha256": self.lock_sha256,
        }


def canonical_json(value: Any) -> bytes:
    """Return the canonical JSON encoding shared with the Magpie consumer."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_directory(path: Path, field: str) -> os.stat_result:
    if path.is_symlink():
        raise BootstrapError(f"{field} must not be a symlink: {path}")
    try:
        observed = path.stat()
    except OSError as error:
        raise BootstrapError(f"cannot stat {field} {path}: {error}") from error
    if not stat.S_ISDIR(observed.st_mode):
        raise BootstrapError(f"{field} must be a directory: {path}")
    if stat.S_IMODE(observed.st_mode) & 0o222:
        raise BootstrapError(f"{field} must be read-only: {path}")
    return observed


def _require_regular(path: Path, field: str) -> tuple[os.stat_result, bytes]:
    if path.is_symlink():
        raise BootstrapError(f"{field} must not be a symlink: {path}")
    try:
        observed = path.stat()
        content = path.read_bytes()
    except OSError as error:
        raise BootstrapError(f"cannot read {field} {path}: {error}") from error
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise BootstrapError(f"{field} must be a non-hardlinked regular file: {path}")
    if stat.S_IMODE(observed.st_mode) & 0o222:
        raise BootstrapError(f"{field} must be read-only: {path}")
    return observed, content


def _load_manifest(root: Path) -> tuple[Mapping[str, Any], bytes]:
    _, content = _require_regular(root / _MANIFEST, "runtime manifest")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as error:
        raise BootstrapError(f"invalid lm-eval runtime manifest: {error}") from error
    keys = {"schema", "runtime_sha256", "site_packages", "identity", "files"}
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise BootstrapError("lm-eval runtime manifest has unexpected fields")
    if raw.get("schema") != RUNTIME_SCHEMA or raw.get("site_packages") != _SITE_PACKAGES:
        raise BootstrapError("lm-eval runtime manifest schema/path is invalid")
    return raw, content


def _record(path: Path, root: Path) -> dict[str, Any]:
    observed, content = _require_regular(path, "runtime file")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": _sha256(content),
    }


def collect_runtime_files(site_packages: Path) -> list[dict[str, Any]]:
    """Hash every regular file and reject links, special files, and writable dirs."""

    _require_directory(site_packages, "site-packages")
    records: list[dict[str, Any]] = []
    for current, directories, filenames in os.walk(site_packages, followlinks=False):
        directory = Path(current)
        _require_directory(directory, "runtime directory")
        for name in sorted(directories):
            child = directory / name
            if child.is_symlink():
                raise BootstrapError(f"runtime directory must not be a symlink: {child}")
        for name in sorted(filenames):
            records.append(_record(directory / name, site_packages))
    records.sort(key=lambda value: value["path"])
    return records


def _validate_manifest(
    raw: Mapping[str, Any], records: list[dict[str, Any]], lock: LmEvalRuntimeLock
) -> None:
    if raw.get("identity") != lock.identity:
        raise BootstrapError("lm-eval runtime identity differs from its lock")
    declared = raw.get("files")
    if not isinstance(declared, list) or declared != records:
        raise BootstrapError("lm-eval runtime file manifest differs from disk")
    tree_sha256 = _sha256(canonical_json(records))
    if tree_sha256 != lock.installed_tree_sha256:
        raise BootstrapError("lm-eval installed tree digest differs from its lock")
    runtime_sha256 = _sha256(
        canonical_json({"identity": dict(lock.identity), "files": records})
    )
    if raw.get("runtime_sha256") != runtime_sha256 or runtime_sha256 != lock.runtime_sha256:
        raise BootstrapError("lm-eval runtime digest differs from its lock")


def verify_lm_eval_runtime(
    root: Path, lock: LmEvalRuntimeLock
) -> LmEvalRuntimeReceipt:
    """Recompute every byte, mode, and identity before returning a receipt."""

    if not root.is_absolute():
        raise BootstrapError("lm-eval runtime path must be absolute")
    if root.is_symlink():
        raise BootstrapError(f"runtime root must not be a symlink: {root}")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise BootstrapError(f"cannot resolve lm-eval runtime root {root}: {error}") from error
    _require_directory(resolved, "runtime root")
    children = {child.name for child in resolved.iterdir()}
    if children != {_MANIFEST, _SITE_PACKAGES}:
        raise BootstrapError("lm-eval runtime root contains missing or extra entries")
    raw, manifest_content = _load_manifest(resolved)
    records = collect_runtime_files(resolved / _SITE_PACKAGES)
    _validate_manifest(raw, records, lock)
    return LmEvalRuntimeReceipt(
        root=resolved,
        runtime_sha256=lock.runtime_sha256,
        manifest_sha256=_sha256(manifest_content),
        identity=dict(lock.identity),
        file_count=len(records),
        lock_sha256=lock.sha256,
    )


def default_lm_eval_runtime_root(apex_root: Path, lock: LmEvalRuntimeLock) -> Path:
    """Return the deterministic local CAS locator for a locked runtime."""

    return (
        apex_root.resolve()
        / ".cache"
        / "apex-runtime"
        / "lm-eval"
        / "sha256"
        / lock.runtime_sha256
    )


__all__ = [
    "LmEvalRuntimeReceipt", "canonical_json", "collect_runtime_files",
    "default_lm_eval_runtime_root", "verify_lm_eval_runtime",
]
