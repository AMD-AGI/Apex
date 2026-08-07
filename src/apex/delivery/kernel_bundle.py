"""Build and verify the generic source-only kernel bundle consumed by AKA."""

from __future__ import annotations

import difflib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes, sha256_file
from apex.intake import ResolvedTaskSpec


@dataclass(frozen=True, slots=True)
class KernelBundle:
    task_id: str
    path: Path
    digest: str
    manifest: Mapping[str, Any]
    changed_files: tuple[str, ...]


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or any(part in {"", "."} for part in path.parts):
        raise IntegrityError(f"Unsafe bundle path: {value!r}", "unsafe_bundle_path")
    return path.as_posix()


def _read_source(path: Path, *, relative: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"Candidate source is not a regular file: {relative}", "invalid_candidate_source")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"Binary source bundles are unsupported: {relative}", "binary_source_unsupported") from error


def _unified_patch(baseline: str, candidate: str, relative: str) -> bytes:
    lines = difflib.unified_diff(
        baseline.splitlines(keepends=True),
        candidate.splitlines(keepends=True),
        fromfile=f"a/{relative}",
        tofile=f"b/{relative}",
        lineterm="\n",
    )
    return "".join(lines).encode("utf-8")


def _write_synced(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def compute_bundle_digest(manifest: Mapping[str, Any], bundle_root: Path) -> str:
    """Hash canonical manifest bytes followed by patch bytes in manifest order."""

    content = bytearray(canonical_json_bytes(manifest))
    patches = manifest.get("patches")
    if not isinstance(patches, Sequence) or isinstance(patches, (str, bytes)):
        raise IntegrityError("bundle patches must be an ordered list", "invalid_bundle_manifest")
    for entry in patches:
        if not isinstance(entry, Mapping):
            raise IntegrityError("bundle patch entry must be an object", "invalid_bundle_manifest")
        relative = _safe_relative(str(entry.get("path", "")))
        content.extend((bundle_root / relative).read_bytes())
    return sha256_bytes(bytes(content))


def build_kernel_bundle(
    resolved: ResolvedTaskSpec,
    *,
    candidate_root: Path,
    bundle_dir: Path,
) -> KernelBundle:
    """Freeze allowed text-source changes into a new atomic bundle directory."""

    if bundle_dir.exists():
        raise ContractError(f"Bundle directory already exists: {bundle_dir}", "bundle_exists")
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_root = candidate_root.resolve(strict=True)
    if not candidate_root.is_dir():
        raise ContractError("candidate_root is not a directory", "candidate_root_invalid")

    changed: list[str] = []
    patch_entries: list[dict[str, str]] = []
    candidate_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix=".apex-bundle-", dir=bundle_dir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for index, relative in enumerate(resolved.task.editable_files, start=1):
            relative = _safe_relative(relative)
            baseline_path = resolved.workspace.joinpath(*relative.split("/"))
            candidate_path = candidate_root.joinpath(*relative.split("/"))
            baseline = _read_source(baseline_path, relative=relative)
            candidate = _read_source(candidate_path, relative=relative)
            candidate_hashes[relative] = sha256_file(candidate_path)
            if baseline == candidate:
                continue
            patch = _unified_patch(baseline, candidate, relative)
            patch_path = f"patches/{index:03d}-{Path(relative).name}.patch"
            _write_synced(staging / patch_path, patch)
            changed.append(relative)
            patch_entries.append({"path": patch_path, "sha256": sha256_bytes(patch)})

        if not changed:
            raise ContractError("Candidate contains no allowed source changes", "no_changed_files")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "task_id": resolved.task.task_id,
            "baseline": {
                "resolution_hash": resolved.resolution_hash,
                "file_hashes": dict(sorted(resolved.baseline_file_hashes.items())),
            },
            "changed_files": changed,
            "candidate_file_hashes": {path: candidate_hashes[path] for path in changed},
            "patches": patch_entries,
            "delivery": {"mode": "bundle", "applied": False},
        }
        _write_synced(staging / "bundle.json", canonical_json_bytes(manifest) + b"\n")
        _fsync_dir(staging / "patches")
        _fsync_dir(staging)
        os.replace(staging, bundle_dir)
        _fsync_dir(bundle_dir.parent)

    digest = compute_bundle_digest(manifest, bundle_dir)
    return KernelBundle(
        task_id=resolved.task.task_id,
        path=bundle_dir,
        digest=digest,
        manifest=manifest,
        changed_files=tuple(changed),
    )


def _declared_files(manifest: Mapping[str, Any]) -> set[str]:
    files = {"bundle.json"}
    patches = manifest.get("patches")
    if not isinstance(patches, list):
        raise IntegrityError("bundle patches must be a list", "invalid_bundle_manifest")
    for entry in patches:
        if not isinstance(entry, Mapping):
            raise IntegrityError("bundle patch entry must be an object", "invalid_bundle_manifest")
        files.add(_safe_relative(str(entry.get("path", ""))))
    return files


def load_and_verify_kernel_bundle(bundle_dir: Path, *, expected_digest: str | None = None) -> KernelBundle:
    """Fail closed on symlinks, undeclared files, hashes, schema, or tree digest."""

    root = bundle_dir.resolve(strict=True)
    if not root.is_dir():
        raise IntegrityError("bundle path is not a directory", "invalid_bundle_path")
    manifest_path = root / "bundle.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise IntegrityError("bundle manifest is missing or unsafe", "invalid_bundle_manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IntegrityError("bundle manifest cannot be decoded", "invalid_bundle_manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise IntegrityError("unsupported bundle manifest", "invalid_bundle_manifest")

    declared = _declared_files(manifest)
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise IntegrityError("bundle may not contain symlinks", "bundle_symlink")
        if not path.is_file() or path.stat().st_nlink != 1:
            raise IntegrityError("bundle may not contain hard links", "bundle_hardlink")
        actual.add(path.relative_to(root).as_posix())
    if actual != declared:
        raise IntegrityError("bundle has missing or undeclared files", "bundle_file_set_mismatch")

    for entry in manifest["patches"]:
        patch_path = root / _safe_relative(str(entry["path"]))
        if sha256_file(patch_path) != str(entry.get("sha256", "")).removeprefix("sha256:"):
            raise IntegrityError("bundle patch digest mismatch", "bundle_patch_digest_mismatch")
    digest = compute_bundle_digest(manifest, root)
    if expected_digest and digest != expected_digest.removeprefix("sha256:"):
        raise IntegrityError("bundle tree digest mismatch", "bundle_digest_mismatch")
    changed_files = manifest.get("changed_files")
    if not isinstance(changed_files, list) or any(not isinstance(path, str) for path in changed_files):
        raise IntegrityError("bundle changed_files is invalid", "invalid_bundle_manifest")
    safe_changed_files = tuple(_safe_relative(path) for path in changed_files)
    return KernelBundle(
        task_id=str(manifest.get("task_id", "")),
        path=root,
        digest=digest,
        manifest=manifest,
        changed_files=safe_changed_files,
    )
