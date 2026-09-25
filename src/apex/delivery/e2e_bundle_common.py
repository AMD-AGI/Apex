"""Shared immutable types and primitives for E2E delivery bundles."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from apex.core import (
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
)

from .e2e_models import (
    BuildRecipeLock,
    BundleProvenanceLock,
    DerivedImageIdentity,
    SourceRepositoryLock,
    replay_semantics,
    safe_bundle_path,
)
from .e2e_receipts import PrimaryVerificationEvidence


SCHEMA = "apex.verified-patch-bundle.v1"
SECOND_RECEIPT_ROLE = "second_clean_replay_receipt"
CONFIG_PATHS = {
    "benchmark_original": "config/benchmark.original.yaml",
    "benchmark_measurement": "config/benchmark.measurement.resolved.yaml",
    "benchmark_diagnostic": "config/benchmark.diagnostic.resolved.yaml",
    "benchmark_replay": "config/benchmark.replay.yaml",
}
PRIMARY_RECEIPT_PATHS = {
    "primary_build_receipt": "verification/primary.build.receipt.json",
    "primary_engagement_receipt": "verification/primary.engagement.receipt.json",
    "primary_benchmark_receipt": "verification/primary.benchmark.receipt.json",
    "primary_safety_receipt": "verification/primary.safety.receipt.json",
}


@dataclass(frozen=True, slots=True)
class E2EPatchBundle:
    """A statically verified delivery bundle, candidate or final."""

    path: Path
    bundle_id: str
    digest: str
    verified: bool
    manifest: Mapping[str, Any]
    repositories: tuple[SourceRepositoryLock, ...]
    recipe: BuildRecipeLock
    derived_image: DerivedImageIdentity
    provenance: BundleProvenanceLock
    primary_evidence: PrimaryVerificationEvidence
    config_paths: Mapping[str, Path]
    primary_receipt_paths: Mapping[str, Path]
    sbom_path: Path
    verification_result: Mapping[str, Any] | None = None


def detect_bundle_kind(bundle_dir: Path) -> str:
    """Safely distinguish the stable AKA kernel and E2E bundle contracts."""

    root = bundle_dir.resolve(strict=True)
    manifest = root / "bundle.json"
    unsafe = (
        not root.is_dir()
        or bundle_dir.is_symlink()
        or manifest.is_symlink()
        or not manifest.is_file()
    )
    if unsafe:
        raise IntegrityError("Bundle path or manifest is unsafe", "invalid_bundle_path")
    value = read_json(manifest, "invalid_bundle_manifest")
    if value.get("schema") == SCHEMA:
        return "e2e"
    if value.get("schema_version") == 1 and "task_id" in value and "patches" in value:
        return "kernel"
    raise IntegrityError("Bundle schema is unknown", "invalid_bundle_manifest")


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise IntegrityError(f"Bundle path already exists: {path}", "bundle_path_collision")
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_entry(path: str, role: str, content: bytes) -> dict[str, str]:
    return {
        "path": safe_bundle_path(path),
        "role": role,
        "sha256": sha256_bytes(content),
    }


def digest_scope(manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise IntegrityError("Bundle files must be an ordered list", "invalid_e2e_bundle")
    scoped_files = []
    for entry in files:
        if not isinstance(entry, Mapping):
            raise IntegrityError("Bundle file entry is invalid", "invalid_e2e_bundle")
        if entry.get("role") != SECOND_RECEIPT_ROLE:
            scoped_files.append(dict(entry))
    excluded = {"bundle_digest", "verified", "verification_receipt", "files"}
    return {key: value for key, value in manifest.items() if key not in excluded} | {
        "files": scoped_files
    }


def compute_e2e_bundle_digest(manifest: Mapping[str, Any], root: Path) -> str:
    """Hash delivery inputs; the second verifier receipt signs this stable scope."""

    scoped = digest_scope(manifest)
    content = bytearray(canonical_json_bytes(scoped))
    for entry in scoped["files"]:
        path = root / safe_bundle_path(str(entry["path"]))
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(
                "Bundle payload file is missing or unsafe",
                "bundle_file_set_mismatch",
            )
        content.extend(path.read_bytes())
    return sha256_bytes(bytes(content))


def verify_replay_config_invariants(
    measurement_path: Path,
    replay_path: Path,
    *,
    expected_image_locator: str,
) -> tuple[str, str, str]:
    """Prove replay differs from measurement only by its image locator/metadata."""

    measurement = _load_yaml(measurement_path)
    replay = _load_yaml(replay_path)
    measurement_semantics = replay_semantics(measurement)
    replay_semantics_value = replay_semantics(replay)
    if measurement_semantics != replay_semantics_value:
        raise IntegrityError("Replay workload/quality protocol changed", "replay_config_tampered")
    benchmark = replay.get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("docker_image") != expected_image_locator:
        raise IntegrityError("Replay does not select the derived image", "replay_image_mismatch")
    return (
        sha256_file(measurement_path),
        sha256_file(replay_path),
        sha256_json(_workload_projection(measurement)),
    )


def _workload_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    """Match the benchmark module's workload hash while preserving strict replay comparison."""

    benchmark = document.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise IntegrityError("Benchmark config lacks benchmark mapping", "invalid_replay_config")
    projected = copy.deepcopy(dict(benchmark))
    projected.pop("docker_image", None)
    projected.pop("profiler", None)
    projected.pop("gap_analysis", None)
    projected.pop("run_kind", None)
    return projected


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as error:
        raise IntegrityError(
            f"Cannot decode benchmark config: {path.name}",
            "invalid_replay_config",
        ) from error
    if not isinstance(value, Mapping):
        raise IntegrityError("Benchmark config must be a mapping", "invalid_replay_config")
    return value


def read_json(path: Path, reason: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"Cannot decode {path.name}", reason) from error
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{path.name} must be an object", reason)
    return value


def role_paths(manifest: Mapping[str, Any], root: Path) -> dict[str, Path]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise IntegrityError("Bundle files are invalid", "invalid_e2e_bundle")
    roles: dict[str, Path] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise IntegrityError("Bundle file entry is invalid", "invalid_e2e_bundle")
        role = str(entry.get("role", ""))
        path = root / safe_bundle_path(str(entry.get("path", "")))
        if not role or role in roles:
            raise IntegrityError(
                "Bundle file roles must be non-empty and unique",
                "invalid_e2e_bundle",
            )
        roles[role] = path
    return roles


__all__ = [
    "CONFIG_PATHS",
    "E2EPatchBundle",
    "PRIMARY_RECEIPT_PATHS",
    "SCHEMA",
    "SECOND_RECEIPT_ROLE",
    "compute_e2e_bundle_digest",
    "detect_bundle_kind",
    "file_entry",
    "fsync_dir",
    "read_json",
    "role_paths",
    "verify_replay_config_invariants",
    "write_new",
]
