"""Validate sidecar runtime engagement and publish workspace evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import ConfigurationError, canonical_json_bytes, sha256_json
from apex.runtime import LmEvalRuntimeReceipt

from .evaluator_execution import LmEvalExecutionReceipt
from .evaluator_preparation import PreparedLmEvalExecution


PROBE_SCHEMA = "apex.lm-eval-runtime-probe/v1"
RECEIPT_SCHEMA = "magpie.lm-eval-runtime-receipt/v1"
EVIDENCE_SCHEMA = "magpie.lm-eval-runtime-evidence/v1"
MANIFEST_SCHEMA = "apex.lm-eval-runtime/v1"
PROBE_NAME = "runtime_probe.json"
MANIFEST_NAME = "lm_eval_runtime_manifest.json"
RECEIPT_NAME = "lm_eval_runtime_receipt.json"
PUBLICATION_NAME = "lm_eval_runtime_publication.json"
MODULE_PATH = "/evaluator/runtime/site-packages/lm_eval/__init__.py"
MODULE_RECORD = "lm_eval/__init__.py"
MAX_PROBE_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_MODULE_BYTES = 8 * 1024 * 1024
MAX_PUBLICATION_BYTES = 2 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ABI = re.compile(r"cpython-([0-9])([0-9]{1,2})")
_IDENTITY_FIELDS = {
    "lm_eval_commit", "lm_eval_tree", "lm_eval_version", "python_abi",
    "python_soabi", "base_image_id", "base_image_repo_digest",
    "inferencex_commit", "inferencex_tree",
}


@dataclass(frozen=True, slots=True)
class PublishedLmEvalRuntimeEvidence:
    """Workspace artifacts and the evidence object embedded in attestation."""

    runtime_probe_sha256: str
    evidence: Mapping[str, object]
    manifest_path: Path
    receipt_path: Path


@dataclass(frozen=True, slots=True)
class _ReadArtifact:
    content: bytes
    mode: int


def publish_lm_eval_runtime_evidence(
    prepared: PreparedLmEvalExecution,
    workspace: Path,
) -> PublishedLmEvalRuntimeEvidence:
    """Validate the exact sidecar runtime probe and publish immutable evidence."""

    runtime = prepared.runtime_receipt
    _validate_runtime_binding(prepared, runtime)
    manifest_artifact = _read_relative(
        runtime.root, (MANIFEST_NAME,), MAX_MANIFEST_BYTES
    )
    manifest_sha256 = _sha256(manifest_artifact.content)
    manifest = _json_object(manifest_artifact.content, "runtime manifest")
    module_record = _validate_manifest(manifest, runtime, manifest_sha256)
    module_artifact = _read_relative(
        runtime.root,
        ("site-packages", "lm_eval", "__init__.py"),
        MAX_MODULE_BYTES,
    )
    _validate_module(module_artifact, module_record)
    probe_artifact = _read_relative(
        prepared.sidecar_root, (PROBE_NAME,), MAX_PROBE_BYTES
    )
    probe = _json_object(probe_artifact.content, "runtime probe")
    _validate_probe(probe, runtime, str(module_record["sha256"]))
    receipt = _runtime_receipt(runtime, manifest_sha256)
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    return _publish(
        workspace,
        manifest_artifact.content,
        receipt_bytes,
        runtime,
        _sha256(probe_artifact.content),
    )


def load_lm_eval_runtime_publication(
    authority_root: Path,
    execution_receipt: LmEvalExecutionReceipt,
) -> Mapping[str, object]:
    """Reload the sidecar authority record bound by an execution receipt."""

    artifact = _read_relative(
        authority_root, (PUBLICATION_NAME,), MAX_PUBLICATION_BYTES
    )
    value = _json_object(artifact.content, "runtime publication")
    if set(value) != {
        "schema", "runtime_probe_sha256", "evidence", "publication_sha256"
    }:
        raise _invalid("Runtime publication schema is invalid")
    evidence = value.get("evidence")
    payload = {
        "schema": value.get("schema"),
        "runtime_probe_sha256": value.get("runtime_probe_sha256"),
        "evidence": evidence,
    }
    if (
        value.get("schema") != "apex.lm-eval-runtime-publication/v1"
        or not isinstance(evidence, Mapping)
        or value.get("runtime_probe_sha256")
        != execution_receipt.runtime_probe_sha256
        or value.get("publication_sha256")
        != execution_receipt.runtime_publication_sha256
        or sha256_json(payload) != value.get("publication_sha256")
    ):
        raise _invalid("Runtime publication differs from execution evidence")
    return dict(evidence)


def _validate_runtime_binding(
    prepared: PreparedLmEvalExecution, runtime: LmEvalRuntimeReceipt
) -> None:
    identity = dict(runtime.identity)
    contract = prepared.contract
    valid_identity = (
        set(identity) == _IDENTITY_FIELDS
        and all(isinstance(value, str) and value for value in identity.values())
        and _ABI.fullmatch(identity.get("python_abi", "")) is not None
    )
    if (
        not valid_identity
        or prepared.runtime_mount != runtime.root
        or contract.runtime_sha256 != runtime.runtime_sha256
        or contract.runtime_manifest_sha256 != runtime.manifest_sha256
        or contract.runtime_lock_sha256 != runtime.lock_sha256
        or contract.image_id != identity.get("base_image_id")
        or contract.image_repo_digest != identity.get("base_image_repo_digest")
        or runtime.file_count <= 0
    ):
        raise _invalid("Prepared runtime receipt differs from its execution contract")


def _validate_manifest(
    value: Mapping[str, Any],
    runtime: LmEvalRuntimeReceipt,
    observed_sha256: str,
) -> Mapping[str, Any]:
    files = value.get("files")
    if (
        set(value) != {"schema", "runtime_sha256", "site_packages", "identity", "files"}
        or value.get("schema") != MANIFEST_SCHEMA
        or value.get("runtime_sha256") != runtime.runtime_sha256
        or value.get("site_packages") != "site-packages"
        or value.get("identity") != dict(runtime.identity)
        or observed_sha256 != runtime.manifest_sha256
        or not isinstance(files, list)
        or len(files) != runtime.file_count
    ):
        raise _invalid("Runtime manifest differs from its verified receipt")
    records = [_manifest_record(item) for item in files]
    paths = [str(item["path"]) for item in records]
    if len(paths) != len(set(paths)) or paths.count(MODULE_RECORD) != 1:
        raise _invalid("Runtime manifest file identities are invalid")
    return records[paths.index(MODULE_RECORD)]


def _manifest_record(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path", "size_bytes", "mode", "sha256"
    }:
        raise _invalid("Runtime manifest file record is invalid")
    path = value.get("path")
    relative = PurePosixPath(str(path))
    size = value.get("size_bytes")
    mode = value.get("mode")
    if (
        not isinstance(path, str)
        or not path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != path
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or isinstance(mode, bool)
        or not isinstance(mode, int)
        or not 0 <= mode <= 0o7777
        or not isinstance(value.get("sha256"), str)
        or _DIGEST.fullmatch(str(value.get("sha256"))) is None
    ):
        raise _invalid("Runtime manifest file record values are invalid")
    return value


def _validate_module(
    artifact: _ReadArtifact, record: Mapping[str, Any]
) -> None:
    if (
        len(artifact.content) != record.get("size_bytes")
        or artifact.mode != record.get("mode")
        or _sha256(artifact.content) != record.get("sha256")
    ):
        raise _invalid("Runtime lm_eval module differs from its manifest")


def _validate_probe(
    value: Mapping[str, Any],
    runtime: LmEvalRuntimeReceipt,
    module_sha256: str,
) -> None:
    python = value.get("python")
    lm_eval = value.get("lm_eval")
    python_path = value.get("python_path")
    if (
        set(value) != {"schema", "python", "lm_eval", "python_path"}
        or value.get("schema") != PROBE_SCHEMA
        or not isinstance(python, Mapping)
        or set(python) != {"implementation", "version", "executable"}
        or not isinstance(lm_eval, Mapping)
        or set(lm_eval) != {"version", "module_path", "module_sha256"}
        or not isinstance(python_path, list)
        or not python_path
        or any(not isinstance(item, str) for item in python_path)
    ):
        raise _invalid("Runtime probe schema is invalid")
    version = python.get("version")
    abi = _ABI.fullmatch(str(runtime.identity.get("python_abi", "")))
    valid_version = (
        isinstance(version, list)
        and len(version) == 3
        and all(isinstance(item, int) and not isinstance(item, bool) for item in version)
        and all(item >= 0 for item in version)
        and abi is not None
        and version[:2] == [int(abi.group(1)), int(abi.group(2))]
    )
    if (
        python.get("implementation") != "cpython"
        or not valid_version
        or not isinstance(python.get("executable"), str)
        or not str(python.get("executable")).startswith("/")
        or "/evaluator/runtime/site-packages" not in python_path
        or lm_eval.get("version") != runtime.identity.get("lm_eval_version")
        or lm_eval.get("module_path") != MODULE_PATH
        or lm_eval.get("module_sha256") != module_sha256
    ):
        raise _invalid("Runtime probe differs from the locked runtime")


def _runtime_receipt(
    runtime: LmEvalRuntimeReceipt, manifest_sha256: str
) -> dict[str, object]:
    return {
        "schema": RECEIPT_SCHEMA,
        "runtime_sha256": runtime.runtime_sha256,
        "identity": dict(runtime.identity),
        "manifest_sha256": manifest_sha256,
        "site_packages": "site-packages",
        "python_abi": runtime.identity["python_abi"],
        "lm_eval_version": runtime.identity["lm_eval_version"],
        "lm_eval_module": "site-packages/lm_eval/__init__.py",
        "execution_mode": "docker",
        "read_only_mount": True,
        "verified": True,
    }


def _publish(
    workspace: Path,
    manifest_bytes: bytes,
    receipt_bytes: bytes,
    runtime: LmEvalRuntimeReceipt,
    probe_sha256: str,
) -> PublishedLmEvalRuntimeEvidence:
    directory = _open_directory(workspace)
    created: list[tuple[str, int, int]] = []
    try:
        manifest_path = _write_exclusive(
            directory, workspace, MANIFEST_NAME, manifest_bytes, created
        )
        receipt_path = _write_exclusive(
            directory, workspace, RECEIPT_NAME, receipt_bytes, created
        )
    except Exception as error:
        _cleanup_created(directory, created)
        if isinstance(error, ConfigurationError):
            raise
        raise _invalid("Cannot publish lm-eval runtime evidence") from error
    finally:
        os.close(directory)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "requested": True,
        "status": "verified",
        "verified": True,
        "evidence_present": True,
        "runtime_sha256": runtime.runtime_sha256,
        "identity": dict(runtime.identity),
        "mount_mode": "read_only",
        "manifest_artifact": _declaration(MANIFEST_NAME, manifest_bytes),
        "receipt_artifact": _declaration(RECEIPT_NAME, receipt_bytes),
        "errors": [],
    }
    return PublishedLmEvalRuntimeEvidence(
        probe_sha256, evidence, manifest_path, receipt_path
    )


def _read_relative(
    root: Path, parts: tuple[str, ...], maximum: int
) -> _ReadArtifact:
    directory = _open_directory(root)
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        try:
            return _read_descriptor(descriptor, maximum)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise _invalid(f"Cannot securely read runtime artifact: {parts[-1]}") from error
    finally:
        os.close(directory)


def _read_descriptor(descriptor: int, maximum: int) -> _ReadArtifact:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o222
        or not 0 <= before.st_size <= maximum
    ):
        raise _invalid("Runtime artifact identity or size is invalid")
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    after = os.fstat(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if len(content) > maximum or identity != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(content) != after.st_size:
        raise _invalid("Runtime artifact changed while being read")
    return _ReadArtifact(content, stat.S_IMODE(after.st_mode))


def _open_directory(path: Path) -> int:
    if not path.is_absolute():
        raise _invalid("Runtime publication path must be absolute")
    descriptor: int | None = None
    try:
        resolved = path.resolve(strict=True)
        observed = path.lstat()
        if resolved != path or not stat.S_ISDIR(observed.st_mode):
            raise _invalid("Runtime publication directory is not canonical")
        descriptor = os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        opened = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise _invalid("Cannot securely open runtime publication directory") from error
    assert descriptor is not None
    if (observed.st_dev, observed.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise _invalid("Runtime publication directory identity changed")
    return descriptor


def _write_exclusive(
    directory: int,
    workspace: Path,
    name: str,
    content: bytes,
    created: list[tuple[str, int, int]],
) -> Path:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
        dir_fd=directory,
    )
    try:
        opened = os.fstat(descriptor)
        created.append((name, opened.st_dev, opened.st_ino))
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise _invalid("Cannot write runtime publication artifact")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o400
            or final.st_size != len(content)
        ):
            raise _invalid("Published runtime artifact identity is invalid")
    finally:
        os.close(descriptor)
    return workspace / name


def _cleanup_created(
    directory: int, created: list[tuple[str, int, int]]
) -> None:
    for name, device, inode in reversed(created):
        try:
            observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (observed.st_dev, observed.st_ino) == (device, inode):
                os.unlink(name, dir_fd=directory)
        except OSError:
            pass


def _declaration(name: str, content: bytes) -> dict[str, object]:
    return {"path": name, "size_bytes": len(content), "sha256": _sha256(content)}


def _json_object(content: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _invalid(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise _invalid(f"{label} must be a JSON object")
    return value


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_runtime_publication_invalid")


__all__ = [
    "PublishedLmEvalRuntimeEvidence",
    "load_lm_eval_runtime_publication",
    "publish_lm_eval_runtime_evidence",
]
