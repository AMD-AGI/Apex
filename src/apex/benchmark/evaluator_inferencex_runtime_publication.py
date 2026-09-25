"""Publish honest Apex-owned evidence for a private InferenceX projection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError, canonical_json_bytes, sha256_json
from apex.runtime import WorkspaceGitIdentityResolver

from .evaluator_handoff import RECEIPT_SCHEMA as HANDOFF_SCHEMA
from .evaluator_inferencex_projection import verify_inferencex_projection
from .evaluator_preparation import PreparedLmEvalExecution


RECEIPT_SCHEMA = "apex.inferencex-runtime-receipt/v2"
RECEIPT_NAME = "inferencex_runtime_receipt.json"
MATERIALIZATION_METHOD = "apex_private_projection"
_EMPTY_STATUS_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb924"
    "27ae41e4649b934ca495991b7852b855"
)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024


def publish_inferencex_projection_evidence(
    prepared: PreparedLmEvalExecution,
    workspace: Path,
    *,
    source_root: Path,
    source_commit: str,
    source_tree: str,
    handoff_receipt_sha256: str,
) -> Mapping[str, object]:
    """Validate the private runtime lineage and publish one immutable receipt."""

    verify_inferencex_projection(prepared.inferencex_projection)
    source = _verified_source(source_root, source_commit, source_tree)
    bindings = _verified_bindings(
        prepared, source, handoff_receipt_sha256
    )
    run_root, workspace_relative, runtime_relative = _run_paths(prepared, workspace)
    if prepared.inferencex_projection.root != run_root / runtime_relative:
        raise _invalid("InferenceX projection is outside its run authority")
    payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "source_root": str(source),
        "source_is_git": True,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_clean": True,
        "source_status_sha256": _EMPTY_STATUS_SHA256,
        "workspace_path": workspace_relative,
        "runtime_path": runtime_relative,
        "materialization_method": MATERIALIZATION_METHOD,
        **bindings,
    }
    receipt = {**payload, "receipt_sha256": sha256_json(payload)}
    _publish_receipt(workspace, canonical_json_bytes(receipt) + b"\n")
    return receipt


def _verified_source(root: Path, commit: str, tree: str) -> Path:
    if _COMMIT.fullmatch(commit) is None or _COMMIT.fullmatch(tree) is None:
        raise _invalid("InferenceX source identity is invalid")
    descriptor = _open_directory(root, "InferenceX source")
    os.close(descriptor)
    source = root.resolve(strict=True)
    identity = WorkspaceGitIdentityResolver().inspect(source)
    if (
        not identity.resolved
        or identity.root != str(source)
        or identity.commit != commit
        or identity.tree != tree
        or identity.dirty_paths
    ):
        raise _invalid("InferenceX source is not the expected clean Git checkout")
    return source


def _verified_bindings(
    prepared: PreparedLmEvalExecution,
    source: Path,
    handoff_receipt_sha256: str,
) -> dict[str, str]:
    authority = prepared.authority_root
    projection = prepared.inferencex_projection
    if (
        projection.receipt != prepared.inferencex_projection_receipt
        or projection.root != authority / "inferencex"
        or prepared.launch_config_receipt.inferencex_source_root != str(source)
        or prepared.launch_config_receipt.inferencex_projection_root
        != str(projection.root)
        or prepared.launch_config_receipt.inferencex_projection_receipt_sha256
        != projection.receipt.sha256
        or prepared.launch_config_receipt.canonical_config_sha256
        != prepared.contract.config_sha256
    ):
        raise _invalid("Prepared InferenceX projection bindings are inconsistent")
    projection_receipt = _read_expected_json(
        authority,
        prepared.inferencex_projection_receipt_path,
        "inferencex_projection_receipt.json",
        projection.receipt.to_dict(),
    )
    launch_receipt = _read_expected_json(
        authority,
        prepared.launch_config_receipt_path,
        "magpie_launch_config_receipt.json",
        prepared.launch_config_receipt.to_dict(),
    )
    contract = _read_expected_json(
        authority,
        prepared.contract_path,
        "execution_contract.json",
        prepared.contract.to_dict(),
    )
    launch = _read_authority_artifact(
        authority, prepared.launch_config_path, "magpie-launch.yaml"
    )
    handoff = _read_authority_artifact(
        authority, authority / "handoff_receipt.json", "handoff_receipt.json"
    )
    handoff_value = _json_object(handoff, "handoff receipt")
    _validate_handoff(handoff_value, handoff_receipt_sha256)
    if hashlib.sha256(launch).hexdigest() != prepared.launch_config_receipt.launch_config_sha256:
        raise _invalid("Magpie launch config differs from its receipt")
    return {
        "projection_receipt_sha256": str(projection_receipt["receipt_sha256"]),
        "projection_manifest_sha256": projection.receipt.projection_manifest_sha256,
        "launch_config_sha256": hashlib.sha256(launch).hexdigest(),
        "launch_config_receipt_sha256": str(launch_receipt["receipt_sha256"]),
        "execution_contract_sha256": str(contract["sha256"]),
        "handoff_receipt_sha256": handoff_receipt_sha256,
    }


def _run_paths(
    prepared: PreparedLmEvalExecution, workspace: Path
) -> tuple[Path, str, str]:
    authority = prepared.authority_root
    if (
        not authority.is_absolute()
        or authority.name != "lm_eval"
        or authority.parent.name != "authority"
    ):
        raise _invalid("Evaluator authority root is invalid")
    authority_descriptor = _open_directory(authority, "evaluator authority")
    os.close(authority_descriptor)
    workspace_descriptor = _open_directory(workspace, "Magpie workspace")
    os.close(workspace_descriptor)
    run_root = authority.parent.parent
    selected_workspace = workspace.resolve(strict=True)
    try:
        workspace_relative = selected_workspace.relative_to(run_root).as_posix()
        runtime_relative = prepared.inferencex_projection.root.relative_to(
            run_root
        ).as_posix()
    except ValueError as error:
        raise _invalid("InferenceX evidence paths do not share one run root") from error
    if (
        not workspace_relative
        or workspace_relative == "."
        or runtime_relative != "authority/lm_eval/inferencex"
    ):
        raise _invalid("InferenceX evidence run-relative paths are invalid")
    return run_root, workspace_relative, runtime_relative


def _read_expected_json(
    authority: Path,
    path: Path,
    name: str,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = _json_object(_read_authority_artifact(authority, path, name), name)
    if dict(value) != dict(expected):
        raise _invalid(f"{name} differs from its prepared receipt")
    return value


def _read_authority_artifact(authority: Path, path: Path, name: str) -> bytes:
    if path != authority / name:
        raise _invalid(f"{name} is outside the evaluator authority")
    directory = _open_directory(authority, "evaluator authority")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        try:
            return _read_immutable(descriptor, name)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise _invalid(f"Cannot securely read {name}") from error
    finally:
        os.close(directory)


def _read_immutable(descriptor: int, name: str) -> bytes:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o222
        or not 0 < before.st_size <= _MAX_ARTIFACT_BYTES
    ):
        raise _invalid(f"{name} is not an immutable single-link artifact")
    content = _read_bounded(descriptor, _MAX_ARTIFACT_BYTES)
    after = os.fstat(descriptor)
    if _identity(before) != _identity(after) or len(content) != after.st_size:
        raise _invalid(f"{name} changed while being read")
    return content


def _read_bounded(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    if len(content) > maximum:
        raise _invalid("Evaluator authority artifact exceeds its size bound")
    return content


def _validate_handoff(value: Mapping[str, Any], expected_sha256: str) -> None:
    ordering = value.get("ordering_ns")
    ordered_values = (
        tuple(
            ordering.get(name)
            for name in (
                "listener_started",
                "request_received",
                "sidecar_started",
                "sidecar_finished",
                "handoff_released",
            )
        )
        if isinstance(ordering, Mapping)
        else ()
    )
    if (
        set(value)
        != {
            "schema",
            "verified",
            "request_sha256",
            "execution_receipt_sha256",
            "ordering_ns",
        }
        or value.get("schema") != HANDOFF_SCHEMA
        or value.get("verified") is not True
        or _DIGEST.fullmatch(str(value.get("request_sha256", ""))) is None
        or _DIGEST.fullmatch(str(value.get("execution_receipt_sha256", ""))) is None
        or not isinstance(ordering, Mapping)
        or set(ordering)
        != {
            "listener_started",
            "request_received",
            "sidecar_started",
            "sidecar_finished",
            "handoff_released",
        }
        or any(isinstance(item, bool) or not isinstance(item, int) for item in ordered_values)
        or ordered_values != tuple(sorted(ordered_values))
        or _DIGEST.fullmatch(expected_sha256) is None
        or sha256_json(value) != expected_sha256
    ):
        raise _invalid("Evaluator handoff receipt is invalid")


def _publish_receipt(workspace: Path, content: bytes) -> None:
    directory = _open_directory(workspace, "Magpie workspace")
    created: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            RECEIPT_NAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o400,
            dir_fd=directory,
        )
        try:
            opened = os.fstat(descriptor)
            created = opened.st_dev, opened.st_ino
            _write_all(descriptor, content)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            final = os.fstat(descriptor)
            if (
                not stat.S_ISREG(final.st_mode)
                or final.st_nlink != 1
                or stat.S_IMODE(final.st_mode) != 0o400
                or final.st_size != len(content)
            ):
                raise _invalid("Published InferenceX runtime receipt is unsafe")
        finally:
            os.close(descriptor)
    except Exception as error:
        if created is not None:
            _cleanup_created(directory, created)
        if isinstance(error, ConfigurationError):
            raise
        raise _invalid("Cannot publish InferenceX runtime receipt") from error
    finally:
        os.close(directory)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise _invalid("Cannot write InferenceX runtime receipt")
        view = view[written:]


def _cleanup_created(directory: int, identity: tuple[int, int]) -> None:
    try:
        observed = os.stat(RECEIPT_NAME, dir_fd=directory, follow_symlinks=False)
        if (observed.st_dev, observed.st_ino) == identity:
            os.unlink(RECEIPT_NAME, dir_fd=directory)
    except OSError:
        pass


def _open_directory(path: Path, label: str) -> int:
    if not path.is_absolute():
        raise _invalid(f"{label} path must be absolute")
    descriptor: int | None = None
    try:
        observed = path.lstat()
        if path.resolve(strict=True) != path or not stat.S_ISDIR(observed.st_mode):
            raise _invalid(f"{label} directory is not canonical")
        descriptor = os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        opened = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise _invalid(f"Cannot securely open {label} directory") from error
    assert descriptor is not None
    if (observed.st_dev, observed.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise _invalid(f"{label} directory identity changed")
    return descriptor


def _json_object(content: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _invalid(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise _invalid(f"{label} must be a JSON object")
    return value


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_inferencex_runtime_publication_invalid")


__all__ = ["publish_inferencex_projection_evidence"]
