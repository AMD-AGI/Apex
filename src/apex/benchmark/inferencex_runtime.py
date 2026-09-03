"""Independent validation of Magpie's run-scoped InferenceX runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import canonical_json_bytes, sha256_json


_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_V1_SCHEMA = "magpie.inferencex-runtime-receipt/v1"
_V2_SCHEMA = "apex.inferencex-runtime-receipt/v2"
_RECEIPT_NAME = "inferencex_runtime_receipt.json"
_RUNTIME_NAME = "inferencex_runtime"
_V1_METHOD = "git_private_index_checkout"
_V2_METHOD = "apex_private_projection"
_V1_FIELDS = {
    "schema", "source_root", "source_is_git", "source_commit", "source_tree",
    "source_clean", "source_status_sha256", "source_status_unchanged",
    "runtime_path", "materialization_method",
}
_V2_FIELDS = {
    "schema", "source_root", "source_is_git", "source_commit", "source_tree",
    "source_clean", "source_status_sha256", "workspace_path", "runtime_path",
    "materialization_method", "projection_receipt_sha256",
    "projection_manifest_sha256", "launch_config_sha256",
    "launch_config_receipt_sha256", "execution_contract_sha256",
    "handoff_receipt_sha256", "receipt_sha256",
}
_EMPTY_STATUS_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb924"
    "27ae41e4649b934ca495991b7852b855"
)
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class InferenceXRuntimeEvidence:
    """Normalized source identity and private materialization evidence."""

    required: bool
    passed: bool
    source_root: Path | None
    source_commit: str | None
    runtime_path: Path | None
    receipt_path: Path | None
    source_status_sha256: str | None
    error: str | None = None
    source_tree: str | None = None


def parse_inferencex_runtime_evidence(
    report: Mapping[str, Any],
    report_path: Path,
    *,
    expected_source_root: Path | None,
    expected_commit: str | None,
    expected_tree: str | None,
) -> InferenceXRuntimeEvidence:
    """Validate the report value against its protected workspace receipt."""

    if expected_source_root is None and expected_commit is None and expected_tree is None:
        return _not_requested(report)
    if expected_source_root is None or expected_commit is None or expected_tree is None:
        return _failed(expected_source_root, expected_commit, "incomplete_inferencex_expectation")
    if (
        not expected_source_root.is_absolute()
        or not _COMMIT.fullmatch(expected_commit)
        or not _COMMIT.fullmatch(expected_tree)
    ):
        return _failed(expected_source_root, expected_commit, "invalid_inferencex_expectation")

    value = report.get("inferencex_runtime_receipt")
    if not isinstance(value, Mapping):
        return _failed(expected_source_root, expected_commit, "inferencex_runtime_evidence_missing")
    receipt_path, receipt_bytes, receipt_mode, artifact_error = _workspace_receipt(
        report_path.parent
    )
    if artifact_error:
        return _failed(expected_source_root, expected_commit, artifact_error)
    assert receipt_path is not None and receipt_bytes is not None
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeError, json.JSONDecodeError):
        return _failed(expected_source_root, expected_commit, "invalid_inferencex_runtime_receipt")
    if not isinstance(receipt, Mapping) or dict(value) != dict(receipt):
        return _failed(expected_source_root, expected_commit, "inferencex_runtime_report_receipt_mismatch")

    expected_root = expected_source_root.resolve()
    if receipt.get("schema") == _V2_SCHEMA:
        return _parse_v2(
            receipt,
            receipt_bytes,
            receipt_mode,
            receipt_path,
            report_path.parent.resolve(),
            expected_root,
            expected_commit,
            expected_tree,
        )
    return _parse_v1(
        receipt,
        receipt_path,
        report_path.parent.resolve(),
        expected_root,
        expected_commit,
        expected_tree,
    )


def _parse_v1(
    receipt: Mapping[str, Any],
    receipt_path: Path,
    workspace: Path,
    expected_root: Path,
    expected_commit: str,
    expected_tree: str,
) -> InferenceXRuntimeEvidence:
    status_digest = receipt.get("source_status_sha256")
    runtime_path = workspace / _RUNTIME_NAME
    valid = (
        set(receipt) == _V1_FIELDS
        and receipt.get("schema") == _V1_SCHEMA
        and receipt.get("source_root") == str(expected_root)
        and receipt.get("source_is_git") is True
        and receipt.get("source_commit") == expected_commit
        and receipt.get("source_tree") == expected_tree
        and receipt.get("source_clean") is True
        and isinstance(status_digest, str)
        and bool(_DIGEST.fullmatch(status_digest))
        and status_digest == _EMPTY_STATUS_SHA256
        and receipt.get("source_status_unchanged") is True
        and receipt.get("runtime_path") == _RUNTIME_NAME
        and receipt.get("materialization_method") == _V1_METHOD
        and _safe_directory(runtime_path)
    )
    if not valid:
        return _failed(expected_root, expected_commit, "invalid_inferencex_runtime_receipt")
    return InferenceXRuntimeEvidence(
        required=True,
        passed=True,
        source_root=expected_root,
        source_commit=expected_commit,
        runtime_path=runtime_path,
        receipt_path=receipt_path,
        source_status_sha256=status_digest,
        source_tree=expected_tree,
    )


def _parse_v2(
    receipt: Mapping[str, Any],
    receipt_bytes: bytes,
    receipt_mode: int | None,
    receipt_path: Path,
    workspace: Path,
    expected_root: Path,
    expected_commit: str,
    expected_tree: str,
) -> InferenceXRuntimeEvidence:
    payload = dict(receipt)
    receipt_digest = payload.pop("receipt_sha256", None)
    digests = (
        "projection_receipt_sha256",
        "projection_manifest_sha256",
        "launch_config_sha256",
        "launch_config_receipt_sha256",
        "execution_contract_sha256",
        "handoff_receipt_sha256",
    )
    base_valid = (
        set(receipt) == _V2_FIELDS
        and receipt.get("schema") == _V2_SCHEMA
        and receipt.get("source_root") == str(expected_root)
        and receipt.get("source_is_git") is True
        and receipt.get("source_commit") == expected_commit
        and receipt.get("source_tree") == expected_tree
        and receipt.get("source_clean") is True
        and receipt.get("source_status_sha256") == _EMPTY_STATUS_SHA256
        and receipt.get("materialization_method") == _V2_METHOD
        and all(_digest(receipt.get(name)) for name in digests)
        and _digest(receipt_digest)
        and sha256_json(payload) == receipt_digest
        and receipt_mode == 0o400
        and receipt_bytes == canonical_json_bytes(receipt) + b"\n"
    )
    paths = _v2_paths(workspace, receipt)
    if not base_valid or paths is None:
        return _failed(expected_root, expected_commit, "invalid_inferencex_runtime_receipt")
    runtime_path, authority = paths
    if not _validate_authority(receipt, authority, runtime_path, expected_root):
        return _failed(expected_root, expected_commit, "invalid_inferencex_runtime_receipt")
    return InferenceXRuntimeEvidence(
        required=True,
        passed=True,
        source_root=expected_root,
        source_commit=expected_commit,
        runtime_path=runtime_path,
        receipt_path=receipt_path,
        source_status_sha256=_EMPTY_STATUS_SHA256,
        source_tree=expected_tree,
    )


def _workspace_receipt(
    workspace: Path,
) -> tuple[Path | None, bytes | None, int | None, str | None]:
    workspace = workspace.resolve()
    receipt_path = workspace / _RECEIPT_NAME
    try:
        descriptor = os.open(
            receipt_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OSError("unsafe receipt")
        content = _read_bounded(descriptor, _MAX_RECEIPT_BYTES)
        after = os.fstat(descriptor)
    except OSError:
        return None, None, None, "inferencex_runtime_receipt_missing"
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    if _identity(before) != _identity(after) or len(content) != after.st_size:
        return None, None, None, "invalid_inferencex_runtime_receipt"
    return receipt_path.resolve(), content, stat.S_IMODE(after.st_mode), None


def _v2_paths(
    workspace: Path, receipt: Mapping[str, Any]
) -> tuple[Path, Path] | None:
    relative_workspace = _relative(receipt.get("workspace_path"))
    relative_runtime = _relative(receipt.get("runtime_path"))
    if (
        relative_workspace is None
        or relative_runtime is None
        or relative_runtime.as_posix() != "authority/lm_eval/inferencex"
    ):
        return None
    run_root = workspace
    for _part in relative_workspace.parts:
        run_root = run_root.parent
    if run_root.joinpath(*relative_workspace.parts) != workspace:
        return None
    runtime = run_root.joinpath(*relative_runtime.parts)
    if not _safe_directory_chain(run_root, relative_runtime.parts):
        return None
    return runtime, runtime.parent


def _validate_authority(
    receipt: Mapping[str, Any],
    authority: Path,
    runtime: Path,
    source_root: Path,
) -> bool:
    projection = _read_authority_json(authority, "inferencex_projection_receipt.json")
    launch_receipt = _read_authority_json(
        authority, "magpie_launch_config_receipt.json"
    )
    contract = _read_authority_json(authority, "execution_contract.json")
    handoff = _read_authority_json(authority, "handoff_receipt.json")
    launch = _read_authority(authority, "magpie-launch.yaml")
    if None in (projection, launch_receipt, contract, handoff, launch):
        return False
    assert isinstance(projection, Mapping)
    assert isinstance(launch_receipt, Mapping)
    assert isinstance(contract, Mapping)
    assert isinstance(handoff, Mapping)
    assert isinstance(launch, bytes)
    inferencex = projection.get("inferencex")
    run = contract.get("run")
    return bool(
        _semantic_digest(projection, "receipt_sha256")
        == receipt.get("projection_receipt_sha256")
        and isinstance(inferencex, Mapping)
        and inferencex.get("commit") == receipt.get("source_commit")
        and inferencex.get("tree") == receipt.get("source_tree")
        and projection.get("projection_manifest_sha256")
        == receipt.get("projection_manifest_sha256")
        and _semantic_digest(launch_receipt, "receipt_sha256")
        == receipt.get("launch_config_receipt_sha256")
        and launch_receipt.get("launch_config_sha256")
        == receipt.get("launch_config_sha256")
        and launch_receipt.get("inferencex_source_root") == str(source_root)
        and launch_receipt.get("inferencex_projection_root") == str(runtime)
        and launch_receipt.get("inferencex_projection_receipt_sha256")
        == receipt.get("projection_receipt_sha256")
        and hashlib.sha256(launch).hexdigest() == receipt.get("launch_config_sha256")
        and _semantic_digest(contract, "sha256")
        == receipt.get("execution_contract_sha256")
        and isinstance(run, Mapping)
        and run.get("config_sha256")
        == launch_receipt.get("canonical_config_sha256")
        and sha256_json(handoff) == receipt.get("handoff_receipt_sha256")
        and handoff.get("schema") == "apex.evaluator-handoff-receipt/v1"
        and handoff.get("verified") is True
    )


def _read_authority_json(authority: Path, name: str) -> Mapping[str, Any] | None:
    content = _read_authority(authority, name)
    if content is None:
        return None
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping) or content != canonical_json_bytes(value) + b"\n":
        return None
    return value


def _read_authority(authority: Path, name: str) -> bytes | None:
    try:
        directory = os.open(
            authority,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            return None
        content = _read_bounded(descriptor, _MAX_RECEIPT_BYTES)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after) or len(content) != after.st_size:
            return None
        return content
    except OSError:
        return None
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
        if "directory" in locals():
            os.close(directory)


def _semantic_digest(value: Mapping[str, Any], field: str) -> str | None:
    payload = dict(value)
    observed = payload.pop(field, None)
    return str(observed) if _digest(observed) and sha256_json(payload) == observed else None


def _relative(value: object) -> PurePosixPath | None:
    if not isinstance(value, str) or not value:
        return None
    selected = PurePosixPath(value)
    if (
        selected.is_absolute()
        or selected.as_posix() != value
        or any(part in {"", ".", ".."} for part in selected.parts)
    ):
        return None
    return selected


def _safe_directory_chain(root: Path, parts: tuple[str, ...]) -> bool:
    current = root
    try:
        for part in parts:
            current /= part
            observed = current.lstat()
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                return False
    except OSError:
        return False
    return True


def _safe_directory(path: Path) -> bool:
    try:
        observed = path.lstat()
        return not stat.S_ISLNK(observed.st_mode) and stat.S_ISDIR(observed.st_mode)
    except OSError:
        return False


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
        raise OSError("artifact exceeds bound")
    return content


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _not_requested(report: Mapping[str, Any]) -> InferenceXRuntimeEvidence:
    if report.get("inferencex_runtime_receipt") is not None:
        return InferenceXRuntimeEvidence(
            False, False, None, None, None, None, None,
            "unexpected_inferencex_runtime_evidence",
        )
    return InferenceXRuntimeEvidence(False, True, None, None, None, None, None)


def _failed(
    source_root: Path | None,
    commit: str | None,
    error: str,
) -> InferenceXRuntimeEvidence:
    return InferenceXRuntimeEvidence(
        True, False, source_root, commit, None, None, None, error
    )


__all__ = [
    "InferenceXRuntimeEvidence",
    "parse_inferencex_runtime_evidence",
]
