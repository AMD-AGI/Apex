"""Independent validation of Magpie's run-scoped InferenceX runtime."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RECEIPT_SCHEMA = "magpie.inferencex-runtime-receipt/v1"
_RECEIPT_NAME = "inferencex_runtime_receipt.json"
_RUNTIME_NAME = "inferencex_runtime"
_METHOD = "git_private_index_checkout"
_RECEIPT_FIELDS = {
    "schema", "source_root", "source_is_git", "source_commit", "source_tree",
    "source_clean", "source_status_sha256", "source_status_unchanged",
    "runtime_path", "materialization_method",
}


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
    receipt_path, runtime_path, artifact_error = _workspace_artifacts(report_path.parent)
    if artifact_error:
        return _failed(expected_source_root, expected_commit, artifact_error)
    assert receipt_path is not None and runtime_path is not None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _failed(expected_source_root, expected_commit, "invalid_inferencex_runtime_receipt")
    if not isinstance(receipt, Mapping) or dict(value) != dict(receipt):
        return _failed(expected_source_root, expected_commit, "inferencex_runtime_report_receipt_mismatch")

    expected_root = expected_source_root.resolve()
    status_digest = receipt.get("source_status_sha256")
    valid = (
        set(receipt) == _RECEIPT_FIELDS
        and receipt.get("schema") == _RECEIPT_SCHEMA
        and receipt.get("source_root") == str(expected_root)
        and receipt.get("source_is_git") is True
        and receipt.get("source_commit") == expected_commit
        and receipt.get("source_tree") == expected_tree
        and receipt.get("source_clean") is True
        and isinstance(status_digest, str)
        and bool(_DIGEST.fullmatch(status_digest))
        and status_digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        and receipt.get("source_status_unchanged") is True
        and receipt.get("runtime_path") == _RUNTIME_NAME
        and receipt.get("materialization_method") == _METHOD
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


def _workspace_artifacts(
    workspace: Path,
) -> tuple[Path | None, Path | None, str | None]:
    workspace = workspace.resolve()
    receipt_path = workspace / _RECEIPT_NAME
    runtime_path = workspace / _RUNTIME_NAME
    if (
        receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_nlink != 1
        or receipt_path.resolve().parent != workspace
    ):
        return None, None, "inferencex_runtime_receipt_missing"
    if runtime_path.is_symlink() or not runtime_path.is_dir():
        return None, None, "inferencex_runtime_tree_missing"
    return receipt_path.resolve(), runtime_path.resolve(), None


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
