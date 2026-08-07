"""Independent validation of Magpie's run-scoped lm-eval runtime evidence."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.runtime import LmEvalRuntimeReceipt


_EVIDENCE_SCHEMA = "magpie.lm-eval-runtime-evidence/v1"
_RECEIPT_SCHEMA = "magpie.lm-eval-runtime-receipt/v1"
_MANIFEST_SCHEMA = "apex.lm-eval-runtime/v1"
_MANIFEST_NAME = "lm_eval_runtime_manifest.json"
_RECEIPT_NAME = "lm_eval_runtime_receipt.json"
_EVIDENCE_FIELDS = {
    "schema", "requested", "status", "verified", "evidence_present",
    "runtime_sha256", "identity", "mount_mode", "manifest_artifact",
    "receipt_artifact", "errors",
}
_RECEIPT_FIELDS = {
    "schema", "runtime_sha256", "identity", "manifest_sha256", "site_packages",
    "python_abi", "lm_eval_version", "lm_eval_module", "execution_mode",
    "read_only_mount", "verified",
}
_MANIFEST_FIELDS = {"schema", "runtime_sha256", "site_packages", "identity", "files"}
_ARTIFACT_FIELDS = {"path", "size_bytes", "sha256"}


@dataclass(frozen=True, slots=True)
class LmEvalRuntimeEvidence:
    """Normalized proof that Magpie consumed the caller-verified runtime."""

    required: bool
    passed: bool
    runtime_sha256: str | None
    identity: Mapping[str, str] | None
    manifest_path: Path | None
    receipt_path: Path | None
    execution_mode: str | None
    read_only_mount: bool | None
    error: str | None = None


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _artifact(
    workspace: Path,
    value: Any,
    expected_name: str,
) -> tuple[Path, bytes]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_FIELDS:
        raise ValueError(f"{expected_name} artifact declaration is invalid")
    if value.get("path") != expected_name:
        raise ValueError(f"{expected_name} artifact path is invalid")
    path = workspace / expected_name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.is_symlink():
        raise ValueError(f"{expected_name} must be a non-hardlinked regular file")
    content = path.read_bytes()
    if value.get("size_bytes") != len(content) or value.get("sha256") != _sha256(content):
        raise ValueError(f"{expected_name} artifact digest is invalid")
    return path.resolve(), content


def _json_object(content: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    expected: LmEvalRuntimeReceipt,
    manifest_sha256: str,
    execution_mode: str,
) -> tuple[bool, str]:
    if set(value) != _RECEIPT_FIELDS or value.get("schema") != _RECEIPT_SCHEMA:
        raise ValueError("lm-eval runtime receipt contract is invalid")
    read_only = value.get("read_only_mount")
    module = value.get("lm_eval_module")
    valid = (
        value.get("runtime_sha256") == expected.runtime_sha256
        and value.get("identity") == expected.identity
        and value.get("manifest_sha256") == manifest_sha256
        and value.get("site_packages") == "site-packages"
        and value.get("python_abi") == expected.identity.get("python_abi")
        and value.get("lm_eval_version") == expected.identity.get("lm_eval_version")
        and value.get("execution_mode") == execution_mode
        and isinstance(read_only, bool)
        and (execution_mode != "docker" or read_only is True)
        and isinstance(module, str)
        and module.startswith("site-packages/lm_eval/")
        and value.get("verified") is True
    )
    if not valid:
        raise ValueError("lm-eval runtime receipt differs from the expected runtime")
    return read_only, module


def parse_lm_eval_runtime_evidence(
    report: Mapping[str, Any],
    report_path: Path,
    *,
    expected: LmEvalRuntimeReceipt | None,
    execution_mode: str | None,
) -> LmEvalRuntimeEvidence:
    """Rehash Magpie's receipt and manifest snapshot against Apex's receipt."""

    value = report.get("lm_eval_runtime_receipt")
    if expected is None and execution_mode is None:
        if value is None:
            return LmEvalRuntimeEvidence(False, True, None, None, None, None, None, None)
        return _failed(None, "unexpected_lm_eval_runtime_evidence")
    if expected is None or execution_mode not in {"docker", "local"}:
        return _failed(expected, "incomplete_lm_eval_runtime_expectation")
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_FIELDS:
        return _failed(expected, "lm_eval_runtime_evidence_missing")
    workspace = report_path.parent.resolve()
    try:
        manifest_path, manifest_bytes = _artifact(
            workspace, value.get("manifest_artifact"), _MANIFEST_NAME
        )
        receipt_path, receipt_bytes = _artifact(
            workspace, value.get("receipt_artifact"), _RECEIPT_NAME
        )
        manifest_sha256 = _sha256(manifest_bytes)
        manifest = _json_object(manifest_bytes, "lm-eval runtime manifest")
        receipt = _json_object(receipt_bytes, "lm-eval runtime receipt")
        if (
            set(manifest) != _MANIFEST_FIELDS
            or manifest.get("schema") != _MANIFEST_SCHEMA
            or manifest.get("runtime_sha256") != expected.runtime_sha256
            or manifest.get("identity") != expected.identity
            or manifest.get("site_packages") != "site-packages"
            or manifest_sha256 != expected.manifest_sha256
        ):
            raise ValueError("lm-eval manifest snapshot differs from Apex's receipt")
        read_only, _ = _validate_receipt(
            receipt,
            expected=expected,
            manifest_sha256=manifest_sha256,
            execution_mode=execution_mode,
        )
        valid_evidence = (
            value.get("schema") == _EVIDENCE_SCHEMA
            and value.get("requested") is True
            and value.get("status") == "verified"
            and value.get("verified") is True
            and value.get("evidence_present") is True
            and value.get("runtime_sha256") == expected.runtime_sha256
            and value.get("identity") == expected.identity
            and value.get("mount_mode") == ("read_only" if execution_mode == "docker" else "local")
            and value.get("errors") == []
        )
        if not valid_evidence:
            raise ValueError("lm-eval runtime report evidence is not verified")
    except (OSError, ValueError) as error:
        return _failed(expected, f"invalid_lm_eval_runtime_evidence:{error}")
    return LmEvalRuntimeEvidence(
        True, True, expected.runtime_sha256, dict(expected.identity),
        manifest_path, receipt_path, execution_mode, read_only,
    )


def _failed(
    expected: LmEvalRuntimeReceipt | None,
    error: str,
) -> LmEvalRuntimeEvidence:
    return LmEvalRuntimeEvidence(
        True,
        False,
        expected.runtime_sha256 if expected else None,
        dict(expected.identity) if expected else None,
        None,
        None,
        None,
        None,
        error,
    )


__all__ = ["LmEvalRuntimeEvidence", "parse_lm_eval_runtime_evidence"]
