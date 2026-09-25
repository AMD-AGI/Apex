"""Independent validation of Magpie model-revision evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import sha256_file


_COMMIT = re.compile(r"[0-9a-f]{40}")
_EVIDENCE_SCHEMA = "magpie.model-revision-evidence/v1"
_RECEIPT_SCHEMA = "magpie.model-revision-receipt/v1"
_RECEIPT_NAME = "model_revision_receipt.json"


@dataclass(frozen=True, slots=True)
class ModelRevisionEvidence:
    """Normalized requested/resolved model identity and artifact binding."""

    required: bool
    passed: bool
    requested_revision: str | None
    resolved_revision: str | None
    source_path: Path | None
    error: str | None = None


def parse_model_revision_evidence(
    report: Mapping[str, Any],
    report_path: Path,
    *,
    expected_model: str | None,
    expected_revision: str | None,
) -> ModelRevisionEvidence:
    """Validate report fields and independently hash the workspace receipt."""

    if expected_revision is None:
        return _not_requested(report)
    if not _COMMIT.fullmatch(expected_revision):
        return _failed(expected_revision, "invalid_expected_model_revision")
    value = report.get("model_revision_receipt")
    if not isinstance(value, Mapping):
        return _failed(expected_revision, "model_revision_evidence_missing")
    field_error = _evidence_field_error(
        value,
        expected_model=expected_model,
        expected_revision=expected_revision,
    )
    if field_error:
        return _failed(expected_revision, field_error)
    artifact = value.get("receipt_artifact")
    assert isinstance(artifact, Mapping)
    source, artifact_error = _artifact_path(report_path.parent, artifact)
    if artifact_error:
        return _failed(expected_revision, artifact_error)
    assert source is not None
    receipt_error = _receipt_error(
        source,
        artifact,
        expected_model=expected_model,
        expected_revision=expected_revision,
    )
    if receipt_error:
        return _failed(expected_revision, receipt_error)
    return ModelRevisionEvidence(
        True,
        True,
        expected_revision,
        expected_revision,
        source,
    )


def _not_requested(report: Mapping[str, Any]) -> ModelRevisionEvidence:
    value = report.get("model_revision_receipt")
    if value is not None and (
        not isinstance(value, Mapping)
        or value.get("requested") is not False
        or value.get("status") != "not_requested"
        or value.get("verified") is not False
        or value.get("evidence_present") is not False
    ):
        return ModelRevisionEvidence(
            False, False, None, None, None, "unexpected_model_revision_evidence"
        )
    return ModelRevisionEvidence(False, True, None, None, None)


def _evidence_field_error(
    value: Mapping[str, Any],
    *,
    expected_model: str | None,
    expected_revision: str,
) -> str | None:
    expected = (
        value.get("schema") == _EVIDENCE_SCHEMA
        and value.get("requested") is True
        and value.get("status") == "verified"
        and value.get("verified") is True
        and value.get("evidence_present") is True
        and value.get("requested_revision") == expected_revision
        and value.get("resolved_revision") == expected_revision
        and (expected_model is None or value.get("model") == expected_model)
        and isinstance(value.get("snapshot_path"), str)
        and Path(str(value.get("snapshot_path"))).is_absolute()
        and value.get("errors") == []
        and isinstance(value.get("receipt_artifact"), Mapping)
    )
    return None if expected else "model_revision_evidence_mismatch"


def _artifact_path(
    workspace: Path, artifact: Mapping[str, Any]
) -> tuple[Path | None, str | None]:
    if artifact.get("path") != _RECEIPT_NAME:
        return None, "model_revision_artifact_path_mismatch"
    candidate = workspace.resolve() / _RECEIPT_NAME
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.stat().st_nlink != 1
    ):
        return None, "model_revision_artifact_missing"
    source = candidate.resolve()
    if source.parent != workspace.resolve():
        return None, "model_revision_artifact_missing"
    size = artifact.get("size_bytes")
    digest = artifact.get("sha256")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or source.stat().st_size != size
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or sha256_file(source) != digest
    ):
        return None, "model_revision_artifact_digest_mismatch"
    return source, None


def _receipt_error(
    source: Path,
    artifact: Mapping[str, Any],
    *,
    expected_model: str | None,
    expected_revision: str,
) -> str | None:
    del artifact
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid_model_revision_receipt"
    valid = (
        isinstance(value, Mapping)
        and value.get("schema") == _RECEIPT_SCHEMA
        and value.get("requested_revision") == expected_revision
        and value.get("resolved_revision") == expected_revision
        and value.get("verified") is True
        and (expected_model is None or value.get("model") == expected_model)
        and isinstance(value.get("snapshot_path"), str)
        and Path(str(value.get("snapshot_path"))).is_absolute()
    )
    return None if valid else "invalid_model_revision_receipt"


def _failed(revision: str, error: str) -> ModelRevisionEvidence:
    return ModelRevisionEvidence(True, False, revision, None, None, error)


__all__ = ["ModelRevisionEvidence", "parse_model_revision_evidence"]
