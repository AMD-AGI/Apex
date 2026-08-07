"""Compose independently parsed benchmark attestations without result-policy cycles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from apex.runtime import LmEvalRuntimeReceipt

from .inferencex_runtime import (
    InferenceXRuntimeEvidence,
    parse_inferencex_runtime_evidence,
)
from .lm_eval_runtime import LmEvalRuntimeEvidence, parse_lm_eval_runtime_evidence
from .model_revision import ModelRevisionEvidence, parse_model_revision_evidence


Attestations = tuple[
    ModelRevisionEvidence, InferenceXRuntimeEvidence, LmEvalRuntimeEvidence
]


def parse_attestations(
    report: Mapping[str, Any],
    report_path: Path,
    expected_model: str | None,
    expected_model_revision: str | None,
    expected_inferencex_root: Path | None,
    expected_inferencex_commit: str | None,
    expected_inferencex_tree: str | None,
    expected_lm_eval_runtime: LmEvalRuntimeReceipt | None,
    expected_lm_eval_execution_mode: str | None,
) -> Attestations:
    """Parse all protected side-artifact lanes against caller expectations."""

    resolved = report_path.resolve()
    model = parse_model_revision_evidence(
        report,
        resolved,
        expected_model=expected_model,
        expected_revision=expected_model_revision,
    )
    inferencex = parse_inferencex_runtime_evidence(
        report,
        resolved,
        expected_source_root=expected_inferencex_root,
        expected_commit=expected_inferencex_commit,
        expected_tree=expected_inferencex_tree,
    )
    lm_eval = parse_lm_eval_runtime_evidence(
        report,
        resolved,
        expected=expected_lm_eval_runtime,
        execution_mode=expected_lm_eval_execution_mode,
    )
    return model, inferencex, lm_eval


def result_verdict(
    report: Mapping[str, Any],
    *,
    quality_passed: bool,
    quality_required: bool,
    command_exit_code: int | None,
    timed_out: bool,
    lane_errors: tuple[str, ...],
    base_errors: tuple[str, ...],
    attestations: Attestations,
) -> tuple[bool, tuple[str, ...]]:
    """Require every requested independent attestation for formal success."""

    errors = base_errors
    for evidence in attestations:
        if evidence.error:
            errors += (evidence.error,)
    success = (
        report.get("success") is True
        and command_exit_code == 0
        and not timed_out
        and (quality_passed or not quality_required)
        and not lane_errors
        and all(evidence.passed for evidence in attestations)
    )
    return success, errors


def evidence_artifacts(attestations: Attestations) -> tuple[Path, ...]:
    """Return every independently rehashed side artifact for persistence."""

    model, inferencex, lm_eval = attestations
    paths: tuple[Path, ...] = ()
    if model.source_path:
        paths += (model.source_path,)
    if inferencex.receipt_path:
        paths += (inferencex.receipt_path,)
    return paths + tuple(
        path for path in (lm_eval.manifest_path, lm_eval.receipt_path) if path
    )


__all__ = ["Attestations", "evidence_artifacts", "parse_attestations", "result_verdict"]
