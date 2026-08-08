"""Evaluator-owned E2E outcome construction and atomic commit."""

from __future__ import annotations

from typing import Mapping

from apex.core import ContractError
from apex.evaluation import E2ERewardGrade, E2EVerdict, grade_e2e_outcome
from apex.storage import ArtifactReceipt

from .learning import record_e2e_learning
from .run_record import E2ERunRecord


ArtifactBindings = tuple[tuple[str, ArtifactReceipt], ...]


def commit_e2e_reject(
    record: E2ERunRecord,
    *,
    attempt_id: str,
    opportunity_id: str,
    candidate_id: str | None,
    candidate_manifest: ArtifactReceipt,
    reason: str,
    evidence_receipts: Mapping[str, str] | None = None,
    evidence_artifacts: ArtifactBindings = (),
) -> ArtifactReceipt:
    """Close one attributable pre-acceptance outcome as an explicit REJECT."""

    evidence: dict[str, object] = {
        "schema_version": 1,
        "candidate_manifest_receipt": candidate_manifest.digest,
        **dict(evidence_receipts or {}),
    }
    return commit_e2e_outcome(
        record,
        attempt_id=attempt_id,
        opportunity_id=opportunity_id,
        candidate_id=candidate_id,
        candidate_manifest=candidate_manifest,
        verdict="reject",
        reason=reason,
        grade=grade_e2e_outcome(
            verdict="reject",
            reason_code=reason,
            candidate_present=candidate_id is not None,
        ),
        evidence=evidence,
        evidence_artifacts=evidence_artifacts,
    )


def commit_measured_e2e_outcome(
    record: E2ERunRecord,
    *,
    attempt_id: str,
    opportunity_id: str,
    candidate_id: str,
    candidate_manifest: ArtifactReceipt,
    verdict: E2EVerdict,
    evidence_receipts: Mapping[str, str],
    evidence_artifacts: ArtifactBindings,
    new_anchor_id: str | None = None,
    accepted_patch_id: str | None = None,
) -> ArtifactReceipt:
    """Grade and commit a profiler-off current-anchor KEEP or REVERT."""

    decision = "keep" if verdict.keep else "revert"
    evidence: dict[str, object] = {
        "schema_version": 1,
        "candidate_manifest_receipt": candidate_manifest.digest,
        "measurement_verdict": verdict.to_dict(),
        **dict(evidence_receipts),
    }
    return commit_e2e_outcome(
        record,
        attempt_id=attempt_id,
        opportunity_id=opportunity_id,
        candidate_id=candidate_id,
        candidate_manifest=candidate_manifest,
        verdict=decision,
        reason=verdict.reason_code,
        grade=grade_e2e_outcome(
            verdict=decision,
            reason_code=verdict.reason_code,
            candidate_present=True,
            measurement_verdict=verdict,
        ),
        evidence=evidence,
        evidence_artifacts=evidence_artifacts,
        new_anchor_id=new_anchor_id,
        accepted_patch_id=accepted_patch_id,
    )


def commit_e2e_outcome(
    record: E2ERunRecord,
    *,
    attempt_id: str,
    opportunity_id: str,
    candidate_id: str | None,
    candidate_manifest: ArtifactReceipt,
    verdict: str,
    reason: str,
    grade: E2ERewardGrade,
    evidence: Mapping[str, object],
    evidence_artifacts: ArtifactBindings = (),
    new_anchor_id: str | None = None,
    accepted_patch_id: str | None = None,
) -> ArtifactReceipt:
    """Bind immutable evidence and atomically commit decision plus reward."""

    document = _authoritative_evidence(
        evidence,
        attempt_id=attempt_id,
        opportunity_id=opportunity_id,
        candidate_id=candidate_id,
        verdict=verdict,
        reason=reason,
    )
    decision, decision_artifacts, reward = record.prepare_outcome(
        attempt_id,
        candidate_id=candidate_id,
        verdict=verdict,
        reason=reason,
        evidence=document,
        grade=grade,
        evidence_artifacts=(
            ("candidate_manifest", candidate_manifest),
            *evidence_artifacts,
        ),
    )
    record.controller.decide_e2e_candidate(
        candidate_id=candidate_id,
        receipt=decision.digest,
        verdict=verdict,
        reason=reason,
        reward_payload=reward,
        decision_artifacts=decision_artifacts,
        new_anchor_id=new_anchor_id,
        accepted_patch_id=accepted_patch_id,
    )
    record_e2e_learning(
        record,
        attempt_id=attempt_id,
        opportunity_id=opportunity_id,
        candidate_id=candidate_id,
        candidate_manifest=candidate_manifest,
        decision=decision,
        verdict=verdict,
        reason=reason,
        grade=grade,
    )
    return decision


def _authoritative_evidence(
    evidence: Mapping[str, object],
    **authoritative: object,
) -> dict[str, object]:
    document = dict(evidence)
    for field, expected in authoritative.items():
        observed = document.get(field)
        if field in document and observed != expected:
            raise ContractError(
                "E2E decision evidence has conflicting lineage",
                "e2e_decision_lineage_mismatch",
                {"field": field, "expected": expected, "observed": observed},
            )
        document[field] = expected
    return document


__all__ = [
    "ArtifactBindings",
    "commit_e2e_outcome",
    "commit_e2e_reject",
    "commit_measured_e2e_outcome",
]
