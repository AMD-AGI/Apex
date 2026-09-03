"""Verified immutable delivery for one formally graded kernel attempt."""

from __future__ import annotations

from apex.core import ContractError, TaskStatus
from apex.delivery import (
    build_kernel_bundle,
    capture_portable_bundle,
    kernel_reproduction_declaration,
)
from apex.evaluation import selected_kernel_terminal_grade
from apex.orchestration import RunPhase

from .attempts import KernelAttemptOutcome
from .formal_campaign import FormalKernelCampaign
from .formal_evidence import (
    attempt_event,
    event_receipt,
    kind_event,
    load_grade,
)
from .formal_result import FormalEvaluatorResult
from .reward_recording import record_kernel_terminal_reward


def build_formal_bundle(
    campaign: FormalKernelCampaign,
    *,
    attempt_id: str,
    contract_digest: str,
    candidate_digest: str,
    finish: bool = True,
) -> FormalEvaluatorResult:
    contract = campaign.authorized_contract
    if contract is None or contract.digest != contract_digest:
        raise ContractError(
            "Verified evaluation contract is missing or differs",
            "evaluation_authority_mismatch",
        )
    evaluation, _ = load_grade(campaign, attempt_id)
    if (
        not evaluation.reward_eligible
        or not evaluation.improved
        or evaluation.execution.candidate_source_sha256 != candidate_digest
    ):
        raise ContractError(
            "Only a verified improving candidate can be bundled",
            "candidate_not_promotion_eligible",
        )
    measurement = attempt_event(campaign, attempt_id, "measurement_result")
    assert measurement is not None
    existing = kind_event(
        campaign, attempt_id, "kernel_winner_bundle", required=False
    )
    if existing is None:
        _build_and_record_delivery(
            campaign, attempt_id, contract, candidate_digest
        )
        existing = kind_event(campaign, attempt_id, "kernel_winner_bundle")
        assert existing is not None
    projected = _bundle_projection(existing)
    bundle_digest = str(projected.receipt["bundle_digest"])
    campaign.record.record_decision(
        attempt_id,
        verdict="keep",
        reason="candidate_verified_by_trusted_measurement",
        bundle_digest=bundle_digest,
        srobust=evaluation.grade.srobust,
        reward=evaluation.grade.reward,
    )
    _record_terminal(
        campaign,
        attempt_id,
        contract.digest,
        candidate_digest,
        evaluation,
        event_receipt(measurement, "kernel_grade").digest,
    )
    if finish:
        campaign.record.finish(RunPhase.SUCCEEDED, "verified_candidate_delivered")
    return projected


def _build_and_record_delivery(
    campaign, attempt_id, contract, candidate_digest
) -> None:
    files = campaign.candidate_files(attempt_id)
    with campaign.project(files) as projection:
        if projection.source_digest != candidate_digest:
            raise ContractError(
                "Candidate digest differs from the frozen attempt",
                "candidate_digest_mismatch",
            )
        bundle = build_kernel_bundle(
            projection.resolved,
            candidate_root=projection.root,
            bundle_dir=campaign.record.root / "bundles" / attempt_id,
        )
    portable = capture_portable_bundle(
        campaign.record.artifacts,
        bundle.path,
        bundle_kind="kernel",
        expected_digest=bundle.digest,
    )
    _record_delivery(campaign, attempt_id, contract, bundle, portable)


def _record_delivery(campaign, attempt_id, contract, bundle, portable) -> None:
    campaign.record.controller.record_domain_event(
        "delivery_result",
        {
            **campaign.record.attempt_payload(attempt_id),
            "kind": "kernel_winner_bundle",
            "verified": True,
            "bundle_kind": "kernel",
            "bundle_digest": bundle.digest,
            "bundle_path": str(bundle.path),
            "replication": kernel_reproduction_declaration(
                contract, bundle, portable
            ),
            "artifacts": list(portable.artifact_bindings()),
        },
        idempotency_key=f"attempt.{attempt_id}.delivery_bundle",
    )


def _record_terminal(
    campaign,
    attempt_id,
    contract_digest,
    candidate_digest,
    evaluation,
    grade_receipt_digest,
) -> None:
    outcome = KernelAttemptOutcome(
        attempt_id=attempt_id,
        cycle=0,
        status=TaskStatus.CANDIDATE_READY,
        reason_code="candidate_verified_by_trusted_measurement",
        strategy_fingerprint=candidate_digest,
        evidence_receipts=(grade_receipt_digest,),
        measurement=evaluation,
    )
    record_kernel_terminal_reward(
        campaign.record,
        task_id=campaign.task.task_id,
        contract_digest=contract_digest,
        grade=selected_kernel_terminal_grade(attempt_id, evaluation.grade),
        outcomes=(outcome,),
    )


def _bundle_projection(event) -> FormalEvaluatorResult:
    evidence = event_receipt(event, "winner_bundle")
    verification = event_receipt(event, "bundle_verification")
    return FormalEvaluatorResult(
        {
            "schema": "apex.kernel-bundle-capability/v1",
            "verified": True,
            "attempt_id": event.payload["attempt_id"],
            "bundle_digest": event.payload["bundle_digest"],
            "bundle_path": event.payload["bundle_path"],
            "bundle_evidence_receipt": evidence.to_dict(),
            "bundle_verification_receipt": verification.to_dict(),
        },
        (evidence, verification),
    )


__all__ = ["build_formal_bundle"]
