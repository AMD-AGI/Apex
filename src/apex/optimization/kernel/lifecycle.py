"""Attempt closure, safety isolation, and measured-history helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.core import TaskStatus
from apex.evaluation.safety import (
    PhaseIsolationReceipt,
    SafetyGateRequest,
    SafetyGateResult,
    VerificationPolicy,
)
from apex.knowledge import ExperienceOutcome
from apex.storage import ArtifactReceipt

from .attempts import AttemptSession, KernelAttemptOutcome, PreparedCandidate
from .context import kernel_experience_identity
from .experience_recording import record_deferred_experience
from .measurement import KernelMeasurementEvaluation


def close_prepared(
    attempt: AttemptSession,
    prepared: PreparedCandidate,
    status: TaskStatus,
    reason: str,
    evidence: tuple[ArtifactReceipt, ...],
    **kwargs: Any,
) -> KernelAttemptOutcome:
    return close_attempt(
        attempt,
        status,
        reason,
        strategy=prepared.normal_source_digest,
        evidence=evidence,
        candidate_root=attempt.candidate.root,
        changed_files=prepared.changed_files,
        **kwargs,
    )


def close_attempt(
    attempt: AttemptSession,
    status: TaskStatus,
    reason: str,
    *,
    strategy: str,
    evidence: tuple[ArtifactReceipt, ...],
    closure: str,
    candidate_root: Path | None = None,
    changed_files: tuple[str, ...] = (),
    safety: SafetyGateResult | None = None,
    safety_receipt: ArtifactReceipt | None = None,
    measurement: KernelMeasurementEvaluation | None = None,
    measurement_fields: Mapping[str, Any] | None = None,
    eligible: bool = False,
    stop_search: bool = False,
) -> KernelAttemptOutcome:
    receipts = unique_receipts(evidence)
    identity = kernel_experience_identity(attempt.run.resolved)
    if status is TaskStatus.CANDIDATE_READY and measurement is None:
        record_deferred_experience(
            attempt.run.record,
            attempt.attempt_id,
            identity=identity,
            strategy_fingerprint=strategy,
            reason=reason,
            evidence=receipts,
        )
    else:
        attempt.run.record.record_experience(
            attempt.attempt_id,
            identity=identity,
            outcome=experience_outcome(status, measurement),
            strategy_fingerprint=strategy,
            mechanism=f"Fresh isolated candidate attempt ended with evaluator outcome {reason}.",
            micro_verdict=reason,
            evidence=receipts,
            failure_reason=None if eligible else reason,
            retry_condition=None if eligible else retry_condition(reason),
        )
    decided = close_action(
        attempt,
        closure=closure,
        reason=reason,
        safety=safety,
        measurement=measurement,
    )
    return KernelAttemptOutcome(
        attempt_id=attempt.attempt_id,
        cycle=attempt.cycle,
        status=status,
        reason_code=reason,
        strategy_fingerprint=strategy,
        evidence_receipts=tuple(item.digest for item in receipts),
        candidate_root=candidate_root,
        changed_files=changed_files,
        safety_result=safety,
        safety_receipt_digest=(safety_receipt.digest if safety_receipt else None),
        measurement=measurement,
        measurement_fields=measurement_fields,
        eligible=eligible,
        stop_search=stop_search,
        decision_recorded=decided,
    )


def close_action(
    attempt: AttemptSession,
    *,
    closure: str,
    reason: str,
    safety: SafetyGateResult | None,
    measurement: KernelMeasurementEvaluation | None,
) -> bool:
    record = attempt.run.record
    if closure == "complete":
        record.complete_verified(attempt.attempt_id)
        return False
    if closure == "complete_revert":
        record.complete_verified(attempt.attempt_id)
        srobust, reward = grade_values(measurement)
        record.record_decision(
            attempt.attempt_id,
            verdict="revert",
            reason=reason,
            safety_result=safety,
            srobust=srobust,
            reward=reward,
        )
        return True
    if closure == "abort":
        record.abort_no_gain(attempt.attempt_id, reason)
        return True
    if closure == "defer":
        record.defer_attempt(attempt.attempt_id, reason)
        return True
    if closure == "reject":
        record.reject_attempt(attempt.attempt_id, reason)
        return True
    raise ValueError(f"unsupported attempt closure: {closure}")


def phase_isolation_receipt(
    attempt: AttemptSession,
    prepared: PreparedCandidate,
) -> PhaseIsolationReceipt:
    plan = prepared.safety_plan
    return PhaseIsolationReceipt(
        run_id=attempt.run.run_id,
        plan_fingerprint=plan.fingerprint,
        anchor_generation=plan.anchor_generation,
        candidate_digest=plan.candidate_digest,
        frozen_root=str(prepared.safety_candidate.root),
        evaluator_artifact_root=str(
            attempt.run.run_root / "safety" / attempt.attempt_id
        ),
        agent_process_tree_terminated=True,
        credentials_revoked=True,
        tool_channels_revoked=True,
        report_directory_hidden_from_agent=True,
        candidate_read_only=True,
    )


def safety_gate_request(
    attempt: AttemptSession,
    prepared: PreparedCandidate,
    isolation: PhaseIsolationReceipt,
    *,
    policy: VerificationPolicy,
) -> SafetyGateRequest:
    plan = prepared.safety_plan
    return SafetyGateRequest(
        plan=plan,
        policy=policy,
        frozen_candidate=prepared.safety_candidate,
        isolation_receipt=isolation,
        artifact_root=attempt.run.run_root / "safety" / attempt.attempt_id,
        current_run_id=attempt.run.run_id,
        current_candidate_id=attempt.attempt_id,
        current_anchor_generation=plan.anchor_generation,
        current_deployed_digest=prepared.normal_source_digest,
    )


def unique_receipts(
    receipts: Sequence[ArtifactReceipt],
) -> tuple[ArtifactReceipt, ...]:
    return tuple({item.digest: item for item in receipts}.values())


def experience_outcome(
    status: TaskStatus,
    measurement: KernelMeasurementEvaluation | None,
) -> ExperienceOutcome:
    if status is TaskStatus.CANDIDATE_READY:
        return ExperienceOutcome.SUCCESS
    if status is TaskStatus.NO_GAIN:
        if measurement is not None and (measurement.grade.srobust or 0.0) < 1.0:
            return ExperienceOutcome.REGRESSION
        return ExperienceOutcome.NO_GAIN
    if status in {TaskStatus.TIMEOUT, TaskStatus.INFRASTRUCTURE_ERROR}:
        return ExperienceOutcome.INFRA_ERROR
    return ExperienceOutcome.FAILURE


def retry_condition(reason: str) -> str:
    if reason == "compile_failed":
        return "Retry with distinct source that passes the frozen compile argv."
    if reason == "correctness_failed":
        return "Retry with distinct source that preserves the frozen correctness oracle."
    if "measurement" in reason or reason == "performance_command_failed":
        return "Retry only with a fresh evaluator report satisfying the raw-sample policy."
    if reason == "srobust_threshold_not_met":
        return "Retry with a distinct strategy that improves both p50 and p99."
    if reason == "timing_noise_exceeds_policy":
        return "Retry with a less noisy implementation under the same frozen timing policy."
    if reason in {"timing_confidence_below_floor", "timing_confidence_unavailable"}:
        return "Retry with a fresh evaluator report whose paired blocks establish confidence."
    if reason == "worst_case_regression":
        return "Retry with a strategy that does not regress any frozen workload case."
    return "Retry only in a fresh workspace and backend invocation under the same contract."


def grade_values(
    measurement: KernelMeasurementEvaluation | None,
) -> tuple[float | None, float | None]:
    if measurement is None:
        return None, None
    return measurement.grade.srobust, measurement.grade.reward


__all__ = [
    "close_attempt",
    "close_prepared",
    "phase_isolation_receipt",
    "safety_gate_request",
]
