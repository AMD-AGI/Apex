"""Single-result and single-bundle finalization for kernel attempt searches."""

from __future__ import annotations

from apex.core import ApexError, IntegrityError, TaskStatus, ValidationLevel
from apex.delivery import TaskResult, build_kernel_bundle, write_task_result
from apex.evaluation import EvaluationContractReceipt
from apex.intake import TaskSpec
from apex.orchestration import RunPhase
from apex.storage import EventRecord

from ..projections import publish_terminal_projections
from .attempts import KernelAttemptOutcome, RunSession, representative_failure
from .bundle_recording import record_kernel_bundle
from .reward_recording import (
    KernelTerminalEvidence,
    record_kernel_terminal_reward,
)
from .safety_bridge import task_safety_fields
from .terminal_reward import derive_kernel_terminal_grade


def publish(session: RunSession) -> None:
    publish_terminal_projections(
        root=session.record.root,
        run_id=session.record.run_id,
        artifacts=session.record.artifacts,
    )


def deliver_best(
    session: RunSession,
    outcomes: tuple[KernelAttemptOutcome, ...],
    best: KernelAttemptOutcome,
) -> TaskResult:
    if best.candidate_root is None or best.safety_result is None:
        raise ValueError("selected kernel candidate is incomplete")
    bundle = build_kernel_bundle(
        session.resolved,
        candidate_root=best.candidate_root,
        bundle_dir=session.run_root / "bundle",
    )
    record_kernel_bundle(session, attempt_id=best.attempt_id, bundle=bundle)
    _record_selection(session, outcomes, best, bundle.digest)
    terminal = _record_terminal(session, outcomes, best)
    session.record.finish(RunPhase.SUCCEEDED, best.reason_code)
    result = TaskResult(
        schema_version=1,
        task_id=session.request.task.task_id,
        status=TaskStatus.CANDIDATE_READY,
        reason_code=best.reason_code,
        applied=False,
        external_verification_required=True,
        bundle_path=str(bundle.path),
        bundle_digest=bundle.digest,
        changed_files=bundle.changed_files,
        validation_level=ValidationLevel.NONE,
        **_lineage_fields(session, best, "keep", terminal),
        **_terminal_fields(terminal),
        **_safety_fields(best),
        **best.result_measurement_fields(),
    )
    write_task_result(result, session.request.result_json)
    return result


def _record_selection(
    session: RunSession,
    outcomes: tuple[KernelAttemptOutcome, ...],
    best: KernelAttemptOutcome,
    bundle_digest: str,
) -> None:
    for outcome in outcomes:
        if not outcome.eligible:
            continue
        selected = outcome.attempt_id == best.attempt_id
        session.record.record_decision(
            outcome.attempt_id,
            verdict="keep" if selected else "revert",
            reason=(
                best.reason_code
                if selected
                else _non_best_reason(outcome)
            ),
            bundle_digest=bundle_digest if selected else None,
            safety_result=outcome.safety_result,
            srobust=_srobust(outcome),
            reward=_reward(outcome),
        )


def finish_without_candidate(
    session: RunSession,
    outcomes: tuple[KernelAttemptOutcome, ...],
) -> TaskResult:
    selected = representative_failure(outcomes)
    terminal = (
        RunPhase.SUCCEEDED
        if selected.status is TaskStatus.NO_GAIN
        else RunPhase.FAILED
    )
    terminal_evidence = _record_terminal(session, outcomes, None)
    session.record.finish(terminal, selected.reason_code)
    result = TaskResult(
        schema_version=1,
        task_id=session.request.task.task_id,
        status=selected.status,
        reason_code=selected.reason_code,
        applied=False,
        external_verification_required=True,
        bundle_path=None,
        bundle_digest=None,
        changed_files=(),
        **_lineage_fields(
            session,
            selected,
            _terminal_verdict(selected.status),
            terminal_evidence,
        ),
        **_terminal_fields(terminal_evidence),
        **_safety_fields(selected),
        **selected.result_measurement_fields(),
    )
    write_task_result(result, session.request.result_json)
    return result


def failure_result(
    task: TaskSpec,
    error: ApexError,
    *,
    run_id: str,
    session: RunSession | None = None,
    evaluation_contract: EvaluationContractReceipt | None = None,
) -> TaskResult:
    status = (
        TaskStatus.REJECTED
        if isinstance(error, IntegrityError)
        else TaskStatus.INVALID_REQUEST
    )
    return TaskResult(
        schema_version=1,
        task_id=task.task_id,
        status=status,
        reason_code=error.reason_code,
        applied=False,
        external_verification_required=True,
        bundle_path=None,
        bundle_digest=None,
        changed_files=(),
        **_failure_lineage(run_id, error, session),
        **_evaluation_contract_fields(session, evaluation_contract),
    )


def _failure_lineage(
    run_id: str,
    error: ApexError,
    session: RunSession | None,
) -> dict[str, object]:
    fields: dict[str, object] = {
        "run_id": run_id,
        "internal_verdict": "reject",
        "error": {
            "reason_code": error.reason_code,
            "message": str(error),
            "details": dict(error.details or {}),
        },
    }
    if session is None:
        return fields
    events = session.record.iter_events()
    head = events[-1]
    verdicts = tuple(
        event for event in events if event.event_type in {"decision", "action.failed"}
    )
    receipts = _event_artifact_digests(events)
    fields.update(
        {
            "baseline_lock": _baseline_lock(session),
            **_gpu_lease_fields(session),
            "internal_verdict_ref": verdicts[-1].event_id if verdicts else None,
            "event_journal_ref": _journal_ref(session, head),
            "artifact_store_ref": {
                "path": str(session.record.artifacts.root.resolve()),
                "receipt_digests": list(receipts),
            },
        }
    )
    return fields


def _event_artifact_digests(events: tuple[EventRecord, ...]) -> tuple[str, ...]:
    receipts: dict[str, None] = {}
    for event in events:
        payload = event.payload
        for binding in payload.get("artifacts", ()):
            receipt = binding.get("receipt", {}) if isinstance(binding, dict) else {}
            digest = receipt.get("digest") if isinstance(receipt, dict) else None
            if isinstance(digest, str):
                receipts[digest] = None
    return tuple(receipts)


def _lineage_fields(
    session: RunSession,
    outcome: KernelAttemptOutcome,
    verdict: str,
    terminal: KernelTerminalEvidence,
) -> dict[str, object]:
    events = session.record.iter_events()
    head = events[-1]
    verdict_events = tuple(
        event
        for event in events
        if event.event_type in {"decision", "action.aborted", "action.failed"}
        and (
            event.payload.get("attempt_id") == outcome.attempt_id
            or event.payload.get("action_id") == outcome.attempt_id
        )
    )
    verified = outcome.eligible or bool(
        outcome.measurement and outcome.measurement.reward_eligible
    )
    return {
        "run_id": session.run_id,
        "baseline_lock": _baseline_lock(session),
        **_evaluation_contract_fields(session),
        **_gpu_lease_fields(session),
        "internal_verdict": verdict,
        "internal_verdict_ref": verdict_events[-1].event_id if verdict_events else None,
        "verification_summary_refs": (
            (outcome.evidence_receipts[-1],) if verified else ()
        ),
        "event_journal_ref": _journal_ref(session, head),
        "artifact_store_ref": {
            "path": str(session.record.artifacts.root.resolve()),
            "receipt_digests": [
                session.evaluation_contract_artifact.digest,
                session.gpu_lease_artifact.digest,
                *outcome.evidence_receipts,
                *_terminal_artifact_digests(terminal),
            ],
        },
        "error": (
            None
            if outcome.status in {TaskStatus.CANDIDATE_READY, TaskStatus.NO_GAIN}
            else {"reason_code": outcome.reason_code}
        ),
    }


def _record_terminal(
    session: RunSession,
    outcomes: tuple[KernelAttemptOutcome, ...],
    selected: KernelAttemptOutcome | None,
) -> KernelTerminalEvidence:
    grade = derive_kernel_terminal_grade(outcomes, selected)
    return record_kernel_terminal_reward(
        session.record,
        task_id=session.request.task.task_id,
        contract_digest=session.evaluation_contract.digest,
        grade=grade,
        outcomes=outcomes,
    )


def _terminal_fields(evidence: KernelTerminalEvidence) -> dict[str, object]:
    grade = evidence.grade
    return {
        "task_reward": grade.scalar_reward,
        "task_reward_vector": (
            grade.to_dict() if grade.scalar_reward is not None else None
        ),
        "reward_policy_id": grade.policy_id,
        "reward_policy_digest": grade.policy_digest,
        "reward_source_receipt": (
            evidence.source.digest if evidence.source is not None else None
        ),
        "raw_measurement_receipts": tuple(
            item.digest for item in evidence.raw_measurements
        ),
        "task_trainability": grade.trainability,
        "untrainable_reason": grade.untrainable_reason,
    }


def _terminal_artifact_digests(
    evidence: KernelTerminalEvidence,
) -> tuple[str, ...]:
    values = [evidence.result.digest, evidence.policy.digest, evidence.vector.digest]
    if evidence.source is not None:
        values.append(evidence.source.digest)
    values.extend(item.digest for item in evidence.raw_measurements)
    return tuple(values)


def _gpu_lease_fields(session: RunSession) -> dict[str, object]:
    return {
        "gpu_lease": session.gpu_lease.to_dict(),
        "gpu_lease_receipt_digest": session.gpu_lease.digest,
    }


def _evaluation_contract_fields(
    session: RunSession | None,
    contract: EvaluationContractReceipt | None = None,
) -> dict[str, object]:
    selected = session.evaluation_contract if session is not None else contract
    return {
        "evaluation_contract_status": (
            selected.status if selected is not None else "not_frozen"
        ),
        "evaluation_contract_receipt_digest": (
            selected.digest if selected is not None else None
        ),
        "evaluation_contract_unverified_reason": (
            selected.unverified_reason if selected is not None else None
        ),
        "evaluation_authority_id": (
            selected.authority.authority.authority_id
            if selected is not None and selected.authority is not None
            else None
        ),
        "evaluation_authority_kind": (
            selected.authority.authority.kind.value
            if selected is not None and selected.authority is not None
            else None
        ),
    }


def _baseline_lock(session: RunSession) -> dict[str, object]:
    return {
        "resolution_hash": session.resolved.resolution_hash,
        "file_hashes": dict(sorted(session.resolved.baseline_file_hashes.items())),
    }


def _journal_ref(session: RunSession, head: EventRecord) -> dict[str, object]:
    return {
        "path": str(session.record.journal.path.resolve()),
        "head_event_id": head.event_id,
        "head_checksum": head.checksum,
    }


def _terminal_verdict(status: TaskStatus) -> str:
    if status is TaskStatus.NO_GAIN:
        return "revert"
    if status is TaskStatus.NO_MEASUREMENT:
        return "needs_more_measurement"
    return "reject"


def _safety_fields(outcome: KernelAttemptOutcome) -> dict[str, object]:
    if outcome.safety_result is None or outcome.safety_receipt_digest is None:
        return {}
    return task_safety_fields(
        outcome.safety_result,
        outcome.safety_receipt_digest,
    )


def _srobust(outcome: KernelAttemptOutcome) -> float | None:
    return outcome.measurement.grade.srobust if outcome.measurement else None


def _reward(outcome: KernelAttemptOutcome) -> float | None:
    return outcome.measurement.grade.reward if outcome.measurement else None


def _non_best_reason(outcome: KernelAttemptOutcome) -> str:
    return (
        "candidate_not_selected_by_robust_grade"
        if outcome.measurement is not None
        else "candidate_not_selected_without_trusted_comparator"
    )


__all__ = [
    "deliver_best",
    "failure_result",
    "finish_without_candidate",
    "publish",
]
