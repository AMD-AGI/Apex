"""Evaluator-owned terminalization of a stopped standalone formal campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from apex.core import IntegrityError, TaskStatus
from apex.orchestration import RunPhase
from apex.storage import ArtifactReceipt

from .attempts import KernelAttemptOutcome
from .formal_campaign import FormalKernelCampaign
from .formal_capability_recording import (
    begin_formal_capability,
    complete_formal_capability,
    fail_formal_capability,
)
from .formal_evidence import attempt_event, event_receipt, load_grade
from .reward_recording import KernelTerminalEvidence, record_kernel_terminal_reward
from .terminal_reward import derive_kernel_terminal_grade


@dataclass(frozen=True, slots=True)
class FormalStopResult:
    run_id: str
    terminal_status: str
    task_reward: float | None
    trainability: str
    untrainable_reason: str | None
    result_receipt: ArtifactReceipt

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.kernel-campaign-stop/v1",
            "run_id": self.run_id,
            "terminal_status": self.terminal_status,
            "task_reward": self.task_reward,
            "trainability": self.trainability,
            "untrainable_reason": self.untrainable_reason,
            "terminal_result_receipt": self.result_receipt.to_dict(),
        }


def stop_formal_campaign(
    campaign: FormalKernelCampaign,
    *,
    reason: str = "user_requested",
    capability_arguments: Mapping[str, object] | None = None,
) -> FormalStopResult:
    """Close without selecting a candidate; measured baseline no-op may still score."""

    invocation = (
        begin_formal_capability(
            campaign.record, "campaign.stop", capability_arguments
        )
        if capability_arguments is not None
        else None
    )
    try:
        result = _stop(campaign, reason)
    except Exception as error:
        if invocation is not None:
            fail_formal_capability(campaign.record, invocation, error)
        raise
    if invocation is not None:
        complete_formal_capability(
            campaign.record,
            invocation,
            {"campaign": {"stop": result.to_dict()}},
        )
    if campaign.record.controller.state.phase is RunPhase.RUNNING:
        campaign.record.finish(RunPhase.CANCELLED, reason)
    return result


def _stop(campaign: FormalKernelCampaign, reason: str) -> FormalStopResult:
    existing = _terminal_result(campaign)
    if existing is not None:
        return _project_existing(campaign, existing)
    if campaign.record.controller.state.phase is not RunPhase.RUNNING:
        raise IntegrityError(
            "Terminal campaign lacks its canonical kernel result",
            "kernel_terminal_result_missing",
        )
    _close_pending(campaign, reason)
    _revert_undecided(campaign, reason)
    outcomes = _outcomes(campaign)
    grade = derive_kernel_terminal_grade(outcomes, None)
    contract = campaign.authorized_contract or campaign.draft_contract
    terminal = record_kernel_terminal_reward(
        campaign.record,
        task_id=campaign.task.task_id,
        contract_digest=contract.digest,
        grade=grade,
        outcomes=outcomes,
    )
    return _result(campaign, terminal)


def _close_pending(campaign: FormalKernelCampaign, reason: str) -> None:
    pending = campaign.record.controller.state.pending_action
    if pending is None:
        return
    campaign.record.defer_attempt(
        pending.action_id, f"campaign_stopped:{reason}"
    )


def _revert_undecided(campaign: FormalKernelCampaign, reason: str) -> None:
    events = campaign.record.iter_events()
    candidates = {
        str(event.payload["attempt_id"])
        for event in events
        if event.event_type == "candidate_frozen"
        and isinstance(event.payload.get("attempt_id"), str)
    }
    decided = {
        str(event.payload["attempt_id"])
        for event in events
        if event.event_type == "decision"
        and isinstance(event.payload.get("attempt_id"), str)
    }
    for attempt_id in sorted(candidates - decided):
        evaluation = _measurement(campaign, attempt_id)
        campaign.record.record_decision(
            attempt_id,
            verdict="revert",
            reason=f"campaign_stopped_without_selection:{reason}",
            srobust=evaluation.grade.srobust if evaluation is not None else None,
            reward=evaluation.grade.reward if evaluation is not None else None,
        )


def _outcomes(
    campaign: FormalKernelCampaign,
) -> tuple[KernelAttemptOutcome, ...]:
    candidates = [
        event
        for event in campaign.record.iter_events()
        if event.event_type == "candidate_frozen"
    ]
    return tuple(
        _outcome(campaign, event, cycle)
        for cycle, event in enumerate(candidates)
    )


def _outcome(campaign, candidate, cycle: int) -> KernelAttemptOutcome:
    attempt_id = str(candidate.payload["attempt_id"])
    source_digest = candidate.payload.get("candidate_source_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise IntegrityError(
            "Stopped attempt lacks its frozen source digest",
            "kernel_terminal_evidence_missing",
        )
    decision = attempt_event(campaign, attempt_id, "decision", required=False)
    reason = (
        str(decision.payload.get("reason"))
        if decision is not None
        else "campaign_stopped_without_selection"
    )
    evaluation = _measurement(campaign, attempt_id)
    receipts = _attempt_receipts(campaign, attempt_id)
    if not receipts:
        raise IntegrityError(
            "Stopped attempt has no canonical evidence",
            "kernel_terminal_evidence_missing",
        )
    return KernelAttemptOutcome(
        attempt_id=attempt_id,
        cycle=cycle,
        status=_status(reason, evaluation is not None),
        reason_code=reason,
        strategy_fingerprint=source_digest,
        evidence_receipts=receipts,
        changed_files=tuple(candidate.payload.get("changed_files", ())),
        measurement=evaluation,
    )


def _measurement(campaign, attempt_id: str):
    measured = attempt_event(
        campaign, attempt_id, "measurement_result", required=False
    )
    rewarded = [
        event
        for event in campaign.record.iter_events()
        if event.event_type == "reward_committed"
        and event.payload.get("scope") == "attempt"
        and event.payload.get("attempt_id") == attempt_id
    ]
    if measured is None or measured.payload.get("measurement_status") != "valid":
        return None
    if len(rewarded) != 1:
        raise IntegrityError(
            "Valid measurement has missing or duplicate attempt reward",
            "attempt_reward_invalid",
        )
    evaluation, _ = load_grade(campaign, attempt_id)
    return evaluation


def _attempt_receipts(campaign, attempt_id: str) -> tuple[str, ...]:
    receipts: dict[str, None] = {}
    for event in campaign.record.iter_events():
        if event.payload.get("attempt_id") != attempt_id:
            continue
        for binding in event.payload.get("artifacts", ()):
            if not isinstance(binding, dict) or not isinstance(
                binding.get("receipt"), dict
            ):
                continue
            receipt = ArtifactReceipt.from_dict(binding["receipt"])
            campaign.record.artifacts.verify(receipt)
            receipts[receipt.digest] = None
    return tuple(receipts)


def _status(reason: str, measured: bool) -> TaskStatus:
    if reason == "compile_failed" or reason == "correctness_failed":
        return TaskStatus.REJECTED
    if measured:
        return TaskStatus.NO_GAIN
    return TaskStatus.NO_MEASUREMENT


def _terminal_result(campaign):
    matches = [
        event
        for event in campaign.record.iter_events()
        if event.event_type == "delivery_result"
        and event.payload.get("kind") == "kernel_terminal_result"
    ]
    if len(matches) > 1:
        raise IntegrityError(
            "Kernel terminal result is ambiguous",
            "kernel_terminal_result_ambiguous",
        )
    return matches[0] if matches else None


def _project_existing(campaign, event) -> FormalStopResult:
    receipt = event_receipt(event, "kernel_terminal_result")
    value = json.loads(campaign.record.artifacts.read_bytes(receipt))
    return FormalStopResult(
        campaign.record.run_id,
        campaign.record.controller.state.phase.value,
        value.get("task_reward"),
        str(value["trainability"]),
        value.get("untrainable_reason"),
        receipt,
    )


def _result(
    campaign: FormalKernelCampaign, terminal: KernelTerminalEvidence
) -> FormalStopResult:
    grade = terminal.grade
    return FormalStopResult(
        campaign.record.run_id,
        "cancelled",
        grade.scalar_reward,
        grade.trainability,
        grade.untrainable_reason,
        terminal.result,
    )


__all__ = ["FormalStopResult", "stop_formal_campaign"]
