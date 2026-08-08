"""Canonical measured-experience and knowledge-read outcome events for E2E."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from apex.core import IntegrityError
from apex.evaluation import E2ERewardGrade
from apex.knowledge import ExperienceIdentity, ExperienceOutcome
from apex.storage import ArtifactReceipt, EventRecord

from .run_record import E2ERunRecord


def record_e2e_learning(
    record: E2ERunRecord,
    *,
    attempt_id: str,
    opportunity_id: str,
    candidate_id: str | None,
    candidate_manifest: ArtifactReceipt,
    decision: ArtifactReceipt,
    verdict: str,
    reason: str,
    grade: E2ERewardGrade,
) -> None:
    """Append replayable learning facts after the atomic decision/reward commit."""

    context = _attempt_event(record, attempt_id, "context_packet_created")
    read = _attempt_event(record, attempt_id, "knowledge_read")
    _validate_context(context, read, opportunity_id)
    common = _frozen_lineage(context.payload, attempt_id, opportunity_id, candidate_id)
    if candidate_id is not None:
        outcome = _experience_outcome(grade)
        record.controller.record_domain_event(
            "experience.measured",
            {
                **common,
                "evidence_class": "measured",
                "dry_run": False,
                "identity": dict(_mapping(context.payload["experience_identity"])),
                "outcome": outcome.value,
                "strategy_fingerprint": candidate_manifest.digest,
                "mechanism": str(context.payload["experience_mechanism"]),
                "micro_verdict": "see_attempt_evaluator_events",
                "e2e_verdict": verdict,
                "evidence_receipts": [decision.digest],
                "failure_reason": None if outcome is ExperienceOutcome.SUCCESS else reason,
                "retry_condition": (
                    None
                    if outcome is ExperienceOutcome.SUCCESS
                    else "new_mechanism_or_compatible_evidence"
                ),
                "artifacts": [_binding("experience_evidence", decision)],
            },
            idempotency_key=f"attempt.{attempt_id}.experience",
        )
    for card_id in sorted(set(_card_ids(read.payload))):
        record.controller.record_domain_event(
            "knowledge_outcome_linked",
            {
                **common,
                "read_id": str(read.payload["read_id"]),
                "card_id": card_id,
                "outcome": "inconclusive",
                "evidence_receipt": decision.digest,
                "verdict": verdict,
                "reason_code": reason,
                "association_only": True,
                "card_action_binding": "unavailable",
                "evidence_class": "derived",
                "artifacts": [_binding("knowledge_outcome_evidence", decision)],
            },
            idempotency_key=f"attempt.{attempt_id}.knowledge_outcome.{card_id}",
        )


def _attempt_event(
    record: E2ERunRecord, attempt_id: str, event_type: str
) -> EventRecord:
    matches = tuple(
        event
        for event in record.iter_events()
        if event.event_type == event_type
        and event.payload.get("attempt_id") == attempt_id
    )
    if len(matches) != 1:
        raise IntegrityError(
            "E2E learning lineage is missing or ambiguous",
            "invalid_e2e_learning_lineage",
            {"attempt_id": attempt_id, "event_type": event_type, "count": len(matches)},
        )
    return matches[0]


def _validate_context(
    context: EventRecord, read: EventRecord, opportunity_id: str
) -> None:
    if (
        context.payload.get("opportunity_id") != opportunity_id
        or read.payload.get("context_packet_id")
        != context.payload.get("context_packet_id")
    ):
        raise IntegrityError(
            "E2E learning lineage conflicts with the frozen context",
            "invalid_e2e_learning_lineage",
        )
    ExperienceIdentity.from_mapping(_mapping(context.payload.get("experience_identity")))
    mechanism = context.payload.get("experience_mechanism")
    if not isinstance(mechanism, str) or not mechanism.strip():
        raise IntegrityError(
            "E2E context has no experience mechanism",
            "invalid_e2e_learning_lineage",
        )


def _frozen_lineage(
    payload: Mapping[str, Any],
    attempt_id: str,
    opportunity_id: str,
    candidate_id: str | None,
) -> dict[str, object]:
    common: dict[str, object] = {
        "attempt_id": attempt_id,
        "opportunity_id": opportunity_id,
        "anchor_generation": int(payload["anchor_generation"]),
        "state_generation": int(payload["state_generation"]),
        "split": str(payload["split"]),
        "visibility": str(payload["visibility"]),
        "context_packet_id": str(payload["context_packet_id"]),
    }
    if candidate_id is not None:
        common["candidate_id"] = candidate_id
    return common


def _experience_outcome(grade: E2ERewardGrade) -> ExperienceOutcome:
    if grade.outcome_class == "accepted":
        return ExperienceOutcome.SUCCESS
    if grade.outcome_class == "hard_gate_regression" or (
        grade.outcome_class == "no_gain"
        and grade.throughput_gain_pct is not None
        and grade.throughput_gain_pct < 0
    ):
        return ExperienceOutcome.REGRESSION
    if grade.outcome_class == "no_gain":
        return ExperienceOutcome.NO_GAIN
    return ExperienceOutcome.FAILURE


def _card_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    values = payload.get("card_ids")
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise IntegrityError(
            "Knowledge read has invalid card lineage",
            "invalid_e2e_learning_lineage",
        )
    return tuple(values)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(
            "E2E learning identity is invalid",
            "invalid_e2e_learning_lineage",
        )
    return value


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = ["record_e2e_learning"]
