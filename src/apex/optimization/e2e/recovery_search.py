"""Canonical recovery projection for an interrupted E2E kernel search."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.benchmark import BenchmarkConfigViews
from apex.core import IntegrityError
from apex.evaluation import E2EAcceptancePolicy, E2EMeasurement
from apex.intake import E2EOptimizeSpec
from apex.orchestration import SearchStage
from apex.storage import ArtifactReceipt, EventJournal, EventRecord

from .benchmarking import Diagnosis
from .candidate import E2ECandidate
from .kernel_lane import KernelOpportunity
from .recovery import receipt_for_digest, recover_diagnosis_history
from .recovery_artifacts import (
    read_json_object,
    recover_candidate,
    recover_deployment,
    recover_micro,
    recover_safety,
)
from .recovery_bindings import (
    unique_role as _unique_role,
    verify_deployment_config_bindings as _verify_deployment_config_bindings,
)
from .promotion import MatchedPromotion
from .promotion_recovery import recover_matched_promotion, validate_promotion_context, verify_promotion_reward_binding
from .run_record import E2ERunRecord
from .search_support import QualifiedAttempt, validate_deployment
from .services import AcceptedCandidate


@dataclass(frozen=True, slots=True)
class RecoveredAttempt:
    """Verified pieces of the currently active attempt, if one exists."""

    attempt_id: str
    opportunity: KernelOpportunity
    candidate: E2ECandidate | None
    candidate_receipt: ArtifactReceipt | None
    micro_pair: tuple[Any, ArtifactReceipt] | None
    safety_pair: tuple[Any, ArtifactReceipt] | None
    deployment_pair: tuple[Any, ArtifactReceipt] | None
    promotion: MatchedPromotion | None

    def qualified(self) -> QualifiedAttempt:
        if (
            self.candidate is None
            or self.candidate_receipt is None
            or self.micro_pair is None
            or self.safety_pair is None
            or self.deployment_pair is None
        ):
            raise IntegrityError(
                "Active attempt qualification is incomplete",
                "recovery_lineage_incomplete",
            )
        micro, micro_receipt = self.micro_pair
        safety, safety_receipt = self.safety_pair
        deployment, delivery_receipt = self.deployment_pair
        return QualifiedAttempt(
            self.attempt_id,
            self.opportunity,
            self.candidate,
            self.candidate_receipt,
            micro,
            micro_receipt,
            safety,
            safety_receipt,
            deployment,
            delivery_receipt,
        )


@dataclass(frozen=True, slots=True)
class RecoveredSearch:
    """Complete search memory reconstructed without reading a snapshot."""

    initial_diagnosis: Diagnosis
    diagnosis: Diagnosis
    accepted: tuple[AcceptedCandidate, ...]
    anchor: E2EMeasurement
    measurement_config: Path
    diagnostic_config: Path
    replay_config: Path
    diagnostic_history: tuple[str, ...]
    active: RecoveredAttempt | None


@dataclass(frozen=True, slots=True)
class _DiagnosisEntry:
    sequence: int
    diagnosis: Diagnosis


class _EventIndex:
    def __init__(self, record: E2ERunRecord) -> None:
        self.record = record
        self.events = EventJournal(record.root / "events" / "run.db").iter_events(
            record.run_id
        )
        self.by_key = {event.idempotency_key: event for event in self.events}

    def keyed(self, key: str) -> EventRecord | None:
        return self.by_key.get(key)

    def attempt_events(self, attempt_id: str) -> tuple[EventRecord, ...]:
        return tuple(
            event
            for event in self.events
            if event.payload.get("attempt_id") == attempt_id
        )

    def receipt(self, event: EventRecord, role: str) -> ArtifactReceipt:
        return _event_receipt(event, role)


def recover_search(
    record: E2ERunRecord,
    *,
    spec: E2EOptimizeSpec,
    views: BenchmarkConfigViews,
    baseline: E2EMeasurement,
) -> RecoveredSearch:
    """Replay accepted/current attempt evidence and recompute the live anchor."""

    search = record.controller.state.e2e
    if search is None:
        raise IntegrityError("E2E state is absent", "recovery_lineage_incomplete")
    index = _EventIndex(record)
    diagnoses = _diagnoses(record, index)
    current = _current_diagnosis(search.diagnostic_receipt, diagnoses)
    accepted, anchor = _accepted_chain(
        record,
        index,
        diagnoses,
        spec=spec,
        baseline=baseline,
        views=views,
    )
    configs = _active_configs(accepted, views)
    active = _active_attempt(
        record,
        index,
        diagnoses,
        search.stage,
        E2EAcceptancePolicy(spec.goal.gates),
    )
    if active is not None and active.promotion is not None and active.attempt_id not in {
        item.attempt_id for item in search.decisions
    }:
        _validate_active_promotion(record, active, accepted, configs)
    return RecoveredSearch(
        diagnoses[0].diagnosis,
        current,
        accepted,
        anchor,
        *configs,
        tuple(str(item.diagnosis.evidence_path) for item in diagnoses),
        active,
    )


def _diagnoses(
    record: E2ERunRecord, index: _EventIndex
) -> tuple[_DiagnosisEntry, ...]:
    values = recover_diagnosis_history(record)
    by_digest = {item[3].digest: item for item in values}
    entries = []
    for event in index.events:
        if event.payload.get("tool") != "kernel_opportunity_planner":
            continue
        receipt = index.receipt(event, "diagnosis_lineage")
        value = by_digest.get(receipt.digest)
        if value is None:
            raise IntegrityError("Diagnosis event is unbound", "invalid_diagnosis")
        plan, path, evidence, lineage, comparison = value
        entries.append(
            _DiagnosisEntry(
                event.sequence,
                Diagnosis(plan, path, evidence, lineage, comparison),
            )
        )
    if not entries:
        raise IntegrityError("Diagnosis history is empty", "diagnosis_not_committed")
    return tuple(entries)


def _current_diagnosis(
    receipt: str | None, diagnoses: tuple[_DiagnosisEntry, ...]
) -> Diagnosis:
    if receipt is None:
        raise IntegrityError("Current diagnosis is absent", "diagnosis_not_committed")
    matches = tuple(
        item.diagnosis
        for item in diagnoses
        if item.diagnosis.state_receipt.digest == receipt
    )
    if len(matches) != 1:
        raise IntegrityError("Current diagnosis is ambiguous", "invalid_diagnosis")
    return matches[0]


def _accepted_chain(
    record: E2ERunRecord,
    index: _EventIndex,
    diagnoses: tuple[_DiagnosisEntry, ...],
    *,
    spec: E2EOptimizeSpec,
    baseline: E2EMeasurement,
    views: BenchmarkConfigViews,
) -> tuple[tuple[AcceptedCandidate, ...], E2EMeasurement]:
    search = record.controller.state.e2e
    assert search is not None
    policy = E2EAcceptancePolicy(spec.goal.gates)
    accepted: list[AcceptedCandidate] = []
    anchor = baseline
    anchor_id = _initial_anchor_id(index)
    anchor_config = views.measurement
    anchor_image_id = None
    for decision in search.decisions:
        reward_event = _verify_decision_reward_transaction(index, decision.attempt_id)
        if decision.verdict == "reject":
            continue
        attempt = _recover_decided_attempt(
            record,
            index,
            diagnoses,
            attempt_id=decision.attempt_id,
            candidate_receipt=receipt_for_digest(
                record, decision.candidate_artifact_ref
            ),
            decision_receipt=receipt_for_digest(record, decision.evidence_ref),
            policy=policy,
        )
        promotion = attempt.promotion
        if promotion is None:
            raise IntegrityError(
                "Measured decision lacks a matched promotion pair",
                "recovery_lineage_incomplete",
            )
        verify_promotion_reward_binding(reward_event, promotion)
        qualified = attempt.qualified()
        validate_promotion_context(
            promotion,
            anchor_id=anchor_id,
            anchor_generation=len(accepted),
            anchor_config=anchor_config,
            anchor_image_id=anchor_image_id,
            deployment=qualified.deployment,
        )
        verdict = promotion.verdict
        expected = "keep" if verdict.keep else "revert"
        if expected != decision.verdict or verdict.reason_code != decision.reason:
            raise IntegrityError("Decision replay drifted", "e2e_decision_replay_mismatch")
        if not verdict.keep:
            continue
        accepted.append(
            AcceptedCandidate(
                qualified.candidate,
                qualified.opportunity,
                qualified.micro,
                qualified.safety,
                qualified.deployment,
                promotion.primary_measurement,
                decision.evidence_ref,
            )
        )
        anchor = promotion.primary_measurement
        anchor_config = qualified.deployment.measurement_config
        anchor_image_id = qualified.deployment.deployed_image_id
        event = index.keyed(f"e2e.attempt.{decision.attempt_id}.decision")
        assert event is not None
        anchor_id = str(event.payload.get("new_anchor_id", ""))
    if record.controller.state.anchor_generation != len(accepted):
        raise IntegrityError("Anchor generation drifted", "anchor_lineage_mismatch")
    return tuple(accepted), anchor


def _verify_decision_reward_transaction(index: _EventIndex, attempt_id: str) -> EventRecord:
    decision = index.keyed(f"e2e.attempt.{attempt_id}.decision")
    reward = index.keyed(f"e2e.attempt.{attempt_id}.reward")
    if (
        decision is None
        or reward is None
        or decision.event_type != "e2e.candidate_decided"
        or reward.event_type != "reward_committed"
        or decision.transaction_id != reward.transaction_id
    ):
        raise IntegrityError(
            "Decision and reward are not one transaction",
            "e2e_reward_transaction_mismatch",
        )
    return reward


def _recover_decided_attempt(
    record: E2ERunRecord,
    index: _EventIndex,
    diagnoses: tuple[_DiagnosisEntry, ...],
    *,
    attempt_id: str,
    candidate_receipt: ArtifactReceipt,
    decision_receipt: ArtifactReceipt,
    policy: E2EAcceptancePolicy,
) -> RecoveredAttempt:
    candidate_event = index.keyed(f"attempt.{attempt_id}.candidate")
    if candidate_event is None:
        raise IntegrityError("Candidate event is missing", "candidate_lineage_mismatch")
    if index.receipt(candidate_event, "candidate_manifest").digest != candidate_receipt.digest:
        raise IntegrityError("Candidate manifest drifted", "candidate_lineage_mismatch")
    candidate = recover_candidate(
        record,
        candidate_receipt,
        attempt_id=attempt_id,
        agent_identity=_agent_identity(index, attempt_id),
    )
    opportunity = _attempt_opportunity(index, diagnoses, attempt_id)
    parts = _attempt_parts(record, index, attempt_id, candidate, opportunity, policy)
    _verify_decision_document(
        record,
        decision_receipt,
        attempt_id=attempt_id,
        opportunity_id=opportunity.opportunity_id,
        candidate_id=candidate.candidate_id,
        candidate_receipt=candidate_receipt,
        parts=parts,
    )
    return RecoveredAttempt(
        attempt_id,
        opportunity,
        candidate,
        candidate_receipt,
        *parts,
    )


def _active_attempt(
    record: E2ERunRecord,
    index: _EventIndex,
    diagnoses: tuple[_DiagnosisEntry, ...],
    stage: SearchStage,
    policy: E2EAcceptancePolicy,
) -> RecoveredAttempt | None:
    search = record.controller.state.e2e
    assert search is not None
    attempt_id = search.active_attempt_id
    if attempt_id is None:
        return None
    opportunity = _attempt_opportunity(index, diagnoses, attempt_id)
    candidate_event = index.keyed(f"attempt.{attempt_id}.candidate")
    receipt = (
        index.receipt(candidate_event, "candidate_manifest")
        if candidate_event is not None
        else None
    )
    if search.candidate_artifact_ref is not None:
        committed = receipt_for_digest(record, search.candidate_artifact_ref)
        if receipt is not None and receipt.digest != committed.digest:
            raise IntegrityError("Candidate receipt drifted", "candidate_lineage_mismatch")
        receipt = committed
    candidate = None
    if receipt is not None:
        candidate = recover_candidate(
            record,
            receipt,
            attempt_id=attempt_id,
            agent_identity=_agent_identity(index, attempt_id),
        )
    if stage is not SearchStage.EXECUTING and candidate is None:
        raise IntegrityError("Active candidate is absent", "candidate_lineage_mismatch")
    parts = _attempt_parts(record, index, attempt_id, candidate, opportunity, policy)
    return RecoveredAttempt(
        attempt_id,
        opportunity,
        candidate,
        receipt,
        *parts,
    )


def _attempt_parts(
    record: E2ERunRecord,
    index: _EventIndex,
    attempt_id: str,
    candidate: E2ECandidate | None,
    opportunity: KernelOpportunity,
    policy: E2EAcceptancePolicy,
) -> tuple[
    tuple[Any, ArtifactReceipt] | None,
    tuple[Any, ArtifactReceipt] | None,
    tuple[Any, ArtifactReceipt] | None,
    MatchedPromotion | None,
]:
    if candidate is None or candidate.candidate_id is None:
        return None, None, None, None
    events = index.attempt_events(attempt_id)
    micro_receipt = _unique_role(events, "micro_qualification")
    safety_receipt = _unique_role(events, "safety_qualification")
    delivery_receipt = _unique_role(events, "primary_delivery")
    micro = _optional_pair(
        micro_receipt,
        lambda receipt: recover_micro(
            record, receipt, candidate_id=candidate.candidate_id or ""
        ),
    )
    safety = _optional_pair(
        safety_receipt,
        lambda receipt: recover_safety(
            record, receipt, candidate_id=candidate.candidate_id or ""
        ),
    )
    deployment = _optional_pair(
        delivery_receipt,
        lambda receipt: recover_deployment(
            record, receipt, candidate_id=candidate.candidate_id or ""
        ),
    )
    if deployment is not None:
        validate_deployment(deployment[0], candidate, _views_from_state(record))
        if deployment[0].deployed:
            _verify_deployment_config_bindings(record, events, deployment[0])
    pair_event = index.keyed(f"attempt.{attempt_id}.promotion_pair")
    promotion = None
    if pair_event is not None:
        promotion = recover_matched_promotion(
            record,
            pair_event=pair_event,
            events_by_key=index.by_key,
            protocol_hash=_protocol_hash(record),
            policy=policy,
            attempt_id=attempt_id,
            candidate_id=candidate.candidate_id,
            opportunity_id=opportunity.opportunity_id,
        )
        if deployment is None:
            raise IntegrityError(
                "Matched promotion lacks deployment", "recovery_lineage_incomplete"
            )
    return micro, safety, deployment, promotion


def _attempt_opportunity(
    index: _EventIndex,
    diagnoses: tuple[_DiagnosisEntry, ...],
    attempt_id: str,
) -> KernelOpportunity:
    selected = index.keyed(f"e2e.attempt.{attempt_id}.selected")
    if selected is None:
        raise IntegrityError("Attempt selection is missing", "attempt_lineage_missing")
    opportunity_id = selected.payload.get("opportunity_id")
    prior = tuple(item for item in diagnoses if item.sequence < selected.sequence)
    if not prior:
        raise IntegrityError("Attempt has no diagnosis", "attempt_lineage_missing")
    matches = tuple(
        item
        for item in prior[-1].diagnosis.plan.opportunities
        if item.opportunity_id == opportunity_id
    )
    if len(matches) != 1:
        raise IntegrityError("Attempt opportunity is missing", "attempt_lineage_missing")
    return matches[0]


def _verify_decision_document(
    record: E2ERunRecord,
    receipt: ArtifactReceipt,
    *,
    attempt_id: str,
    opportunity_id: str,
    candidate_id: str | None,
    candidate_receipt: ArtifactReceipt,
    parts: tuple[Any, Any, Any, Any],
) -> None:
    value = read_json_object(record, receipt, label="decision evidence")
    expected = {
        "attempt_id": attempt_id,
        "opportunity_id": opportunity_id,
        "candidate_id": candidate_id,
        "candidate_manifest_receipt": candidate_receipt.digest,
    }
    for name, observed in expected.items():
        if value.get(name) != observed:
            raise IntegrityError("Decision lineage drifted", "e2e_decision_lineage_mismatch")
    for name, pair in zip(
        ("micro_receipt", "safety_receipt", "delivery_receipt"),
        parts[:3],
        strict=True,
    ):
        if pair is None or value.get(name) != pair[1].digest:
            raise IntegrityError("Decision evidence is incomplete", "recovery_lineage_incomplete")
    promotion = parts[3]
    if (
        promotion is None
        or value.get("promotion_pair_receipt") != promotion.receipt.digest
        or value.get("measurement_verdict") != promotion.verdict.to_dict()
    ):
        raise IntegrityError("Decision lacks matched proof", "recovery_lineage_incomplete")


def _agent_identity(index: _EventIndex, attempt_id: str) -> Mapping[str, Any] | None:
    events = tuple(
        event
        for event in index.attempt_events(attempt_id)
        if event.event_type in {"agent_completed", "agent_failed"}
    )
    if len(events) != 1:
        return None
    event = events[0]
    if event.event_type == "agent_completed" and event.payload.get(
        "candidate_capture_allowed"
    ) is not True:
        raise IntegrityError("Agent capture was not allowed", "agent_lineage_missing")
    return event.payload


def _event_receipt(event: EventRecord, role: str) -> ArtifactReceipt:
    artifacts = event.payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise IntegrityError(f"{role} receipt is missing", "recovery_lineage_incomplete")
    matches = tuple(
        ArtifactReceipt.from_dict(dict(item["receipt"]))
        for item in artifacts
        if isinstance(item, Mapping)
        if item.get("role") == role and isinstance(item.get("receipt"), Mapping)
    )
    if len(matches) != 1:
        raise IntegrityError(f"{role} receipt is missing", "recovery_lineage_incomplete")
    return matches[0]


def _optional_pair(receipt, loader):
    return (loader(receipt), receipt) if receipt is not None else None


def _active_configs(
    accepted: tuple[AcceptedCandidate, ...], views: BenchmarkConfigViews
) -> tuple[Path, Path, Path]:
    if not accepted:
        return views.measurement, views.diagnostic, views.replay
    deployment = accepted[-1].deployment
    return (
        deployment.measurement_config,
        deployment.diagnostic_config,
        deployment.replay_config,
    )

def _initial_anchor_id(index: _EventIndex) -> str:
    events = tuple(event for event in index.events if event.event_type == "run.started")
    if len(events) != 1 or not isinstance(events[0].payload.get("initial_anchor_id"), str):
        raise IntegrityError("Initial anchor is missing", "anchor_lineage_mismatch")
    return str(events[0].payload["initial_anchor_id"])


def _validate_active_promotion(
    record: E2ERunRecord,
    active: RecoveredAttempt,
    accepted: tuple[AcceptedCandidate, ...],
    configs: tuple[Path, Path, Path],
) -> None:
    assert active.promotion is not None and active.deployment_pair is not None
    validate_promotion_context(
        active.promotion,
        anchor_id=record.controller.state.anchor_id,
        anchor_generation=record.controller.state.anchor_generation,
        anchor_config=configs[0],
        anchor_image_id=(accepted[-1].deployment.deployed_image_id if accepted else None),
        deployment=active.deployment_pair[0],
    )

def _protocol_hash(record: E2ERunRecord) -> str:
    search = record.controller.state.e2e
    if search is None:
        raise IntegrityError("E2E state is absent", "recovery_lineage_incomplete")
    return search.measurement_protocol_hash


def _views_from_state(record: E2ERunRecord) -> BenchmarkConfigViews:
    """Only workload hash is used by validate_deployment during recovery."""

    search = record.controller.state.e2e
    if search is None:
        raise IntegrityError("E2E state is absent", "recovery_lineage_incomplete")
    placeholder = record.root / "run.request.json"
    return BenchmarkConfigViews(
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        "0" * 64,
        search.measurement_protocol_hash,
        "",
        None,
    )

__all__ = ["RecoveredAttempt", "RecoveredSearch", "recover_search"]
