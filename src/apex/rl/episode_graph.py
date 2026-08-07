"""Materialize RL episodes from the canonical journal and artifact CAS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from apex.context import ContextPacket
from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_json
from apex.orchestration import WorkloadState
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal, EventRecord

from .models import (
    CandidateEpisode,
    EpisodeArtifact,
    EpisodeEvent,
    EpisodeGraph,
    EvidenceClass,
    ParentEpisode,
    SemanticRole,
    episode_id,
)


@dataclass(slots=True)
class _ChildBuilder:
    attempt_id: str
    events: list[EpisodeEvent] = field(default_factory=list)
    candidate_id: str | None = None
    task_id: str | None = None
    kernel_id: str | None = None
    state_generation: int | None = None
    anchor_generation: int | None = None
    context_packet_id: str | None = None
    context_packet_receipt: ArtifactReceipt | None = None
    verdict: str | None = None
    scalar_reward: float | None = None
    reward_vector: Mapping[str, Any] | None = None
    reward_count: int = 0
    policy_ids: set[str] = field(default_factory=set)
    splits: set[str] = field(default_factory=set)
    visibilities: set[str] = field(default_factory=set)
    validation_reasons: set[str] = field(default_factory=set)


class EpisodeGraphMaterializer:
    """Read-only projection; it never appends events or writes mutable state."""

    def __init__(self, journal: EventJournal, artifacts: ArtifactStore) -> None:
        self._journal = journal
        self._artifacts = artifacts

    def materialize(
        self,
        run_id: str,
        *,
        workload_state: WorkloadState | None = None,
        context_packets: Mapping[str, ContextPacket] | None = None,
    ) -> EpisodeGraph:
        records = self._journal.iter_events(run_id, verify=True)
        if not records:
            raise ContractError("Run has no canonical events", "episode_run_empty")
        self._validate_state(run_id, records, workload_state)
        packets = context_packets or {}
        candidate_actions = _candidate_action_ids(records)
        parent_events: list[EpisodeEvent] = []
        children: dict[str, _ChildBuilder] = {}
        workload_id: str | None = None
        task_id: str | None = None
        provenance: Mapping[str, Any] = {}

        for record in records:
            event = self._project_event(record)
            payload = event.payload
            workload_id = workload_id or _text(payload.get("workload_id"))
            task_id = task_id or _text(payload.get("task_id"))
            if not provenance and isinstance(payload.get("provenance"), Mapping):
                provenance = dict(payload["provenance"])
            attempt_id = _attempt_id(record, candidate_actions)
            if attempt_id is None:
                parent_events.append(event)
                continue
            builder = children.setdefault(attempt_id, _ChildBuilder(attempt_id))
            builder.events.append(event)
            self._update_child(builder, event, packets, workload_state)

        kind = "workload" if workload_id is not None else "standalone_task"
        parent_id = episode_id(run_id, workload_id or task_id or "root")
        frozen_children = tuple(
            self._freeze_child(run_id, parent_id, item)
            for _, item in sorted(children.items())
        )
        all_policy_ids = tuple(
            sorted({policy for child in frozen_children for policy in child.policy_ids})
        )
        parent = ParentEpisode(
            episode_id=parent_id,
            kind=kind,
            run_id=run_id,
            workload_id=workload_id,
            task_id=task_id,
            events=tuple(parent_events),
            child_episode_ids=tuple(child.episode_id for child in frozen_children),
            terminal_status=_parent_status(records),
        )
        return EpisodeGraph(
            schema_version=1,
            run_id=run_id,
            high_water_mark=records[-1].sequence,
            journal_head_event_id=records[-1].event_id,
            workload_state_hash=(
                sha256_json(workload_state.to_dict()) if workload_state is not None else None
            ),
            parent=parent,
            children=frozen_children,
            provenance=provenance,
            policy_ids=all_policy_ids,
        )

    def _project_event(self, record: EventRecord) -> EpisodeEvent:
        artifacts = tuple(self._extract_artifacts(record))
        evidence = _evidence_class(record.payload.get("evidence_class"))
        return EpisodeEvent(
            sequence=record.sequence,
            event_id=record.event_id,
            parent_event_id=record.parent_event_id,
            event_type=record.event_type,
            semantic_role=_semantic_role(record.event_type),
            evidence_class=evidence,
            payload=record.payload,
            artifacts=artifacts,
            causation_id=_text(record.payload.get("causation_id")),
            correlation_id=_text(record.payload.get("correlation_id")),
            agent_run_id=_text(record.payload.get("agent_run_id")),
        )

    def _extract_artifacts(self, record: EventRecord) -> Sequence[EpisodeArtifact]:
        found: list[EpisodeArtifact] = []
        raw = record.payload.get("artifacts", ())
        if raw is not None and not isinstance(raw, (list, tuple)):
            raise ContractError("Event artifacts must be a list", "invalid_event_artifacts")
        for item in raw or ():
            if not isinstance(item, Mapping) or not isinstance(item.get("receipt"), Mapping):
                raise ContractError("Malformed event artifact binding", "invalid_event_artifacts")
            receipt = ArtifactReceipt.from_dict(dict(item["receipt"]))
            self._artifacts.verify(receipt)
            found.append(EpisodeArtifact(str(item.get("role", "")), receipt, record.event_id))
        special = {
            "context_packet_receipt": "context_packet",
            "candidate_receipt": "candidate",
            "measurement_receipt": "raw_measurement",
            "policy_source_receipt": "reward_policy",
        }
        existing = {(item.role, item.receipt.digest) for item in found}
        for key, role in special.items():
            value = record.payload.get(key)
            if not isinstance(value, Mapping):
                continue
            receipt = ArtifactReceipt.from_dict(dict(value))
            self._artifacts.verify(receipt)
            if (role, receipt.digest) not in existing:
                found.append(EpisodeArtifact(role, receipt, record.event_id))
        return sorted(found, key=lambda item: (item.role, item.receipt.digest))

    def _update_child(
        self,
        child: _ChildBuilder,
        event: EpisodeEvent,
        packets: Mapping[str, ContextPacket],
        workload_state: WorkloadState | None,
    ) -> None:
        payload = event.payload
        child.candidate_id = child.candidate_id or _text(payload.get("candidate_id"))
        child.task_id = child.task_id or _text(payload.get("task_id"))
        child.kernel_id = child.kernel_id or _text(payload.get("kernel_id"))
        _merge_int(child, "state_generation", payload.get("state_generation"))
        _merge_int(child, "anchor_generation", payload.get("anchor_generation"))
        _merge_int(child, "anchor_generation", payload.get("parent_anchor_generation"))
        split = _text(payload.get("split"))
        visibility = _text(payload.get("visibility"))
        if split:
            child.splits.add(split)
        if visibility:
            child.visibilities.add(visibility)
        if event.semantic_role is SemanticRole.DECISION:
            child.verdict = _text(payload.get("verdict")) or _decision_from_type(event.event_type)
        if event.semantic_role is SemanticRole.REWARD:
            self._capture_reward(child, event)
        context_artifacts = [
            item for item in event.artifacts if item.role == "context_packet"
        ]
        if len(context_artifacts) > 1:
            raise IntegrityError(
                "An event declared multiple ContextPacket observations",
                "multiple_context_packets",
            )
        context_artifact = context_artifacts[0] if context_artifacts else None
        if context_artifact is not None:
            self._capture_context(child, context_artifact.receipt, payload, packets, workload_state)

    def _capture_context(
        self,
        child: _ChildBuilder,
        receipt: ArtifactReceipt,
        payload: Mapping[str, Any],
        packets: Mapping[str, ContextPacket],
        workload_state: WorkloadState | None,
    ) -> None:
        try:
            raw_packet = self._artifacts.read_bytes(receipt)
            document = json.loads(raw_packet)
            identity = document["identity"]
            packet_id = str(identity["context_packet_id"])
            packet_state_generation = int(identity["state_generation"])
            anchor_generation = int(document["current_anchor"]["generation"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise IntegrityError(
                "ContextPacket artifact is not a canonical packet",
                "invalid_context_packet_artifact",
            ) from error
        if canonical_json_bytes(document) != raw_packet:
            raise IntegrityError(
                "ContextPacket artifact is not canonically serialized",
                "invalid_context_packet_artifact",
            )
        semantic = dict(document)
        semantic_identity = dict(identity)
        semantic_identity.pop("context_packet_id", None)
        semantic["identity"] = semantic_identity
        expected_packet_id = f"context-{sha256_json(semantic)[:24]}"
        if packet_id != expected_packet_id:
            raise IntegrityError(
                "ContextPacket content hash does not match its identity",
                "context_packet_id_mismatch",
            )
        declared = _text(payload.get("context_packet_id"))
        if declared is not None and declared != packet_id:
            raise IntegrityError("ContextPacket identity mismatch", "context_packet_id_mismatch")
        supplied = packets.get(packet_id)
        if supplied is not None and supplied.canonical_bytes != self._artifacts.read_bytes(receipt):
            raise IntegrityError("Supplied ContextPacket differs from CAS", "context_packet_mismatch")
        if workload_state is not None:
            max_generation = (
                workload_state.e2e.state_generation
                if workload_state.e2e is not None
                else workload_state.sequence
            )
            if packet_state_generation > max_generation:
                raise IntegrityError("ContextPacket is newer than workload state", "stale_state_projection")
            if anchor_generation > workload_state.anchor_generation:
                raise IntegrityError(
                    "ContextPacket anchor is newer than workload state",
                    "stale_state_projection",
                )
        if child.context_packet_receipt is not None:
            child.validation_reasons.add("multiple_context_packets")
        child.context_packet_receipt = receipt
        child.context_packet_id = packet_id
        _merge_int(child, "state_generation", packet_state_generation)
        _merge_int(child, "anchor_generation", anchor_generation)

    def _capture_reward(self, child: _ChildBuilder, event: EpisodeEvent) -> None:
        payload = event.payload
        child.reward_count += 1
        if child.reward_count > 1:
            child.validation_reasons.add("multiple_reward_events")
        policy = _text(payload.get("policy_id")) or _text(payload.get("reward_policy_id"))
        if policy:
            child.policy_ids.add(policy)
        vector = payload.get("reward_vector")
        if isinstance(vector, Mapping):
            child.reward_vector = dict(vector)
        scalar = payload.get("scalar_reward", payload.get("reward"))
        if scalar is not None:
            try:
                child.scalar_reward = float(scalar)
            except (TypeError, ValueError):
                child.validation_reasons.add("invalid_scalar_reward")
        roles = {item.role for candidate_event in child.events for item in candidate_event.artifacts}
        required = {
            "source": "reward_source_receipt_missing",
            "raw_measurement": "reward_measurement_receipt_missing",
            "reward_policy": "reward_policy_receipt_missing",
        }
        if not ({"harness", "reference"} & roles):
            child.validation_reasons.add("reward_harness_receipt_missing")
        for role, reason in required.items():
            if role not in roles:
                child.validation_reasons.add(reason)
        if not policy:
            child.validation_reasons.add("reward_policy_id_missing")
        if event.evidence_class is not EvidenceClass.MEASURED:
            child.validation_reasons.add("reward_not_measured")

    def _freeze_child(
        self, run_id: str, parent_id: str, child: _ChildBuilder
    ) -> CandidateEpisode:
        roles = {artifact.role for event in child.events for artifact in event.artifacts}
        failure = any(event.semantic_role is SemanticRole.FAILURE for event in child.events)
        if child.context_packet_receipt is None:
            child.validation_reasons.add("context_packet_missing")
        if not failure and not ({"candidate", "candidate_patch", "solution"} & roles):
            child.validation_reasons.add("candidate_artifact_missing")
        if not failure and child.verdict is None:
            child.validation_reasons.add("decision_missing")
        if len(child.splits) > 1:
            child.validation_reasons.add("conflicting_split")
        if len(child.visibilities) > 1:
            child.validation_reasons.add("conflicting_visibility")
        status = _child_status(child, failure)
        reasons = tuple(sorted(child.validation_reasons))
        return CandidateEpisode(
            episode_id=episode_id(run_id, child.attempt_id),
            parent_episode_id=parent_id,
            attempt_id=child.attempt_id,
            candidate_id=child.candidate_id,
            task_id=child.task_id,
            kernel_id=child.kernel_id,
            state_generation=child.state_generation,
            anchor_generation=child.anchor_generation,
            context_packet_id=child.context_packet_id,
            context_packet_receipt=child.context_packet_receipt,
            events=tuple(child.events),
            status=status,
            verdict=child.verdict,
            scalar_reward=child.scalar_reward,
            reward_vector=child.reward_vector,
            policy_ids=tuple(sorted(child.policy_ids)),
            split=next(iter(child.splits), "unspecified"),
            visibility=next(iter(child.visibilities), "unspecified"),
            trainability="complete" if not reasons else "truncated",
            validation_reasons=reasons,
        )

    @staticmethod
    def _validate_state(
        run_id: str,
        records: Sequence[EventRecord],
        state: WorkloadState | None,
    ) -> None:
        if state is None:
            return
        if state.run_id != run_id or state.sequence > records[-1].sequence:
            raise IntegrityError("WorkloadState is not anchored to this run", "state_run_mismatch")
        if state.sequence:
            event = next((item for item in records if item.sequence == state.sequence), None)
            if event is None or event.event_id != state.last_event_id:
                raise IntegrityError("WorkloadState head does not match journal", "state_head_mismatch")


def _candidate_action_ids(records: Sequence[EventRecord]) -> set[str]:
    return {
        str(record.payload["action_id"])
        for record in records
        if record.event_type in {"action.queued", "action_queued"}
        and "candidate" in str(record.payload.get("action_type", ""))
        and record.payload.get("action_id")
    }


def _attempt_id(record: EventRecord, candidate_actions: set[str]) -> str | None:
    explicit = _text(record.payload.get("attempt_id"))
    if explicit:
        return explicit
    candidate = _text(record.payload.get("candidate_id"))
    if candidate and _semantic_role(record.event_type) in {
        SemanticRole.ACTION,
        SemanticRole.OUTCOME,
        SemanticRole.DECISION,
        SemanticRole.REWARD,
        SemanticRole.FAILURE,
    }:
        return candidate
    action = _text(record.payload.get("action_id"))
    return action if action in candidate_actions else None


def _semantic_role(event_type: str) -> SemanticRole:
    normalized = event_type.replace(".", "_")
    if normalized in {"tool_called", "tool_result"}:
        return SemanticRole.TOOL
    if "reward" in normalized:
        return SemanticRole.REWARD
    if "cost" in normalized or normalized in {"usage_recorded"}:
        return SemanticRole.COST
    if normalized in {"error", "run_failed", "action_failed", "agent_failed"}:
        return SemanticRole.FAILURE
    if normalized in {
        "decision",
        "e2e_candidate_decided",
        "action_committed",
        "action_aborted",
    }:
        return SemanticRole.DECISION
    if normalized in {
        "compile_result",
        "correctness_result",
        "safety_result",
        "measurement_result",
        "e2e_result",
        "delivery_verified",
        "action_verified",
        "e2e_micro_verified",
        "e2e_safety_verified",
        "e2e_delivery_verified",
    }:
        return SemanticRole.OUTCOME
    if normalized in {
        "observation_created",
        "context_packet_created",
        "knowledge_read",
        "knowledge_outcome_linked",
        "e2e_baseline_committed",
        "e2e_diagnostics_committed",
        "e2e_reprofiled",
    }:
        return SemanticRole.OBSERVATION
    if normalized in {
        "prompt_sent",
        "agent_message",
        "candidate_materialized",
        "candidate_frozen",
        "delivery_materialized",
        "action_queued",
        "action_started",
        "action_artifacts_ready",
        "e2e_candidate_frozen",
    }:
        return SemanticRole.ACTION
    return SemanticRole.CONTROL


def _evidence_class(value: object) -> EvidenceClass:
    if value is None:
        return EvidenceClass.UNSPECIFIED
    try:
        return EvidenceClass(str(value))
    except ValueError as error:
        raise ContractError("Unknown evidence class", "invalid_evidence_class") from error


def _parent_status(records: Sequence[EventRecord]) -> str:
    terminal = records[-1].event_type.replace(".", "_")
    if terminal in {"run_succeeded", "run_finished"}:
        return str(records[-1].payload.get("status", "succeeded"))
    if terminal == "run_failed":
        return "failed"
    if terminal == "run_cancelled":
        return "cancelled"
    return "incomplete"


def _child_status(child: _ChildBuilder, failure: bool) -> str:
    compile_failed = any(
        event.event_type.replace(".", "_") == "compile_result"
        and event.payload.get("passed") is False
        for event in child.events
    )
    correctness_failed = any(
        event.event_type.replace(".", "_") == "correctness_result"
        and event.payload.get("passed") is False
        for event in child.events
    )
    if compile_failed:
        return "compile_failed"
    if correctness_failed:
        return "wrong"
    if failure:
        return "infrastructure_error"
    if child.verdict == "keep":
        return "success"
    if child.verdict in {"revert", "reject"}:
        return "no_gain"
    if child.verdict == "needs_more_measurement":
        return "no_measurement"
    return "incomplete"


def _decision_from_type(event_type: str) -> str | None:
    normalized = event_type.replace(".", "_")
    if normalized == "action_committed":
        return "keep"
    if normalized == "action_aborted":
        return "revert"
    return None


def _merge_int(child: _ChildBuilder, field_name: str, value: object) -> None:
    if value is None:
        return
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        child.validation_reasons.add(f"invalid_{field_name}")
        return
    existing = getattr(child, field_name)
    if existing is not None and existing != parsed:
        child.validation_reasons.add(f"conflicting_{field_name}")
    elif existing is None:
        setattr(child, field_name, parsed)


def _text(value: object) -> str | None:
    return None if value is None or not str(value).strip() else str(value)


__all__ = ["EpisodeGraphMaterializer"]
