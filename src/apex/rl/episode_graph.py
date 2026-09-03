"""Materialize RL episodes from the canonical journal and artifact CAS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from apex.context import ContextPacket
from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_json,
    validate_identifier,
)
from apex.orchestration import WorkloadState
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal, EventRecord

from .e2e_validation import (
    E2E_REWARD_POLICY_ID,
    MEASURED_E2E_ARTIFACT_ROLES,
    artifact_roles,
    e2e_completion_reasons,
    e2e_outcome_transaction_reasons,
    explicit_attempt_id,
    optional_identifier,
    transaction_members,
)
from .episode_semantics import (
    decision_from_type as _decision_from_type,
    evidence_class as _evidence_class,
    parent_status as _parent_status,
    semantic_role as _semantic_role,
    text as _text,
)
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
from .kernel_measurement_validation import (
    kernel_gate_reward_evidence_reasons,
    kernel_measurement_evidence_reasons,
)
from .projection_validation import merge_projected_identifier, merge_projected_int
from .parent_reward import project_parent_reward
from .state_validation import validate_workload_state
from .e2e_server_lineage_validation import validate_e2e_server_lineage


@dataclass(slots=True)
class _ChildBuilder:
    attempt_id: str
    events: list[EpisodeEvent] = field(default_factory=list)
    candidate_id: str | None = None
    opportunity_id: str | None = None
    task_id: str | None = None
    kernel_id: str | None = None
    state_generation: int | None = None
    anchor_generation: int | None = None
    context_packet_id: str | None = None
    context_packet_receipt: ArtifactReceipt | None = None
    verdict: str | None = None
    scalar_reward: float | None = None
    reward_vector: Mapping[str, Any] | None = None
    decision_reason: str | None = None
    decision_count: int = 0
    e2e_decision_count: int = 0
    reward_count: int = 0
    is_e2e: bool = False
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
        validate_workload_state(run_id, records, workload_state)
        packets = context_packets or {}
        parent_events: list[EpisodeEvent] = []
        children: dict[str, _ChildBuilder] = {}
        candidate_owners: dict[str, str] = {}
        transactions = transaction_members(records)
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
            attempt_id = explicit_attempt_id(record)
            if attempt_id is None:
                parent_events.append(event)
                continue
            candidate_id = optional_identifier(payload, "candidate_id")
            if candidate_id is not None:
                owner = candidate_owners.setdefault(candidate_id, attempt_id)
                if owner != attempt_id:
                    raise IntegrityError(
                        "Candidate ID belongs to multiple attempts",
                        "candidate_id_mismatch",
                    )
            builder = children.setdefault(attempt_id, _ChildBuilder(attempt_id))
            builder.events.append(event)
            self._update_child(builder, event, packets, workload_state)

        parent_id = episode_id(run_id, workload_id or task_id or "root")
        frozen_children = tuple(
            self._freeze_child(run_id, parent_id, item, transactions)
            for _, item in sorted(children.items())
        )
        parent, all_policy_ids = _freeze_parent(
            run_id,
            records,
            tuple(parent_events),
            frozen_children,
            workload_id,
            task_id,
            self._artifacts,
        )
        graph = EpisodeGraph(
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
        lineage_events = (
            *parent.events,
            *(event for child in frozen_children for event in child.events),
        )
        validate_e2e_server_lineage(
            run_id, tuple(sorted(lineage_events, key=lambda event: event.sequence)),
            self._artifacts,
        )
        return graph

    def _project_event(self, record: EventRecord) -> EpisodeEvent:
        artifacts = tuple(self._extract_artifacts(record))
        evidence = _evidence_class(record.payload.get("evidence_class"))
        return EpisodeEvent(
            sequence=record.sequence,
            event_id=record.event_id,
            transaction_id=record.transaction_id,
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
        normalized = event.event_type.replace(".", "_")
        child.is_e2e = child.is_e2e or normalized.startswith("e2e_")
        merge_projected_identifier(
            child, "candidate_id", optional_identifier(payload, "candidate_id")
        )
        merge_projected_identifier(
            child,
            "opportunity_id",
            optional_identifier(payload, "opportunity_id"),
        )
        child.task_id = child.task_id or _text(payload.get("task_id"))
        child.kernel_id = child.kernel_id or _text(payload.get("kernel_id"))
        if not normalized.startswith("e2e_"):
            merge_projected_int(child, "state_generation", payload.get("state_generation"))
        merge_projected_int(child, "anchor_generation", payload.get("anchor_generation"))
        merge_projected_int(child, "anchor_generation", payload.get("parent_anchor_generation"))
        split = _text(payload.get("split"))
        visibility = _text(payload.get("visibility"))
        if split:
            child.splits.add(split)
        if visibility:
            child.visibilities.add(visibility)
        if event.semantic_role is SemanticRole.DECISION:
            child.decision_count += 1
            if child.decision_count > 1:
                child.validation_reasons.add("multiple_decision_events")
            if normalized == "e2e_candidate_decided":
                child.e2e_decision_count += 1
            child.verdict = _text(payload.get("verdict")) or _decision_from_type(event.event_type)
            child.decision_reason = _text(payload.get("reason"))
            untrainable = _text(payload.get("untrainable_reason"))
            if payload.get("trainability") == "untrainable" and untrainable:
                child.validation_reasons.add(f"untrainable:{untrainable}")
        if event.semantic_role is SemanticRole.REWARD:
            self._capture_reward(child, event)
        if normalized == "experience_deferred":
            child.validation_reasons.add("external_evaluation_pending")
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
            target = document["target"]
            packet_id = str(identity["context_packet_id"])
            opportunity_id = str(target["opportunity_id"])
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
        try:
            opportunity_id = validate_identifier(
                opportunity_id,
                field_name="opportunity_id",
            )
        except ContractError as error:
            raise IntegrityError(
                "ContextPacket opportunity identity is invalid",
                "opportunity_id_invalid",
            ) from error
        merge_projected_identifier(child, "opportunity_id", opportunity_id)
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
        merge_projected_int(child, "state_generation", packet_state_generation)
        merge_projected_int(child, "anchor_generation", anchor_generation)

    def _capture_reward(self, child: _ChildBuilder, event: EpisodeEvent) -> None:
        payload = event.payload
        child.reward_count += 1
        if child.reward_count > 1:
            child.validation_reasons.add("multiple_reward_events")
        policy = _text(payload.get("policy_id")) or _text(payload.get("reward_policy_id"))
        if policy:
            child.policy_ids.add(policy)
        if policy == E2E_REWARD_POLICY_ID:
            child.is_e2e = True
        vector = payload.get("reward_vector")
        if isinstance(vector, Mapping):
            child.reward_vector = dict(vector)
        scalar = payload.get("scalar_reward", payload.get("reward"))
        if scalar is not None:
            try:
                child.scalar_reward = float(scalar)
            except (TypeError, ValueError):
                child.validation_reasons.add("invalid_scalar_reward")
        roles = artifact_roles(child.events)
        if policy == E2E_REWARD_POLICY_ID:
            self._validate_e2e_reward(child, event, roles)
            return
        self._validate_kernel_reward(child, event, roles, policy)

    def _validate_kernel_reward(
        self,
        child: _ChildBuilder,
        event: EpisodeEvent,
        roles: set[str],
        policy: str | None,
    ) -> None:
        required = {
            "source": "reward_source_receipt_missing",
            "reward_policy": "reward_policy_receipt_missing",
        }
        for role, reason in required.items():
            if role not in roles:
                child.validation_reasons.add(reason)
        if not policy:
            child.validation_reasons.add("reward_policy_id_missing")
        if event.evidence_class is not EvidenceClass.MEASURED:
            child.validation_reasons.add("reward_not_measured")
        vector = child.reward_vector
        stage = vector.get("kernel_reward_stage") if isinstance(vector, Mapping) else None
        if stage == "measurement":
            self._validate_kernel_measurement_reward(child, roles)
        elif stage in {"compile", "correctness"}:
            self._validate_kernel_gate_reward(child, roles, str(stage))
        else:
            child.validation_reasons.add("kernel_reward_stage_invalid")

    def _validate_kernel_measurement_reward(
        self, child: _ChildBuilder, roles: set[str]
    ) -> None:
        required = {
            "raw_measurement": "reward_measurement_receipt_missing",
            "measurement_execution": "reward_measurement_execution_receipt_missing",
        }
        if not ({"harness", "reference"} & roles):
            child.validation_reasons.add("reward_harness_receipt_missing")
        for role, reason in required.items():
            if role not in roles:
                child.validation_reasons.add(reason)
        child.validation_reasons.update(
            kernel_measurement_evidence_reasons(child.events, self._artifacts)
        )

    def _validate_kernel_gate_reward(
        self, child: _ChildBuilder, roles: set[str], stage: str
    ) -> None:
        required = {"compile_evidence"}
        if stage == "correctness":
            required.add("correctness_evidence")
        if not required <= roles:
            child.validation_reasons.add("reward_gate_evidence_missing")
        child.validation_reasons.update(
            kernel_gate_reward_evidence_reasons(
                child.events, self._artifacts, stage
            )
        )

    @staticmethod
    def _validate_e2e_reward(
        child: _ChildBuilder,
        event: EpisodeEvent,
        roles: set[str],
    ) -> None:
        required = {
            "candidate_manifest": "candidate_manifest_receipt_missing",
            "decision_evidence": "decision_evidence_receipt_missing",
            "e2e_reward_vector": "e2e_reward_vector_receipt_missing",
            "reward_policy": "reward_policy_receipt_missing",
        }
        for role, reason in required.items():
            if role not in roles:
                child.validation_reasons.add(reason)
        if event.evidence_class is not EvidenceClass.DERIVED:
            child.validation_reasons.add("e2e_reward_not_derived")
        vector = child.reward_vector
        if not isinstance(vector, Mapping):
            child.validation_reasons.add("reward_vector_missing")
            return
        if vector.get("policy_id") != E2E_REWARD_POLICY_ID:
            child.validation_reasons.add("reward_policy_id_mismatch")
        if event.payload.get("policy_digest") != vector.get("policy_digest"):
            child.validation_reasons.add("reward_policy_digest_mismatch")
        if event.payload.get("verdict") != vector.get("verdict") or (
            event.payload.get("reason_code") != vector.get("reason_code")
        ):
            child.validation_reasons.add("reward_decision_mismatch")
        verdict = vector.get("verdict")
        if verdict in {"keep", "revert"} and not MEASURED_E2E_ARTIFACT_ROLES.issubset(roles):
            child.validation_reasons.add("e2e_measurement_evidence_missing")

    def _freeze_child(
        self,
        run_id: str,
        parent_id: str,
        child: _ChildBuilder,
        transaction_members: Mapping[str, Sequence[str]],
    ) -> CandidateEpisode:
        roles = artifact_roles(child.events)
        observed_failure = any(
            event.semantic_role is SemanticRole.FAILURE for event in child.events
        )
        e2e_terminal = child.is_e2e and (
            child.decision_count > 0 or child.reward_count > 0
        )
        failure = observed_failure and not e2e_terminal if child.is_e2e else observed_failure
        if child.context_packet_receipt is None:
            child.validation_reasons.add("context_packet_missing")
        if child.is_e2e:
            child.validation_reasons.update(
                e2e_completion_reasons(
                    infrastructure_failure=failure,
                    terminal=e2e_terminal,
                    e2e_decision_count=child.e2e_decision_count,
                    reward_count=child.reward_count,
                    roles=roles,
                    vector=child.reward_vector,
                    candidate_id=child.candidate_id,
                    opportunity_id=child.opportunity_id,
                    verdict=child.verdict,
                    decision_reason=child.decision_reason,
                )
            )
            if e2e_terminal and not failure:
                child.validation_reasons.update(
                    e2e_outcome_transaction_reasons(
                        child.events,
                        transaction_members,
                    )
                )
        elif not failure and not ({"candidate", "candidate_patch", "solution"} & roles):
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
            opportunity_id=child.opportunity_id,
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


def _freeze_parent(
    run_id: str,
    records: Sequence[EventRecord],
    events: tuple[EpisodeEvent, ...],
    children: tuple[CandidateEpisode, ...],
    workload_id: str | None,
    task_id: str | None,
    artifacts: ArtifactStore,
) -> tuple[ParentEpisode, tuple[str, ...]]:
    reward = project_parent_reward(run_id, events, artifacts)
    policy_ids = tuple(sorted({
        *reward.policy_ids,
        *(policy for child in children for policy in child.policy_ids),
    }))
    parent = ParentEpisode(
        episode_id=episode_id(run_id, workload_id or task_id or "root"),
        kind="e2e_kernel_only" if workload_id is not None else "single_kernel",
        run_id=run_id,
        workload_id=workload_id,
        task_id=task_id,
        events=events,
        child_episode_ids=tuple(child.episode_id for child in children),
        terminal_status=_parent_status(records),
        task_reward=reward.task_reward,
        reward_vector=reward.reward_vector,
        reward_policy_id=reward.reward_policy_id,
        reward_policy_digest=reward.reward_policy_digest,
        reward_source_receipt=reward.reward_source_receipt,
        raw_measurement_receipts=reward.raw_measurement_receipts,
        trainability=reward.trainability,
        untrainable_reason=reward.untrainable_reason,
    )
    return parent, policy_ids


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


__all__ = ["EpisodeGraphMaterializer"]
