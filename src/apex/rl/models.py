"""Immutable, deterministic projections for Apex RL episodes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from apex.core import ContractError, canonical_json_bytes, sha256_json, validate_identifier
from apex.storage import ArtifactReceipt


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class EvidenceClass(str, Enum):
    """Authority attached to an observation or outcome."""

    MEASURED = "measured"
    DIAGNOSTIC = "diagnostic"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    SELF_REPORTED = "self_reported"
    UNSPECIFIED = "unspecified"


class SemanticRole(str, Enum):
    """RL meaning of one canonical journal event."""

    CONTROL = "control"
    OBSERVATION = "observation"
    ACTION = "action"
    TOOL = "tool"
    OUTCOME = "outcome"
    DECISION = "decision"
    REWARD = "reward"
    COST = "cost"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class EpisodeArtifact:
    """A role-labelled, verified CAS receipt referenced by an event."""

    role: str
    receipt: ArtifactReceipt
    event_id: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ContractError("Artifact role is required", "missing_artifact_role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "event_id": self.event_id,
            "receipt": self.receipt.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EpisodeEvent:
    """Lossless semantic view of a canonical event record."""

    sequence: int
    event_id: str
    transaction_id: str
    parent_event_id: str | None
    event_type: str
    semantic_role: SemanticRole
    evidence_class: EvidenceClass
    payload: Mapping[str, Any]
    artifacts: tuple[EpisodeArtifact, ...]
    causation_id: str | None = None
    correlation_id: str | None = None
    agent_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "transaction_id": self.transaction_id,
            "parent_event_id": self.parent_event_id,
            "event_type": self.event_type,
            "semantic_role": self.semantic_role.value,
            "evidence_class": self.evidence_class.value,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "agent_run_id": self.agent_run_id,
            "payload": dict(self.payload),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class CandidateEpisode:
    """Every attempt, including failures and non-best candidates."""

    episode_id: str
    parent_episode_id: str
    attempt_id: str
    candidate_id: str | None
    opportunity_id: str | None
    task_id: str | None
    kernel_id: str | None
    state_generation: int | None
    anchor_generation: int | None
    context_packet_id: str | None
    context_packet_receipt: ArtifactReceipt | None
    events: tuple[EpisodeEvent, ...]
    status: str
    verdict: str | None
    scalar_reward: float | None
    reward_vector: Mapping[str, Any] | None
    policy_ids: tuple[str, ...]
    split: str
    visibility: str
    trainability: str
    validation_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "parent_episode_id": self.parent_episode_id,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "opportunity_id": self.opportunity_id,
            "task_id": self.task_id,
            "kernel_id": self.kernel_id,
            "state_generation": self.state_generation,
            "anchor_generation": self.anchor_generation,
            "context_packet_id": self.context_packet_id,
            "context_packet_receipt": (
                self.context_packet_receipt.to_dict()
                if self.context_packet_receipt is not None
                else None
            ),
            "events": [event.to_dict() for event in self.events],
            "status": self.status,
            "verdict": self.verdict,
            "scalar_reward": self.scalar_reward,
            "reward_vector": dict(self.reward_vector) if self.reward_vector else None,
            "policy_ids": list(self.policy_ids),
            "split": self.split,
            "visibility": self.visibility,
            "trainability": self.trainability,
            "validation_reasons": list(self.validation_reasons),
        }


@dataclass(frozen=True, slots=True)
class ParentEpisode:
    """Root standalone task or parent E2E workload episode."""

    episode_id: str
    kind: str
    run_id: str
    workload_id: str | None
    task_id: str | None
    events: tuple[EpisodeEvent, ...]
    child_episode_ids: tuple[str, ...]
    terminal_status: str
    task_reward: float | None
    reward_vector: Mapping[str, Any] | None
    reward_policy_id: str | None
    reward_policy_digest: str | None
    reward_source_receipt: str | None
    raw_measurement_receipts: tuple[str, ...]
    trainability: str
    untrainable_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "kind": self.kind,
            "run_id": self.run_id,
            "workload_id": self.workload_id,
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
            "child_episode_ids": list(self.child_episode_ids),
            "terminal_status": self.terminal_status,
            "task_reward": self.task_reward,
            "reward_vector": (
                dict(self.reward_vector) if self.reward_vector is not None else None
            ),
            "reward_policy_id": self.reward_policy_id,
            "reward_policy_digest": self.reward_policy_digest,
            "reward_source_receipt": self.reward_source_receipt,
            "raw_measurement_receipts": list(self.raw_measurement_receipts),
            "trainability": self.trainability,
            "untrainable_reason": self.untrainable_reason,
        }


@dataclass(frozen=True, slots=True)
class EpisodeGraph:
    """Rebuildable parent/child projection over journal and CAS evidence."""

    schema_version: int
    run_id: str
    high_water_mark: int
    journal_head_event_id: str
    workload_state_hash: str | None
    parent: ParentEpisode
    children: tuple[CandidateEpisode, ...]
    provenance: Mapping[str, Any]
    policy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field_name="run_id")
        if self.schema_version != 1 or self.high_water_mark < 1:
            raise ContractError("Unsupported EpisodeGraph", "invalid_episode_graph")
        if self.workload_state_hash is not None and not _DIGEST.fullmatch(
            self.workload_state_hash
        ):
            raise ContractError("Invalid workload state hash", "invalid_episode_graph")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": "apex.episode_graph",
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "high_water_mark": self.high_water_mark,
            "journal_head_event_id": self.journal_head_event_id,
            "workload_state_hash": self.workload_state_hash,
            "parent": self.parent.to_dict(),
            "children": [child.to_dict() for child in self.children],
            "provenance": dict(self.provenance),
            "policy_ids": list(self.policy_ids),
        }

    @property
    def graph_id(self) -> str:
        return f"episode-graph-{sha256_json(self.to_dict())[:24]}"

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def episode_id(*parts: str) -> str:
    """Derive an opaque stable ID without leaking path-like identifiers."""

    return f"episode-{sha256_json(list(parts))[:24]}"


__all__ = [
    "CandidateEpisode",
    "EpisodeArtifact",
    "EpisodeEvent",
    "EpisodeGraph",
    "EvidenceClass",
    "ParentEpisode",
    "SemanticRole",
    "episode_id",
]
