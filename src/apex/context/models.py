"""Immutable ContextPacket values shown to stateless agent backends."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from apex.core import ContractError, canonical_json_bytes, sha256_json, validate_identifier
from apex.knowledge import KnowledgeCard


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
Scalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Enforced input/action limits plus an advisory response allocation."""

    input_tokens: int
    response_token_allocation: int
    turns: int
    wall_seconds: int
    gpu_seconds_remaining: int

    def __post_init__(self) -> None:
        if min(
            self.input_tokens,
            self.response_token_allocation,
            self.turns,
            self.wall_seconds,
            self.gpu_seconds_remaining,
        ) < 1:
            raise ContractError("Context budgets must be positive", "invalid_context_budget")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "input_tokens": self.input_tokens,
            "response_token_allocation": self.response_token_allocation,
            "response_token_enforcement": "context_advisory_not_backend_enforced",
            "turns": self.turns,
            "wall_seconds": self.wall_seconds,
            "gpu_seconds_remaining": self.gpu_seconds_remaining,
        }


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Independent diagnosis formed from live evidence before KB retrieval."""

    hypothesis_id: str
    mechanism: str
    falsification_condition: str

    def __post_init__(self) -> None:
        validate_identifier(self.hypothesis_id, field_name="hypothesis_id")
        _require_text(self.mechanism, "hypothesis.mechanism")
        _require_text(self.falsification_condition, "hypothesis.falsification_condition")

    def to_dict(self) -> dict[str, str]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "mechanism": self.mechanism,
            "falsification_condition": self.falsification_condition,
        }


@dataclass(frozen=True, slots=True)
class TargetEvidence:
    """Measured target and content-addressed bottleneck evidence."""

    opportunity_id: str
    source_and_symbol: str
    phase_shape_regime: str
    evidence_receipts: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.opportunity_id, field_name="opportunity_id")
        _require_text(self.source_and_symbol, "target.source_and_symbol")
        _require_text(self.phase_shape_regime, "target.phase_shape_regime")
        _validate_digests(self.evidence_receipts, "target.evidence_receipts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "source_and_symbol": self.source_and_symbol,
            "phase_shape_regime": self.phase_shape_regime,
            "evidence_receipts": list(self.evidence_receipts),
        }


@dataclass(frozen=True, slots=True)
class AnchorView:
    """Exact current anchor facts; an LLM cannot rewrite this value."""

    anchor_id: str
    generation: int
    metrics: tuple[tuple[str, Scalar], ...]
    accepted_patch_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.anchor_id, field_name="anchor_id")
        if self.generation < 0 or not self.metrics:
            raise ContractError("Anchor facts are incomplete", "invalid_context_anchor")
        _validate_digests(self.accepted_patch_refs, "anchor.accepted_patch_refs", allow_empty=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "generation": self.generation,
            "metrics": dict(self.metrics),
            "accepted_patch_refs": list(self.accepted_patch_refs),
        }


@dataclass(frozen=True, slots=True)
class AttemptView:
    """Receipt-linked measured attempt selected for the current target."""

    candidate_id: str
    outcome: str
    mechanism: str
    evidence_receipts: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.candidate_id, field_name="candidate_id")
        _require_text(self.outcome, "attempt.outcome")
        _require_text(self.mechanism, "attempt.mechanism")
        _validate_digests(self.evidence_receipts, "attempt.evidence_receipts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "outcome": self.outcome,
            "mechanism": self.mechanism,
            "evidence_receipts": list(self.evidence_receipts),
        }


@dataclass(frozen=True, slots=True)
class DeadEndView:
    """A scoped failed strategy plus the exact condition permitting retry."""

    strategy_fingerprint: str
    reason: str
    retry_condition: str
    applicability_hash: str
    evidence_receipts: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.strategy_fingerprint, "dead_end.strategy_fingerprint")
        _validate_digest(self.applicability_hash, "dead_end.applicability_hash")
        _require_text(self.reason, "dead_end.reason")
        _require_text(self.retry_condition, "dead_end.retry_condition")
        _validate_digests(self.evidence_receipts, "dead_end.evidence_receipts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_fingerprint": self.strategy_fingerprint,
            "reason": self.reason,
            "retry_condition": self.retry_condition,
            "applicability_hash": self.applicability_hash,
            "evidence_receipts": list(self.evidence_receipts),
        }


@dataclass(frozen=True, slots=True)
class CampaignAttemptView:
    """Receipt-linked outcome retained across different E2E opportunities."""

    attempt_id: str
    opportunity_id: str
    candidate_id: str | None
    verdict: str
    reason: str
    anchor_generation: int
    context_packet_id: str
    evidence_receipts: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.attempt_id, field_name="attempt_id")
        validate_identifier(self.opportunity_id, field_name="opportunity_id")
        if self.candidate_id is not None:
            validate_identifier(self.candidate_id, field_name="candidate_id")
        if self.verdict not in {
            "keep",
            "revert",
            "reject",
            "needs_more_measurement",
        }:
            raise ContractError(
                "Campaign attempt verdict is invalid", "invalid_campaign_history"
            )
        if self.anchor_generation < 0:
            raise ContractError(
                "Campaign attempt anchor is invalid", "invalid_campaign_history"
            )
        validate_identifier(self.context_packet_id, field_name="context_packet_id")
        _require_text(self.reason, "campaign_attempt.reason")
        _validate_digests(
            self.evidence_receipts, "campaign_attempt.evidence_receipts"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "opportunity_id": self.opportunity_id,
            "candidate_id": self.candidate_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "anchor_generation": self.anchor_generation,
            "context_packet_id": self.context_packet_id,
            "evidence_receipts": list(self.evidence_receipts),
        }


@dataclass(frozen=True, slots=True)
class AdvisoryCard:
    """Quoted card projection; it has no instruction or execution authority."""

    card_id: str
    content_hash: str
    kind: str
    status: str
    claim: str
    apply: str
    verify: str
    caution: str
    source: tuple[tuple[str, str], ...]

    @classmethod
    def from_card(cls, card: KnowledgeCard) -> "AdvisoryCard":
        return cls(
            card_id=card.card_id,
            content_hash=card.content_hash,
            kind=card.kind.value,
            status=card.status.value,
            claim=card.claim,
            apply=card.apply,
            verify=card.verify,
            caution=card.caution,
            source=tuple(sorted(card.source.to_dict().items())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "content_hash": self.content_hash,
            "kind": self.kind,
            "status": self.status,
            "claim": self.claim,
            "apply": self.apply,
            "verify": self.verify,
            "caution": self.caution,
            "source": dict(self.source),
        }


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Read-only content receipt; large artifact bodies stay outside the packet."""

    kind: str
    sha256: str
    locator: str

    def __post_init__(self) -> None:
        _require_text(self.kind, "artifact.kind")
        _validate_digest(self.sha256, "artifact.sha256")
        expected = f"artifact://sha256/{self.sha256.removeprefix('sha256:')}"
        if self.locator != expected:
            raise ContractError(
                "Artifact locator is not content addressed", "invalid_artifact_locator"
            )

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "sha256": self.sha256, "locator": self.locator}


@dataclass(frozen=True, slots=True)
class ContextContract:
    """Hard action, edit, verification, and output boundaries."""

    allowed_actions: tuple[str, ...]
    editable_files: tuple[str, ...]
    acceptance_policy: str
    stop_policy: str
    required_output_schema: str = "ActionProposal"

    def __post_init__(self) -> None:
        if not self.allowed_actions or not self.editable_files:
            raise ContractError("Context action contract is empty", "invalid_context_contract")
        for action in self.allowed_actions:
            _require_text(action, "context.allowed_action")
        for path in self.editable_files:
            _validate_relative_path(path)
        for field in (self.acceptance_policy, self.stop_policy, self.required_output_schema):
            _require_text(field, "context.contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_actions": list(self.allowed_actions),
            "editable_files": list(self.editable_files),
            "acceptance_policy": self.acceptance_policy,
            "stop_policy": self.stop_policy,
            "required_output_schema": self.required_output_schema,
        }


@dataclass(frozen=True, slots=True)
class ContextPacket:
    """Canonical, task-local observation for one stateless invocation."""

    run_id: str
    workload_id: str
    phase: str
    cycle: int
    state_generation: int
    role_kind: str
    role_objective: str
    primary_metric: str
    hard_constraints: tuple[str, ...]
    target: TargetEvidence
    hypothesis: Hypothesis
    current_anchor: AnchorView
    attempts: tuple[AttemptView, ...]
    dead_ends: tuple[DeadEndView, ...]
    knowledge_cards: tuple[AdvisoryCard, ...]
    knowledge_selection_receipt: str | None
    knowledge_unavailable_reason: str | None
    budget: ContextBudget
    contract: ContextContract
    artifact_refs: tuple[ArtifactReference, ...]
    campaign_attempts: tuple[CampaignAttemptView, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field_name="run_id")
        validate_identifier(self.workload_id, field_name="workload_id")
        if self.cycle < 0 or self.state_generation < 0 or not self.hard_constraints:
            raise ContractError(
                "Context identity or constraints are invalid", "invalid_context_packet"
            )
        if self.current_anchor.generation > self.state_generation:
            raise ContractError("Anchor is newer than state", "invalid_context_generation")
        for field in (self.phase, self.role_kind, self.role_objective, self.primary_metric):
            _require_text(field, "context.mandatory_fact")
        if self.knowledge_cards and len(self.knowledge_cards) not in (2, 3, 4):
            raise ContractError(
                "Context must contain zero or 2-4 cards", "invalid_context_card_count"
            )
        if self.knowledge_cards:
            _validate_digest(self.knowledge_selection_receipt, "knowledge.selection_receipt")
        elif self.knowledge_selection_receipt is not None:
            _validate_digest(self.knowledge_selection_receipt, "knowledge.selection_receipt")
        if not self.knowledge_cards and not self.knowledge_unavailable_reason:
            raise ContractError(
                "Empty knowledge requires a typed reason", "missing_knowledge_status"
            )

    @property
    def context_packet_id(self) -> str:
        return f"context-{sha256_json(self.semantic_dict())[:24]}"

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "identity": {
                "run_id": self.run_id,
                "workload_id": self.workload_id,
                "phase": self.phase,
                "cycle": self.cycle,
                "anchor_id": self.current_anchor.anchor_id,
                "state_generation": self.state_generation,
            },
            "role": {"kind": self.role_kind, "objective": self.role_objective},
            "objective": {
                "primary_metric": self.primary_metric,
                "hard_constraints": list(self.hard_constraints),
            },
            "target": self.target.to_dict(),
            "hypothesis": self.hypothesis.to_dict(),
            "current_anchor": self.current_anchor.to_dict(),
            "relevant_history": {
                "attempts": [item.to_dict() for item in self.attempts],
                "dead_ends": [item.to_dict() for item in self.dead_ends],
                "campaign_attempts": [
                    item.to_dict() for item in self.campaign_attempts
                ],
            },
            "knowledge": {
                "selection_receipt": self.knowledge_selection_receipt,
                "unavailable_reason": self.knowledge_unavailable_reason,
                "cards": [card.to_dict() for card in self.knowledge_cards],
            },
            "budget": self.budget.to_dict(),
            "contract": self.contract.to_dict(),
            "artifact_refs": [item.to_dict() for item in self.artifact_refs],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.semantic_dict()
        value["identity"]["context_packet_id"] = self.context_packet_id
        return value

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def freeze_metrics(values: Mapping[str, Scalar]) -> tuple[tuple[str, Scalar], ...]:
    """Copy JSON-scalar metrics into a sorted immutable representation."""

    scalars = (str, int, float, bool, type(None))
    if not values or any(not isinstance(item, scalars) for item in values.values()):
        raise ContractError("Anchor metrics must be JSON scalars", "invalid_context_metrics")
    if any(isinstance(item, float) and not math.isfinite(item) for item in values.values()):
        raise ContractError("Anchor metrics must be finite", "invalid_context_metrics")
    if any(not str(key).strip() for key in values):
        raise ContractError("Anchor metric keys are empty", "invalid_context_metrics")
    return tuple(sorted((str(key), item) for key, item in values.items()))


def _validate_digests(values: tuple[str, ...], field: str, *, allow_empty: bool = False) -> None:
    if not values and not allow_empty:
        raise ContractError(f"{field} is empty", "missing_context_receipt")
    for value in values:
        _validate_digest(value, field)


def _validate_digest(value: object, field: str) -> None:
    if value is None or not _DIGEST.fullmatch(str(value).removeprefix("sha256:")):
        raise ContractError(f"Invalid {field}", "invalid_context_receipt")


def _require_text(value: object, field: str) -> None:
    if not str(value).strip() or len(str(value)) > 8_192:
        raise ContractError(f"Invalid {field}", "invalid_context_text")


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ContractError("Unsafe editable path", "unsafe_context_editable_path")


__all__ = [
    "AdvisoryCard",
    "AnchorView",
    "ArtifactReference",
    "AttemptView",
    "ContextBudget",
    "ContextContract",
    "ContextPacket",
    "DeadEndView",
    "Hypothesis",
    "TargetEvidence",
    "freeze_metrics",
]
