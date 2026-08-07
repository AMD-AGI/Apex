"""Deterministic compiler from authoritative facts to a bounded packet."""

from __future__ import annotations

from dataclasses import dataclass, replace

from apex.core import ContractError, sha256_json
from apex.knowledge import (
    ExperienceIdentity,
    ExperienceOutcome,
    ExperienceRecord,
    ExperienceView,
    KnowledgeRetriever,
    KnowledgeScope,
    KnowledgeSelection,
    RetrievalQuery,
)

from .models import (
    AdvisoryCard,
    AnchorView,
    ArtifactReference,
    AttemptView,
    ContextBudget,
    ContextContract,
    ContextPacket,
    DeadEndView,
    Hypothesis,
    TargetEvidence,
)
from .renderer import render_context_packet


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Versioned deterministic section limits for one observation."""

    policy_id: str = "context_packet_v1"
    max_attempts: int = 6
    max_dead_ends: int = 4
    max_knowledge_tokens: int = 1_600
    chars_per_token: int = 4

    def __post_init__(self) -> None:
        if min(
            self.max_attempts,
            self.max_dead_ends,
            self.max_knowledge_tokens,
            self.chars_per_token,
        ) < 1:
            raise ContractError("Context policy limits must be positive", "invalid_context_policy")


@dataclass(frozen=True, slots=True)
class ContextCompileRequest:
    """Authoritative inputs from one committed state generation."""

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
    budget: ContextBudget
    contract: ContextContract
    artifact_refs: tuple[ArtifactReference, ...]
    retrieval_scope: KnowledgeScope
    experience_identity: ExperienceIdentity
    experience_view: ExperienceView


@dataclass(frozen=True, slots=True)
class CompiledContext:
    """Packet plus the exact read receipt that a journal adapter records."""

    packet: ContextPacket
    knowledge_selection: KnowledgeSelection
    estimated_input_tokens: int
    policy_id: str

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "context_packet_id": self.packet.context_packet_id,
            "context_packet_sha256": sha256_json(self.packet.to_dict()),
            "estimated_input_tokens": self.estimated_input_tokens,
            "policy_id": self.policy_id,
            "knowledge_selection": self.knowledge_selection.to_dict(),
        }


class ContextCompiler:
    """Compile one packet without consulting mutable global files or transcripts."""

    def __init__(
        self,
        retriever: KnowledgeRetriever,
        *,
        policy: ContextPolicy | None = None,
    ) -> None:
        self._retriever = retriever
        self._policy = policy or ContextPolicy()

    def compile(self, request: ContextCompileRequest) -> CompiledContext:
        empty_selection = self._retriever.retrieve(
            RetrievalQuery(
                scope=request.retrieval_scope,
                independent_hypothesis=request.hypothesis.mechanism,
                max_tokens=1,
            )
        )
        base = self._base_packet(request, empty_selection)
        base_tokens = self._tokens(base)
        if base_tokens > request.budget.input_tokens:
            raise ContractError(
                "Mandatory context exceeds input budget", "mandatory_context_over_budget"
            )
        knowledge = self._retrieve_knowledge(request, request.budget.input_tokens - base_tokens)
        packet = self._base_packet(request, knowledge)
        if self._tokens(packet) > request.budget.input_tokens:
            raise ContractError(
                "Knowledge cards exceed context budget", "knowledge_context_over_budget"
            )
        packet = self._add_history(packet, request)
        return CompiledContext(packet, knowledge, self._tokens(packet), self._policy.policy_id)

    def _retrieve_knowledge(
        self, request: ContextCompileRequest, remaining_tokens: int
    ) -> KnowledgeSelection:
        budget = max(1, min(self._policy.max_knowledge_tokens, remaining_tokens))
        return self._retriever.retrieve(
            RetrievalQuery(
                scope=request.retrieval_scope,
                independent_hypothesis=request.hypothesis.mechanism,
                limit=4,
                max_tokens=budget,
            )
        )

    def _base_packet(
        self, request: ContextCompileRequest, selection: KnowledgeSelection
    ) -> ContextPacket:
        cards = tuple(AdvisoryCard.from_card(card) for card in selection.cards)
        return ContextPacket(
            run_id=request.run_id,
            workload_id=request.workload_id,
            phase=request.phase,
            cycle=request.cycle,
            state_generation=request.state_generation,
            role_kind=request.role_kind,
            role_objective=request.role_objective,
            primary_metric=request.primary_metric,
            hard_constraints=request.hard_constraints,
            target=request.target,
            hypothesis=request.hypothesis,
            current_anchor=request.current_anchor,
            attempts=(),
            dead_ends=(),
            knowledge_cards=cards,
            knowledge_selection_receipt=selection.digest,
            knowledge_unavailable_reason=selection.unavailable_reason,
            budget=request.budget,
            contract=request.contract,
            artifact_refs=request.artifact_refs,
        )

    def _add_history(self, packet: ContextPacket, request: ContextCompileRequest) -> ContextPacket:
        records = request.experience_view.compatible(
            request.experience_identity,
            limit=self._policy.max_attempts + self._policy.max_dead_ends,
        )
        attempts = tuple(_attempt(record) for record in records[: self._policy.max_attempts])
        dead_ends = tuple(
            _dead_end(record, request.experience_identity)
            for record in records
            if record.outcome is not ExperienceOutcome.SUCCESS and record.retry_condition
        )[: self._policy.max_dead_ends]
        return self._fit_history(packet, attempts, dead_ends)

    def _fit_history(
        self,
        packet: ContextPacket,
        attempts: tuple[AttemptView, ...],
        dead_ends: tuple[DeadEndView, ...],
    ) -> ContextPacket:
        selected_attempts: list[AttemptView] = []
        selected_dead_ends: list[DeadEndView] = []
        for dead_end in dead_ends:
            candidate = replace(packet, dead_ends=tuple((*selected_dead_ends, dead_end)))
            if self._tokens(candidate) <= packet.budget.input_tokens:
                selected_dead_ends.append(dead_end)
        packet = replace(packet, dead_ends=tuple(selected_dead_ends))
        for attempt in attempts:
            candidate = replace(packet, attempts=tuple((*selected_attempts, attempt)))
            if self._tokens(candidate) <= packet.budget.input_tokens:
                selected_attempts.append(attempt)
        return replace(packet, attempts=tuple(selected_attempts))

    def _tokens(self, packet: ContextPacket) -> int:
        rendered = render_context_packet(packet).encode("utf-8")
        rounded = len(rendered) + self._policy.chars_per_token - 1
        return max(1, rounded // self._policy.chars_per_token)


def _attempt(record: ExperienceRecord) -> AttemptView:
    return AttemptView(
        candidate_id=record.candidate_id,
        outcome=record.outcome.value,
        mechanism=record.mechanism,
        evidence_receipts=record.evidence_receipts,
    )


def _dead_end(record: ExperienceRecord, identity: ExperienceIdentity) -> DeadEndView:
    return DeadEndView(
        strategy_fingerprint=record.strategy_fingerprint,
        reason=record.failure_reason or record.outcome.value,
        retry_condition=record.retry_condition or "provenance_or_shape_changes",
        applicability_hash=sha256_json(identity.to_dict()),
        evidence_receipts=record.evidence_receipts,
    )


__all__ = [
    "CompiledContext",
    "ContextCompileRequest",
    "ContextCompiler",
    "ContextPolicy",
]
