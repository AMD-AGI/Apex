"""Read-only knowledge capability projected from the attributed card catalog."""

from __future__ import annotations

from typing import Any, Mapping

from apex.core import ContractError
from apex.knowledge import KnowledgeRetriever, KnowledgeScope, RetrievalQuery
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityKind,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRewardRole,
    CapabilitySideEffect,
)


def knowledge_search_descriptor() -> CapabilityDescriptor:
    """Return the canonical descriptor without import-time construction."""

    return CapabilityDescriptor(
        capability_id="knowledge.search",
        title="Search AMD kernel knowledge",
        summary=(
            "Retrieve bounded, attributed, inert advisory cards after an independent "
            "kernel hypothesis has been formed."
        ),
        kind=CapabilityKind.TOOL,
        input_schema={
            "type": "object",
            "properties": {
                "gpu_arch": {"type": "string", "minLength": 1},
                "language": {"type": "string", "minLength": 1},
                "independent_hypothesis": {"type": "string", "minLength": 1},
                "operator": {"type": ["string", "null"]},
                "framework": {"type": ["string", "null"]},
                "regime": {"type": ["string", "null"]},
                "dtype": {"type": ["string", "null"]},
                "software_version": {"type": ["string", "null"]},
                "limit": {"type": "integer", "minimum": 2, "maximum": 4},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 8000},
            },
            "required": ["gpu_arch", "language", "independent_hypothesis"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "selection_policy": {"type": "string"},
                "card_ids": {"type": "array", "items": {"type": "string"}},
                "cards": {"type": "array", "items": {"type": "object"}},
                "token_count": {"type": "integer"},
                "unavailable_reason": {"type": ["string", "null"]},
                "advisory_only": {"const": True},
            },
            "required": [
                "selection_policy",
                "card_ids",
                "cards",
                "token_count",
                "unavailable_reason",
                "advisory_only",
            ],
            "additionalProperties": False,
        },
        side_effects=(CapabilitySideEffect.NONE,),
        required_authority=CapabilityAuthority.NONE,
        gpu_requirement=CapabilityGpuRequirement.NONE,
        timeout_seconds=5,
        artifact_classes=("attributed_advisory",),
        reward_role=CapabilityRewardRole.INELIGIBLE,
    )


def knowledge_explain_descriptor() -> CapabilityDescriptor:
    """Describe exact-card lookup without turning card prose into authority."""

    return CapabilityDescriptor(
        capability_id="knowledge.explain",
        title="Explain one AMD kernel knowledge card",
        summary="Return one exact attributed inert advisory card by stable ID.",
        kind=CapabilityKind.TOOL,
        input_schema={
            "type": "object",
            "properties": {"card_id": {"type": "string", "minLength": 1}},
            "required": ["card_id"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "card_id": {"type": "string"},
                "card": {"type": ["object", "null"]},
                "unavailable_reason": {"type": ["string", "null"]},
                "advisory_only": {"const": True},
            },
            "required": [
                "card_id",
                "card",
                "unavailable_reason",
                "advisory_only",
            ],
            "additionalProperties": False,
        },
        side_effects=(CapabilitySideEffect.NONE,),
        required_authority=CapabilityAuthority.NONE,
        gpu_requirement=CapabilityGpuRequirement.NONE,
        timeout_seconds=5,
        artifact_classes=("attributed_advisory",),
        reward_role=CapabilityRewardRole.INELIGIBLE,
    )


class KnowledgeSearchHandler:
    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        arguments = request.arguments
        query = RetrievalQuery(
            scope=KnowledgeScope(
                operator=_term(arguments, "operator"),
                gpu_arch=_required_term(arguments, "gpu_arch"),
                language=_required_term(arguments, "language"),
                framework=_term(arguments, "framework"),
                regime=_term(arguments, "regime"),
                dtype=_term(arguments, "dtype"),
                versions=_version(arguments.get("software_version")),
            ),
            independent_hypothesis=_required_text(arguments, "independent_hypothesis"),
            limit=_integer(arguments, "limit", 4),
            max_tokens=_integer(arguments, "max_tokens", 1600),
        )
        selection = self._retriever.retrieve(query)
        return CapabilityResult(
            capability_id=request.capability_id,
            content={
                "selection_policy": selection.selection_policy,
                "card_ids": [card.card_id for card in selection.cards],
                "cards": [card.to_dict() for card in selection.cards],
                "token_count": selection.token_count,
                "unavailable_reason": selection.unavailable_reason,
                "advisory_only": True,
            },
        )


class KnowledgeExplainHandler:
    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        card_id = _required_text(request.arguments, "card_id")
        card = self._retriever.get_card(card_id)
        return CapabilityResult(
            capability_id=request.capability_id,
            content={
                "card_id": card_id,
                "card": card.to_dict() if card is not None else None,
                "unavailable_reason": None if card is not None else "knowledge_card_unavailable",
                "advisory_only": True,
            },
        )


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ContractError(f"{key} is required", "invalid_capability_arguments")
    return item.strip()


def _required_term(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return (_required_text(value, key),)


def _term(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if item is None:
        return ()
    if not isinstance(item, str) or not item.strip():
        raise ContractError(f"{key} is invalid", "invalid_capability_arguments")
    return (item.strip(),)


def _version(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, str) or not value.strip():
        raise ContractError("software_version is invalid", "invalid_capability_arguments")
    return (("software", value.strip()),)


def _integer(value: Mapping[str, Any], key: str, default: int) -> int:
    item = value.get(key, default)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ContractError(f"{key} is invalid", "invalid_capability_arguments")
    return item


__all__ = [
    "KnowledgeExplainHandler",
    "KnowledgeSearchHandler",
    "knowledge_explain_descriptor",
    "knowledge_search_descriptor",
]
