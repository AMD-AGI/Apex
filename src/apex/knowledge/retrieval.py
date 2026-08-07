"""Deterministic, bounded, scope-first knowledge retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from apex.core import ContractError, canonical_json_bytes, sha256_json

from .cards import CardKind, CardStatus, KnowledgeCard, KnowledgeScope, validate_catalog


SELECTION_POLICY_ID = "scope_complementary_advisory_v1"
_NON_WORD = re.compile(r"[^a-z0-9]+")
_OPERATOR_ALIASES = (
    (("fused", "add", "rms"), ("fused_add_rmsnorm", "rmsnorm")),
    (("allreduce", "rms"), ("fused_allreduce_rmsnorm", "rmsnorm")),
    (("rms", "norm"), ("rmsnorm",)),
    (("rmsnorm",), ("rmsnorm",)),
    (("paged", "att"), ("attention_decode_paged",)),
    (("paged", "attention"), ("attention_decode_paged",)),
    (("flash", "att"), ("attention_prefill_fmha",)),
    (("flash", "attention"), ("attention_prefill_fmha",)),
    (("fused", "moe"), ("fused_moe_grouped_gemm", "grouped_gemm_moe")),
    (("grouped", "moe"), ("grouped_gemm_moe", "fused_moe_grouped_gemm")),
    (("grouped", "gemm"), ("grouped_gemm_moe", "batched_gemm")),
    (("topk",), ("moe_routing_topk", "sampling_topk_topp")),
    (("layer", "norm"), ("layernorm",)),
    (("softmax",), ("softmax",)),
    (("rope",), ("rope", "mrope")),
    (("gemv",), ("skinny_gemv_decode",)),
    (("gemm",), ("dense_gemm", "batched_gemm")),
)


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """A query formed only after an independent live-evidence hypothesis."""

    scope: KnowledgeScope
    independent_hypothesis: str
    limit: int = 4
    max_tokens: int = 1_600

    def __post_init__(self) -> None:
        if not self.independent_hypothesis.strip():
            raise ContractError(
                "Independent hypothesis is required", "missing_independent_hypothesis"
            )
        if not 2 <= self.limit <= 4:
            raise ContractError(
                "Knowledge limit must be between 2 and 4", "invalid_knowledge_limit"
            )
        if self.max_tokens < 1:
            raise ContractError(
                "Knowledge token budget must be positive", "invalid_knowledge_budget"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "independent_hypothesis": self.independent_hypothesis,
            "limit": self.limit,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeSelection:
    """Auditable selection result suitable for a ``knowledge_read`` event."""

    query: RetrievalQuery
    selection_policy: str
    cards: tuple[KnowledgeCard, ...]
    token_count: int
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "query_sha256": sha256_json(self.query.to_dict()),
            "selection_policy": self.selection_policy,
            "card_ids": [card.card_id for card in self.cards],
            "card_content_sha256": [card.content_hash for card in self.cards],
            "token_count": self.token_count,
            "unavailable_reason": self.unavailable_reason,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class KnowledgeRetriever:
    """Read-only catalog retriever; it never mutates or promotes cards."""

    def __init__(self, cards: tuple[KnowledgeCard, ...], *, enabled: bool = True) -> None:
        self._cards = validate_catalog(cards)
        self._enabled = enabled

    def retrieve(self, query: RetrievalQuery) -> KnowledgeSelection:
        if not self._enabled:
            return _unavailable(query, "knowledge_disabled")
        eligible = tuple(card for card in self._cards if card.scope.matches(query.scope))
        if len(eligible) < 2:
            return _unavailable(query, "insufficient_complementary_cards")
        ordered = _complementary_order(eligible, query.scope)
        selected = _fit_budget(ordered, query.limit, query.max_tokens)
        if len(selected) < 2:
            return _unavailable(query, "knowledge_budget_too_small")
        return KnowledgeSelection(
            query=query,
            selection_policy=SELECTION_POLICY_ID,
            cards=selected,
            token_count=sum(_card_tokens(card) for card in selected),
        )


def normalize_operator_terms(symbols: tuple[str, ...]) -> tuple[str, ...]:
    """Map source symbols to the reviewed GEAK operator taxonomy without guessing a winner."""

    terms: set[str] = set()
    for symbol in symbols:
        normalized = _NON_WORD.sub("_", symbol.lower()).strip("_")
        if normalized:
            terms.add(normalized)
        tokens = frozenset(part for part in normalized.split("_") if part)
        for required, aliases in _OPERATOR_ALIASES:
            if all(term in tokens or term in normalized for term in required):
                terms.update(aliases)
    return tuple(sorted(terms))


def _unavailable(query: RetrievalQuery, reason: str) -> KnowledgeSelection:
    return KnowledgeSelection(query, SELECTION_POLICY_ID, (), 0, reason)


def _complementary_order(
    cards: tuple[KnowledgeCard, ...], query_scope: KnowledgeScope
) -> tuple[KnowledgeCard, ...]:
    ranked = sorted(cards, key=lambda card: (_rank_key(card, query_scope), card.card_id))
    ordered: list[KnowledgeCard] = []
    primary = next((card for card in ranked if not _is_counterexample(card)), ranked[0])
    ordered.append(primary)
    counterexample = next((card for card in ranked if _is_counterexample(card)), None)
    if counterexample is not None and counterexample not in ordered:
        ordered.append(counterexample)
    for kind in CardKind:
        match = next((card for card in ranked if card.kind is kind and card not in ordered), None)
        if match is not None:
            ordered.append(match)
    ordered.extend(card for card in ranked if card not in ordered)
    return tuple(ordered)


def _rank_key(card: KnowledgeCard, query_scope: KnowledgeScope) -> tuple[int, int, str]:
    weights = {
        "operator": 64,
        "gpu_arch": 16,
        "dtype": 8,
        "regime": 4,
        "language": 2,
        "framework": 1,
    }
    exact = sum(
        weight
        for name, weight in weights.items()
        if bool(getattr(card.scope, name))
        and bool(set(getattr(card.scope, name)).intersection(getattr(query_scope, name)))
    )
    status_rank = {
        CardStatus.VALIDATED: 0,
        CardStatus.IMPORTED_UNVERIFIED: 1,
        CardStatus.STALE: 2,
        CardStatus.CONTRADICTED: 3,
    }[card.status]
    return (-exact, status_rank, card.source.path)


def _is_counterexample(card: KnowledgeCard) -> bool:
    return card.kind is CardKind.ANTI_PATTERN or card.status in {
        CardStatus.STALE,
        CardStatus.CONTRADICTED,
    }


def _fit_budget(
    cards: tuple[KnowledgeCard, ...], limit: int, max_tokens: int
) -> tuple[KnowledgeCard, ...]:
    selected: list[KnowledgeCard] = []
    used = 0
    for card in cards:
        cost = _card_tokens(card)
        if used + cost <= max_tokens:
            selected.append(card)
            used += cost
        if len(selected) == limit:
            break
    return tuple(selected)


def _card_tokens(card: KnowledgeCard) -> int:
    return max(1, (len(canonical_json_bytes(card.to_dict())) + 3) // 4)


__all__ = [
    "KnowledgeRetriever",
    "KnowledgeSelection",
    "normalize_operator_terms",
    "RetrievalQuery",
    "SELECTION_POLICY_ID",
]
