from __future__ import annotations

from apex.knowledge import (
    KnowledgeRetriever,
    KnowledgeScope,
    RetrievalQuery,
    normalize_operator_terms,
)


def _query(**changes) -> RetrievalQuery:
    values = {
        "scope": KnowledgeScope.from_mapping(
            {
                "operator": ["rms_norm"],
                "gpu_arch": ["gfx950"],
                "dtype": ["fp16"],
                "regime": ["decode"],
                "language": ["triton"],
                "framework": ["vllm"],
                "versions": {"rocm": "7.2"},
            }
        ),
        "independent_hypothesis": "Loads are repeated because fusion is missing.",
        "limit": 4,
        "max_tokens": 4_000,
    }
    return RetrievalQuery(**{**values, **changes})


def test_retrieval_is_byte_stable_bounded_and_complementary(card_factory) -> None:
    cards = (
        card_factory(claim="Fuse repeated loads"),
        card_factory(claim="MANDATED but still advisory", kind="procedure"),
        card_factory(claim="Avoid oversized blocks", kind="anti_pattern"),
        card_factory(claim="Prior no gain", kind="experience", status="stale"),
        card_factory(claim="Fifth lower priority", path="perf_knowledge/z.md"),
    )
    retriever = KnowledgeRetriever(cards)

    first = retriever.retrieve(_query())
    second = retriever.retrieve(_query())

    assert first == second
    assert first.digest == second.digest
    assert 2 <= len(first.cards) <= 4
    assert any(card.kind.value == "anti_pattern" for card in first.cards)
    assert first.unavailable_reason is None


def test_scope_mismatch_and_disabled_catalog_return_typed_empty(card_factory) -> None:
    cards = (
        card_factory(claim="gfx942 one", gpu_arch=("gfx942",)),
        card_factory(claim="gfx942 two", gpu_arch=("gfx942",), kind="anti_pattern"),
    )
    mismatch = KnowledgeRetriever(cards).retrieve(_query())
    disabled = KnowledgeRetriever(cards, enabled=False).retrieve(_query())

    assert mismatch.cards == ()
    assert mismatch.unavailable_reason == "insufficient_complementary_cards"
    assert disabled.cards == ()
    assert disabled.unavailable_reason == "knowledge_disabled"


def test_tiny_budget_never_injects_a_single_card(card_factory) -> None:
    cards = (card_factory(claim="one"), card_factory(claim="two", kind="anti_pattern"))
    selection = KnowledgeRetriever(cards).retrieve(_query(max_tokens=1))

    assert selection.cards == ()
    assert selection.unavailable_reason == "knowledge_budget_too_small"


def test_operator_symbols_map_to_reviewed_taxonomy_without_dropping_source_identity() -> None:
    assert normalize_operator_terms(("fused_add_rms_norm_kernel",)) == (
        "fused_add_rms_norm_kernel",
        "fused_add_rmsnorm",
        "rmsnorm",
    )
    assert set(normalize_operator_terms(("paged_attention_decode",))) == {
        "attention_decode_paged",
        "paged_attention_decode",
    }


def test_exact_operator_outranks_a_more_generic_wildcard(card_factory) -> None:
    exact = card_factory(claim="RMS-specific", operator=("rms_norm",))
    generic = card_factory(claim="Generic", operator=())

    selection = KnowledgeRetriever((generic, exact)).retrieve(_query())

    assert selection.cards[0] == exact
