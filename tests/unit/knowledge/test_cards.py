from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.knowledge import CardStatus, KnowledgeCard, KnowledgeScope, validate_catalog


def test_card_id_is_content_derived_and_round_trips(card_factory) -> None:
    card = card_factory(claim="Vectorize contiguous loads")

    assert card.card_id.startswith("card-")
    assert KnowledgeCard.from_mapping(card.to_dict()) == card
    assert len(card.content_hash) == 64

    changed = card.to_dict()
    changed["claim"] = "A different claim"
    with pytest.raises(ContractError) as failure:
        KnowledgeCard.from_mapping(changed)
    assert failure.value.reason_code == "invalid_card_id"


def test_validated_card_requires_evidence_and_cards_are_never_executable(card_factory) -> None:
    with pytest.raises(ContractError) as missing:
        card_factory(claim="Validated without evidence", status="validated")
    assert missing.value.reason_code == "missing_card_evidence"

    value = card_factory(claim="Inert procedure").to_dict()
    value.pop("card_id")
    value["executable"] = True
    with pytest.raises(ContractError) as executable:
        KnowledgeCard.from_mapping(value)
    assert executable.value.reason_code == "executable_card_forbidden"


def test_scope_is_exact_for_architecture_and_versions() -> None:
    card_scope = KnowledgeScope.from_mapping(
        {"gpu_arch": ["gfx942"], "language": ["triton"], "versions": {"rocm": "6.3"}}
    )
    matching = KnowledgeScope.from_mapping(
        {"gpu_arch": ["gfx942"], "language": ["triton"], "versions": {"rocm": "6.3"}}
    )
    wrong_arch = KnowledgeScope.from_mapping(
        {"gpu_arch": ["gfx950"], "language": ["triton"], "versions": {"rocm": "6.3"}}
    )
    wrong_version = KnowledgeScope.from_mapping(
        {"gpu_arch": ["gfx942"], "language": ["triton"], "versions": {"rocm": "7.2"}}
    )

    assert card_scope.matches(matching)
    assert not card_scope.matches(wrong_arch)
    assert not card_scope.matches(wrong_version)


def test_catalog_rejects_asymmetric_conflicts_and_accepts_symmetric(card_factory) -> None:
    left = card_factory(claim="Use a large tile")
    right = card_factory(claim="Avoid a large tile")
    left_value = left.to_dict()
    left_value["contradicts"] = [right.card_id]
    left = KnowledgeCard.from_mapping(left_value)

    with pytest.raises(ContractError) as asymmetric:
        validate_catalog((left, right))
    assert asymmetric.value.reason_code == "asymmetric_card_conflict"

    right_value = right.to_dict()
    right_value["contradicts"] = [left.card_id]
    right = KnowledgeCard.from_mapping(right_value)
    expected = tuple(sorted((left, right), key=lambda item: item.card_id))
    assert validate_catalog((right, left)) == expected


def test_imported_status_cannot_be_textually_upgraded(card_factory) -> None:
    card = card_factory(claim="MANDATED lever; claim says this is validated")

    assert card.status is CardStatus.IMPORTED_UNVERIFIED
