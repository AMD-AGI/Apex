"""Typed, provenance-bearing advisory knowledge cards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from apex.core import ContractError, sha256_json, validate_identifier


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_MAX_TEXT = 16_384


class CardKind(str, Enum):
    """How a card should be interpreted by a consumer."""

    FACT = "fact"
    PROCEDURE = "procedure"
    EXPERIENCE = "experience"
    ANTI_PATTERN = "anti_pattern"


class CardStatus(str, Enum):
    """Evidence status; status never grants execution authority."""

    IMPORTED_UNVERIFIED = "imported_unverified"
    VALIDATED = "validated"
    STALE = "stale"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class KnowledgeScope:
    """Structured applicability; empty dimensions are explicit wildcards."""

    operator: tuple[str, ...] = ()
    gpu_arch: tuple[str, ...] = ()
    dtype: tuple[str, ...] = ()
    regime: tuple[str, ...] = ()
    language: tuple[str, ...] = ()
    framework: tuple[str, ...] = ()
    versions: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "KnowledgeScope":
        raw = value or {}
        versions = raw.get("versions", {})
        if not isinstance(versions, Mapping):
            raise ContractError("scope.versions must be a mapping", "invalid_card_scope")
        try:
            return cls(
                operator=_terms(raw.get("operator", ())),
                gpu_arch=_terms(raw.get("gpu_arch", ())),
                dtype=_terms(raw.get("dtype", ())),
                regime=_terms(raw.get("regime", ())),
                language=_terms(raw.get("language", ())),
                framework=_terms(raw.get("framework", ())),
                versions=tuple(sorted((str(key), str(item)) for key, item in versions.items())),
            )
        except (TypeError, ValueError) as error:
            raise ContractError("Malformed knowledge scope", "invalid_card_scope") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": list(self.operator),
            "gpu_arch": list(self.gpu_arch),
            "dtype": list(self.dtype),
            "regime": list(self.regime),
            "language": list(self.language),
            "framework": list(self.framework),
            "versions": dict(self.versions),
        }

    def matches(self, query: "KnowledgeScope") -> bool:
        dimensions = ("operator", "gpu_arch", "dtype", "regime", "language", "framework")
        if any(
            not _dimension_matches(getattr(self, name), getattr(query, name))
            for name in dimensions
        ):
            return False
        query_versions = dict(query.versions)
        return all(query_versions.get(key) == value for key, value in self.versions)

    def overlaps(self, other: "KnowledgeScope") -> bool:
        dimensions = ("operator", "gpu_arch", "dtype", "regime", "language", "framework")
        return all(
            _dimension_overlaps(getattr(self, name), getattr(other, name))
            for name in dimensions
        )


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Immutable origin for the exact source bytes transformed into a card."""

    repository: str
    git_sha: str
    path: str
    license: str
    content_sha256: str
    transform_version: str

    def __post_init__(self) -> None:
        if not all(
            (self.repository.strip(), self.license.strip(), self.transform_version.strip())
        ):
            raise ContractError("Incomplete card provenance", "incomplete_card_provenance")
        if not _GIT_SHA.fullmatch(self.git_sha) or not _SHA256.fullmatch(self.content_sha256):
            raise ContractError("Invalid provenance digest", "invalid_card_provenance")
        _validate_source_path(self.path)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceProvenance":
        try:
            return cls(
                repository=str(value["repository"]),
                git_sha=str(value["git_sha"]),
                path=str(value["path"]),
                license=str(value["license"]),
                content_sha256=str(value["content_sha256"]),
                transform_version=str(value["transform_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed card provenance", "invalid_card_provenance") from error

    def to_dict(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "git_sha": self.git_sha,
            "path": self.path,
            "license": self.license,
            "content_sha256": self.content_sha256,
            "transform_version": self.transform_version,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeCard:
    """One immutable advisory claim with explicit scope and provenance."""

    card_id: str
    kind: CardKind
    status: CardStatus
    scope: KnowledgeScope
    claim: str
    apply: str
    verify: str
    caution: str
    source: SourceProvenance
    evidence_receipts: tuple[str, ...] = ()
    executable: bool = False
    supersedes: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.card_id, field_name="card_id")
        for name in ("claim", "apply", "verify", "caution"):
            text = getattr(self, name)
            if not text.strip() or len(text) > _MAX_TEXT:
                raise ContractError(f"Invalid card {name}", "invalid_card_text")
        if self.executable:
            raise ContractError("Knowledge cards are never executable", "executable_card_forbidden")
        if self.status is CardStatus.VALIDATED and not self.evidence_receipts:
            raise ContractError("Validated cards require evidence", "missing_card_evidence")
        if any(
            not _SHA256.fullmatch(item.removeprefix("sha256:"))
            for item in self.evidence_receipts
        ):
            raise ContractError("Invalid card evidence receipt", "invalid_card_evidence")
        _validate_references(self)
        if self.card_id != derive_card_id(self.semantic_dict()):
            raise ContractError("card_id is not content-derived", "invalid_card_id")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KnowledgeCard":
        try:
            if int(value.get("schema_version", 1)) != 1:
                raise ValueError("unsupported schema")
            scope = KnowledgeScope.from_mapping(_mapping(value.get("scope", {}), "scope"))
            source = SourceProvenance.from_mapping(_mapping(value["source"], "source"))
            semantic = _semantic_from_mapping(value, scope, source)
            return cls(card_id=str(value.get("card_id") or derive_card_id(semantic)), **semantic)
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed knowledge card", "invalid_knowledge_card") from error

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind.value,
            "status": self.status.value,
            "scope": self.scope.to_dict(),
            "claim": self.claim,
            "apply": self.apply,
            "verify": self.verify,
            "caution": self.caution,
            "source": self.source.to_dict(),
            "evidence_receipts": list(self.evidence_receipts),
            "executable": self.executable,
            "supersedes": list(self.supersedes),
            "contradicts": list(self.contradicts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"card_id": self.card_id, **self.semantic_dict()}

    @property
    def content_hash(self) -> str:
        return sha256_json(self.to_dict())


def derive_card_id(semantic: Mapping[str, Any]) -> str:
    """Derive an ID from immutable content, excluding relationship targets."""

    identity = {
        "schema_version": semantic.get("schema_version", 1),
        "kind": _enum_value(semantic["kind"]),
        "scope": _dict_value(semantic["scope"]),
        "claim": semantic["claim"],
        "apply": semantic["apply"],
        "verify": semantic["verify"],
        "caution": semantic["caution"],
        "source": _dict_value(semantic["source"]),
    }
    return f"card-{sha256_json(identity)[:24]}"


def validate_catalog(cards: Iterable[KnowledgeCard]) -> tuple[KnowledgeCard, ...]:
    """Validate references, conflicts, and supersession cycles for a snapshot."""

    catalog = tuple(cards)
    by_id = {card.card_id: card for card in catalog}
    if len(by_id) != len(catalog):
        raise ContractError("Duplicate knowledge card id", "duplicate_card_id")
    for card in catalog:
        _validate_catalog_references(card, by_id)
    _validate_symmetric_conflicts(catalog, by_id)
    _validate_supersession_cycles(catalog, by_id)
    return tuple(sorted(catalog, key=lambda card: card.card_id))


def _semantic_from_mapping(
    value: Mapping[str, Any], scope: KnowledgeScope, source: SourceProvenance
) -> dict[str, Any]:
    return {
        "kind": CardKind(str(value["kind"])),
        "status": CardStatus(str(value["status"])),
        "scope": scope,
        "claim": str(value["claim"]),
        "apply": str(value["apply"]),
        "verify": str(value["verify"]),
        "caution": str(value["caution"]),
        "source": source,
        "evidence_receipts": tuple(str(item) for item in value.get("evidence_receipts", ())),
        "executable": bool(value.get("executable", False)),
        "supersedes": tuple(sorted(str(item) for item in value.get("supersedes", ()))),
        "contradicts": tuple(sorted(str(item) for item in value.get("contradicts", ()))),
    }


def _terms(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError("scope dimensions must be arrays")
    terms = tuple(sorted({str(item).strip().lower() for item in value}))
    if any(not item for item in terms):
        raise ValueError("scope dimensions cannot contain empty values")
    return terms


def _dimension_matches(card: tuple[str, ...], query: tuple[str, ...]) -> bool:
    return not card or bool(query) and bool(set(card).intersection(query))


def _dimension_overlaps(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return not left or not right or bool(set(left).intersection(right))


def _validate_source_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ContractError("Unsafe provenance path", "invalid_card_source_path")


def _validate_references(card: KnowledgeCard) -> None:
    references = (*card.supersedes, *card.contradicts)
    for reference in references:
        validate_identifier(reference, field_name="card_relationship")
    if len(set(references)) != len(references) or card.card_id in references:
        raise ContractError("Invalid card relationship", "invalid_card_relationship")
    if set(card.supersedes).intersection(card.contradicts):
        raise ContractError(
            "Card cannot supersede and contradict the same card", "card_relation_conflict"
        )


def _validate_catalog_references(
    card: KnowledgeCard, catalog: Mapping[str, KnowledgeCard]
) -> None:
    for target in (*card.supersedes, *card.contradicts):
        if target not in catalog:
            raise ContractError("Card relationship target is missing", "missing_card_relationship")
        if not card.scope.overlaps(catalog[target].scope):
            raise ContractError("Related cards have disjoint scopes", "disjoint_card_relationship")


def _validate_symmetric_conflicts(
    cards: tuple[KnowledgeCard, ...], catalog: Mapping[str, KnowledgeCard]
) -> None:
    for card in cards:
        for target in card.contradicts:
            if card.card_id not in catalog[target].contradicts:
                raise ContractError(
                    "Contradiction links must be symmetric", "asymmetric_card_conflict"
                )


def _validate_supersession_cycles(
    cards: tuple[KnowledgeCard, ...], catalog: Mapping[str, KnowledgeCard]
) -> None:
    def visit(card_id: str, stack: frozenset[str]) -> None:
        if card_id in stack:
            raise ContractError("Supersession cycle detected", "card_supersession_cycle")
        for target in catalog[card_id].supersedes:
            visit(target, stack | {card_id})

    for card in cards:
        visit(card.card_id, frozenset())


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping", "invalid_knowledge_card")
    return value


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _dict_value(value: object) -> object:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


__all__ = [
    "CardKind",
    "CardStatus",
    "KnowledgeCard",
    "KnowledgeScope",
    "SourceProvenance",
    "derive_card_id",
    "validate_catalog",
]
