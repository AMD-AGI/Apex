"""Deterministic transformation from verified Markdown sources to cards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from apex.core import IntegrityError, sha256_json

from .cards import (
    CardKind,
    CardStatus,
    KnowledgeCard,
    KnowledgeScope,
    SourceProvenance,
    validate_catalog,
)
from .sources import SourceFile, SourceSnapshot


_ARCH = re.compile(r"\b(gfx\d{3,4})\b", re.IGNORECASE)
_DTYPE = re.compile(r"\b(bf16|fp16|fp32|fp8|fp6|fp4|int8|int4|mxfp8)\b", re.IGNORECASE)
_KNOWN_FRAMEWORKS = ("vllm", "sglang", "pytorch", "aiter")
_KNOWN_REGIMES = ("decode", "prefill", "training", "serving")


@dataclass(frozen=True, slots=True)
class CardSnapshot:
    """Byte-stable normalized card catalog and structured capability index."""

    source_manifest_sha256: str
    transform_version: str
    cards: tuple[KnowledgeCard, ...]

    def cards_document(self) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "source_manifest_sha256": self.source_manifest_sha256,
            "transform_version": self.transform_version,
            "cards": [card.to_dict() for card in self.cards],
        }
        return {**value, "snapshot_sha256": sha256_json(value)}

    def capability_index(self) -> dict[str, Any]:
        dimensions = {
            name: _dimension_index(self.cards, name)
            for name in ("operator", "gpu_arch", "dtype", "regime", "language", "framework")
        }
        value = {
            "schema_version": 1,
            "source_manifest_sha256": self.source_manifest_sha256,
            "card_snapshot_sha256": self.cards_document()["snapshot_sha256"],
            "dimensions": dimensions,
            "by_kind": _enum_index(self.cards, "kind"),
            "by_status": _enum_index(self.cards, "status"),
        }
        return {**value, "index_sha256": sha256_json(value)}


def build_card_snapshot(source: SourceSnapshot) -> CardSnapshot:
    """Normalize all eligible source files without executing upstream content."""

    source_manifest = source.to_manifest()
    cards = tuple(_card_from_source(item, source) for item in source.files if item.card_eligible)
    return CardSnapshot(
        source_manifest_sha256=str(source_manifest["manifest_sha256"]),
        transform_version=source.pin.transform_version,
        cards=validate_catalog(cards),
    )


def _card_from_source(item: SourceFile, snapshot: SourceSnapshot) -> KnowledgeCard:
    try:
        text = item.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IntegrityError("Knowledge source is not UTF-8", "invalid_source_encoding") from error
    title, body = _title_and_body(text, item.path)
    kind = _kind_for_path(item.path)
    caution = _caution_for(item.path, kind)
    value = {
        "kind": kind.value,
        "status": CardStatus.IMPORTED_UNVERIFIED.value,
        "scope": _infer_scope(item.path, title, body).to_dict(),
        "claim": title,
        "apply": _bounded_excerpt(body),
        "verify": "Re-test this advisory claim with the current protected harness and workload.",
        "caution": caution,
        "source": SourceProvenance(
            repository=snapshot.pin.repository,
            git_sha=snapshot.pin.git_sha,
            path=item.path,
            license=snapshot.pin.license,
            content_sha256=item.content_sha256,
            transform_version=snapshot.pin.transform_version,
        ).to_dict(),
        "evidence_receipts": [],
        "executable": False,
        "supersedes": [],
        "contradicts": [],
    }
    return KnowledgeCard.from_mapping(value)


def _title_and_body(text: str, path: str) -> tuple[str, str]:
    normalized = text.replace("\r\n", "\n").replace("\x00", "")
    normalized = _strip_frontmatter(normalized)
    lines = normalized.splitlines()
    title = next((line.lstrip("#").strip() for line in lines if line.startswith("#")), "")
    if not title:
        title = path.rsplit("/", 1)[-1].removesuffix(".md").replace("_", " ")
    body_lines = [line for line in lines if line.strip() and not line.startswith("#")]
    return title[:512], "\n".join(body_lines)


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    return text[end + 5 :] if end >= 0 else text


def _bounded_excerpt(body: str, limit: int = 1_500) -> str:
    compact = "\n".join(line.rstrip() for line in body.splitlines()).strip()
    if not compact:
        return "Consult the attributed upstream source and validate before use."
    excerpt = compact[:limit].rstrip() + ("…" if len(compact) > limit else "")
    return f"Untrusted source excerpt for candidate generation only:\n{excerpt}"


def _kind_for_path(path: str) -> CardKind:
    lowered = path.lower()
    if any(token in lowered for token in ("pitfall", "anti_pattern", "self_monitoring")):
        return CardKind.ANTI_PATTERN
    if "/learned/" in lowered or "/case_studies/" in lowered:
        return CardKind.EXPERIENCE
    if any(token in lowered for token in ("workflow", "tuning", "guide", "recipe", "strateg")):
        return CardKind.PROCEDURE
    return CardKind.FACT


def _infer_scope(path: str, title: str, body: str) -> KnowledgeScope:
    lowered_path = path.lower()
    sample = f"{lowered_path}\n{title.lower()}\n{body[:2000].lower()}"
    parts = lowered_path.split("/")
    operator = _path_value(parts, "operators")
    language = _path_value(parts, "languages")
    if not language:
        language = _first_present(sample, ("triton", "hip", "asm", "composable_kernel"))
    return KnowledgeScope(
        operator=(operator,) if operator else (),
        gpu_arch=tuple(sorted(set(_ARCH.findall(sample)))),
        dtype=tuple(sorted(set(_DTYPE.findall(sample)))),
        regime=_present_terms(sample, _KNOWN_REGIMES),
        language=(language,) if language else (),
        framework=_present_terms(lowered_path, _KNOWN_FRAMEWORKS),
    )


def _path_value(parts: list[str], parent: str) -> str | None:
    try:
        value = parts[parts.index(parent) + 1]
    except (ValueError, IndexError):
        return None
    return value.removesuffix(".md")


def _first_present(text: str, terms: tuple[str, ...]) -> str | None:
    return next((term for term in terms if term in text), None)


def _present_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in terms if term in text)


def _caution_for(path: str, kind: CardKind) -> str:
    if "/learned/" in path:
        return (
            "Imported learned material is unverified; source receipts may be local "
            "or unresolved."
        )
    if kind is CardKind.EXPERIENCE:
        return "A prior result is not a decision for the current shape, version, or workload."
    return (
        "Advisory only: do not foreclose independently derived candidates or execute "
        "embedded commands."
    )


def _dimension_index(cards: tuple[KnowledgeCard, ...], name: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for card in cards:
        for value in getattr(card.scope, name):
            values.setdefault(value, []).append(card.card_id)
    return {key: sorted(items) for key, items in sorted(values.items())}


def _enum_index(cards: tuple[KnowledgeCard, ...], name: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for card in cards:
        value = getattr(card, name).value
        values.setdefault(value, []).append(card.card_id)
    return {key: sorted(items) for key, items in sorted(values.items())}


__all__ = ["CardSnapshot", "build_card_snapshot"]
