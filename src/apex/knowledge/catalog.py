"""Strict loading of deterministic generated knowledge-card catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_file, sha256_json

from .cards import KnowledgeCard, validate_catalog


@dataclass(frozen=True, slots=True)
class KnowledgeCatalog:
    """Verified generated cards and the source-manifest identity they retain."""

    path: Path
    source_manifest_sha256: str
    snapshot_sha256: str
    file_sha256: str
    transform_version: str
    cards: tuple[KnowledgeCard, ...]


def load_knowledge_catalog(path: Path) -> KnowledgeCatalog:
    """Load canonical generated JSON and fail closed on schema/hash drift."""

    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ContractError("Knowledge catalog is not a regular file", "invalid_knowledge_catalog")
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IntegrityError("Knowledge catalog cannot be decoded", "invalid_knowledge_catalog") from error
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ContractError("Unsupported knowledge catalog schema", "unsupported_knowledge_catalog")
    raw_cards = document.get("cards")
    if not isinstance(raw_cards, list) or any(not isinstance(item, Mapping) for item in raw_cards):
        raise ContractError("Knowledge catalog cards must be objects", "invalid_knowledge_catalog")
    semantic = dict(document)
    declared_snapshot = str(semantic.pop("snapshot_sha256", ""))
    if declared_snapshot != sha256_json(semantic):
        raise IntegrityError("Knowledge snapshot digest mismatch", "knowledge_snapshot_mismatch")
    if source.read_bytes() != canonical_json_bytes(document) + b"\n":
        raise IntegrityError("Knowledge catalog is not canonical JSON", "noncanonical_knowledge_catalog")
    cards = validate_catalog(KnowledgeCard.from_mapping(item) for item in raw_cards)
    source_manifest = str(document.get("source_manifest_sha256", ""))
    if not _is_digest(source_manifest) or not str(document.get("transform_version", "")).strip():
        raise ContractError("Knowledge catalog provenance is incomplete", "invalid_knowledge_catalog")
    return KnowledgeCatalog(
        path=source,
        source_manifest_sha256=source_manifest,
        snapshot_sha256=declared_snapshot,
        file_sha256=sha256_file(source),
        transform_version=str(document["transform_version"]),
        cards=cards,
    )


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


__all__ = ["KnowledgeCatalog", "load_knowledge_catalog"]
