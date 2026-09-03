"""Read-only scoped knowledge retrieval boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class KnowledgeQuery:
    """Scope-first query formed only after an independent live diagnosis."""

    operator: str | None
    gpu_arch: str
    language: str
    independent_hypothesis: str
    framework: str | None = None
    regime: str | None = None
    dtype: str | None = None
    software_version: str | None = None
    limit: int = 4
    max_tokens: int = 1_600


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    card_ids: Sequence[str]
    cards: Sequence[Mapping[str, object]]
    selection_policy: str
    unavailable_reason: str | None = None


class KnowledgePort(Protocol):
    def retrieve(self, query: KnowledgeQuery) -> KnowledgeResult: ...
