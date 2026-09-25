"""Exact-value redaction for backend credentials before result construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import quote


_REPLACEMENT = "[REDACTED_BACKEND_CREDENTIAL]"


@dataclass(frozen=True, slots=True)
class RedactedText:
    text: str
    replacements: int


def redact_secret_values(text: str, values: tuple[str, ...]) -> RedactedText:
    """Replace literal, JSON-escaped, and URL-encoded forms of exact secrets."""

    result = text
    count = 0
    variants = {
        variant
        for value in values
        if value
        for variant in _variants(value)
        if variant
    }
    for variant in sorted(variants, key=len, reverse=True):
        occurrences = result.count(variant)
        if occurrences:
            result = result.replace(variant, _REPLACEMENT)
            count += occurrences
    return RedactedText(result, count)


def _variants(value: str) -> tuple[str, ...]:
    return (
        value,
        json.dumps(value, ensure_ascii=True)[1:-1],
        json.dumps(value, ensure_ascii=False)[1:-1],
        quote(value, safe=""),
    )


__all__ = ["RedactedText", "redact_secret_values"]
