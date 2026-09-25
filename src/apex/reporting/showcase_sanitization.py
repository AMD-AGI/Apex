"""Shared deterministic secret and private-host-path handling for showcases."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError


_SECRET_TEXT = re.compile(
    r"(?:(?<![A-Za-z0-9])sk-(?:ant-)?[A-Za-z0-9_-]{20,}|"
    r"ghp_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,}|"
    r"Bearer\s+[A-Za-z0-9._~+/-]{12,}|https?://[^\s/@:]+:[^\s/@]+@)"
)
_HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:"
    r"/(?:home|root|tmp|var/tmp|Users|workspace|workspaces|mnt)/[^\s\"'`]+"
    r"|[A-Za-z]:\\(?:Users|Temp)\\[^\s\"'`]+)"
)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)$", re.I
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:export\s+)?(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)"
    r"\s*[:=]\s*(?P<value>[^\s,;#}]+)"
)
_SAFE_SENTINELS = frozenset({"", "empty", "[redacted]"})
HOST_PATH_REDACTION_POLICY = "host_path_redaction_v1"


def sanitize_projection_text(value: str) -> str:
    """Redact secret-like values and private host paths in derived projections."""

    return _HOST_PATH.sub("[REDACTED_PATH]", _SECRET_TEXT.sub("[REDACTED]", value))


def export_text_artifact(content: bytes, *, digest: str) -> tuple[bytes, str | None]:
    """Reject credentials and deterministically redact only private host paths."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(
            "Showcase text artifact is not UTF-8", "showcase_artifact_invalid"
        ) from error
    if _contains_secret(text):
        raise ContractError(
            "Showcase artifact contains a secret",
            "showcase_secret_detected",
            {"digest": digest},
        )
    redacted = _HOST_PATH.sub("[REDACTED_PATH]", text)
    policy = HOST_PATH_REDACTION_POLICY if redacted != text else None
    return redacted.encode("utf-8"), policy


def validate_exported_text_artifact(content: bytes, policy: object) -> None:
    """Independently reject leaked secrets/paths and unknown redaction policies."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IntegrityError(
            "Showcase text artifact is not UTF-8", "showcase_artifact_invalid"
        ) from error
    if policy not in {None, HOST_PATH_REDACTION_POLICY}:
        raise IntegrityError(
            "Showcase artifact redaction policy is invalid",
            "showcase_artifact_manifest_invalid",
        )
    if _contains_secret(text) or _HOST_PATH.search(text):
        raise IntegrityError(
            "Showcase artifact exposes a secret or private host path",
            "showcase_secret_detected",
        )


def projection_contains_private_text(value: str) -> bool:
    return bool(_contains_secret(value) or _HOST_PATH.search(value))


def _contains_secret(text: str) -> bool:
    if _SECRET_TEXT.search(text):
        return True
    for match in _SECRET_ASSIGNMENT.finditer(text):
        value = match.group("value").strip("\"'").casefold()
        if value not in _SAFE_SENTINELS and len(value) >= 4:
            return True
    candidate = text.strip()
    if len(candidate) < 2 or candidate[0] not in "[{":
        return False
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return _structured_secret(value)


def _structured_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (
                _SECRET_KEY.search(str(key)) is not None
                and not _safe_secret_value(item)
            )
            or _structured_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_structured_secret(item) for item in value)
    return False


def _safe_secret_value(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().casefold() in _SAFE_SENTINELS
    )


__all__ = [
    "HOST_PATH_REDACTION_POLICY",
    "export_text_artifact",
    "projection_contains_private_text",
    "sanitize_projection_text",
    "validate_exported_text_artifact",
]
