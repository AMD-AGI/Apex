"""Deterministic private-host-path policy for public RL projections."""

from __future__ import annotations

import base64
import re
from typing import Any, Mapping, Sequence

from apex.storage import ArtifactReceipt


HOST_PATH_POLICY = "host_absolute_path_redaction_v1"
_HOST_PATH_SENTINEL = "[REDACTED_PATH]"
_HOST_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:"
    r"/(?:home|root|tmp|var/tmp|Users|workspace|workspaces|mnt)/[^\s\"'`]+"
    r"|[A-Za-z]:\\(?:Users|Temp)\\[^\s\"'`]+)"
)


def encode_public_artifact(
    content: bytes, receipt: ArtifactReceipt
) -> tuple[str, str, str | None]:
    """Encode artifact bytes and declare any deterministic path rewrite."""

    textual = receipt.media_type.startswith("text/") or receipt.media_type in {
        "application/json",
        "application/x-ndjson",
        "application/yaml",
    }
    if textual:
        try:
            original = content.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            redacted = redact_host_path_text(original)
            policy = HOST_PATH_POLICY if redacted != original else None
            return "utf-8", redacted, policy
    return "base64", base64.b64encode(content).decode("ascii"), None


def redact_host_paths(value: Any) -> Any:
    """Recursively replace only private host absolute paths."""

    if isinstance(value, Mapping):
        return {key: redact_host_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_host_paths(item) for item in value]
    if isinstance(value, str):
        return redact_host_path_text(value)
    return value


def summarize_export(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the manifest summary that the strict consumer recomputes."""

    artifacts = tuple(
        artifact
        for record in records
        for values in record.get("artifacts_by_role", {}).values()
        for artifact in values
    )
    return {
        "record_count": len(records),
        "split_counts": _value_counts(records, "split"),
        "visibility_counts": _value_counts(records, "visibility"),
        "artifact_count": len(artifacts),
        "redacted_artifact_count": sum(
            item.get("redaction_policy_id") == HOST_PATH_POLICY
            for item in artifacts
        ),
    }


def redact_host_path_text(value: str) -> str:
    return _HOST_PATH.sub(_HOST_PATH_SENTINEL, value)


def _value_counts(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record[field])
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


__all__ = [
    "HOST_PATH_POLICY",
    "encode_public_artifact",
    "redact_host_paths",
    "summarize_export",
]
