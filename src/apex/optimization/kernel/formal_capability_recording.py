"""Canonical typed MCP call/result evidence for formal kernel campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from apex.core import ApexError, canonical_json_bytes, sha256_json
from apex.orchestration import RunPhase
from apex.storage import ArtifactReceipt

from .run_record import KernelRunRecord


@dataclass(frozen=True, slots=True)
class FormalCapabilityInvocation:
    capability_id: str
    call_id: str
    arguments_digest: str
    attempt_id: str | None
    arguments: ArtifactReceipt


def begin_formal_capability(
    record: KernelRunRecord,
    capability_id: str,
    arguments: Mapping[str, object],
    *,
    grant_receipt: Mapping[str, object] | None = None,
) -> FormalCapabilityInvocation:
    """Seal typed arguments and append one retry-aware logical call event."""

    document = {
        "schema": "apex.formal-capability-arguments/v2",
        "run_id": record.run_id,
        "capability_id": capability_id,
        "arguments": dict(arguments),
        "capability_grant": (
            dict(grant_receipt) if grant_receipt is not None else None
        ),
    }
    digest = sha256_json(document)
    call_id = _call_id(record, capability_id, digest)
    attempt_id = _attempt_id(arguments)
    receipt = record.artifacts.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )
    record.controller.record_domain_event(
        "tool_called",
        {
            **_common(record, attempt_id),
            "tool_name": capability_id,
            "call_id": call_id,
            "arguments_digest": digest,
            "evidence_class": "diagnostic",
            "reward_eligible": False,
            "artifacts": [_binding("formal_capability_arguments", receipt)],
        },
        idempotency_key=f"formal_capability.{call_id}.called",
    )
    return FormalCapabilityInvocation(
        capability_id, call_id, digest, attempt_id, receipt
    )


def complete_formal_capability(
    record: KernelRunRecord,
    invocation: FormalCapabilityInvocation,
    content: Mapping[str, object],
) -> ArtifactReceipt:
    return _record_result(record, invocation, True, dict(content), None)


def fail_formal_capability(
    record: KernelRunRecord,
    invocation: FormalCapabilityInvocation,
    error: Exception,
) -> ArtifactReceipt:
    failure = {
        "reason_code": (
            error.reason_code
            if isinstance(error, ApexError)
            else "formal_capability_failed"
        ),
        "error_type": type(error).__name__,
    }
    return _record_result(record, invocation, False, None, failure)


def _record_result(record, invocation, succeeded, content, error):
    document = {
        "schema": "apex.formal-capability-result/v1",
        "run_id": record.run_id,
        "capability_id": invocation.capability_id,
        "call_id": invocation.call_id,
        "succeeded": succeeded,
        "content": content,
        "error": error,
    }
    receipt = record.artifacts.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )
    record.controller.record_domain_event(
        "tool_result",
        {
            **_common(record, invocation.attempt_id),
            "tool_name": invocation.capability_id,
            "call_id": invocation.call_id,
            "arguments_digest": invocation.arguments_digest,
            "succeeded": succeeded,
            "evidence_class": "diagnostic",
            "reward_eligible": False,
            "artifacts": [_binding("formal_capability_result", receipt)],
        },
        idempotency_key=f"formal_capability.{invocation.call_id}.result",
    )
    return receipt


def _call_id(record, capability_id: str, digest: str) -> str:
    calls = [
        event
        for event in record.iter_events()
        if event.event_type == "tool_called"
        and event.payload.get("tool_name") == capability_id
        and event.payload.get("arguments_digest") == digest
    ]
    results = {
        event.payload.get("call_id")
        for event in record.iter_events()
        if event.event_type == "tool_result"
    }
    open_calls = [event for event in calls if event.payload.get("call_id") not in results]
    if open_calls:
        return str(open_calls[-1].payload["call_id"])
    if record.controller.state.phase is not RunPhase.RUNNING and calls:
        return str(calls[-1].payload["call_id"])
    return f"mcp-{digest[:16]}-{len(calls) + 1}"


def _attempt_id(arguments: Mapping[str, object]) -> str | None:
    value = arguments.get("attempt_id")
    return value if isinstance(value, str) and value else None


def _common(record, attempt_id: str | None) -> dict[str, object]:
    return record.attempt_payload(attempt_id) if attempt_id is not None else {}


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = [
    "FormalCapabilityInvocation",
    "begin_formal_capability",
    "complete_formal_capability",
    "fail_formal_capability",
]
