"""Shared canonical recording for backend-neutral agent observations."""

from __future__ import annotations

from typing import Mapping

from apex.orchestration import RunController
from apex.ports import AgentResult, AgentSemanticEvent
from apex.storage import ArtifactReceipt


def record_agent_observations(
    controller: RunController,
    *,
    result: AgentResult,
    common_payload: Mapping[str, object],
    transcript: ArtifactReceipt,
    idempotency_prefix: str,
) -> None:
    """Record semantic actions and explicit usage/cost against one transcript."""

    common = {
        **common_payload,
        "backend": result.backend.value,
        "model": result.model,
        "effort": result.effort,
    }
    binding = [_artifact_binding("agent_transcript", transcript)]
    for event in result.semantic_events:
        controller.record_domain_event(
            event.kind,
            {
                **common,
                **_semantic_payload(event),
                "artifacts": binding,
            },
            idempotency_key=f"{idempotency_prefix}.agent_event.{event.index}",
        )
    if result.usage is not None:
        controller.record_domain_event(
            "usage_recorded",
            {
                **common,
                "evidence_class": "self_reported",
                **result.usage.to_dict(),
                "artifacts": binding,
            },
            idempotency_key=f"{idempotency_prefix}.usage",
        )
    if result.cost is not None:
        controller.record_domain_event(
            "cost_recorded",
            {
                **common,
                "evidence_class": "self_reported",
                **result.cost.to_dict(),
                "artifacts": binding,
            },
            idempotency_key=f"{idempotency_prefix}.cost",
        )


def _semantic_payload(event: AgentSemanticEvent) -> dict[str, object]:
    payload: dict[str, object] = {
        "semantic_index": event.index,
        "source_event_index": event.source_event_index,
        "source_kind": event.source_kind,
        "evidence_class": "self_reported",
    }
    if event.kind == "agent_message":
        payload.update(
            {
                "role": event.role or "assistant",
                "has_text": event.text is not None,
                "text_length": len(event.text) if event.text is not None else 0,
            }
        )
    else:
        payload.update(
            {
                "tool_name": event.tool_name,
                "call_id": event.tool_call_id,
                "succeeded": event.succeeded,
            }
        )
    return payload


def _artifact_binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = ["record_agent_observations"]
