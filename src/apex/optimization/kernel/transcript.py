"""Canonical agent transcript serialization and structured metadata extraction."""

from __future__ import annotations

from apex.ports import AgentResult


def transcript_document(result: AgentResult) -> dict[str, object]:
    """Return every normalized event without reconstructing data from human text."""

    return {
        "schema": "apex.agent-transcript/v1",
        "backend": result.backend.value,
        "model": result.model,
        "effort": result.effort,
        "invocation": result.invocation.to_dict() if result.invocation else None,
        "budget": {
            "exceeded": result.budget_exceeded,
            "enforcement_failed": result.budget_enforcement_failed,
            "reason": result.budget_reason,
            "observed_turns": result.observed_turns,
            "max_turns": result.invocation.max_turns if result.invocation else None,
            "turn_policy": result.invocation.turn_policy if result.invocation else None,
        },
        "events": [
            {
                "kind": event.kind,
                "text": event.text,
                "metadata": dict(event.metadata),
            }
            for event in result.events
        ],
        "semantic_events": [event.to_dict() for event in result.semantic_events],
        "usage": result.usage.to_dict() if result.usage is not None else None,
        "cost": result.cost.to_dict() if result.cost is not None else None,
    }


def transcript_metadata(result: AgentResult) -> dict[str, object]:
    """Project only explicit machine fields useful for RL reconstruction."""

    semantic = tuple(result.semantic_events)
    usage = result.usage
    payload: dict[str, object] = {
        "effort": result.effort,
        "budget_exceeded": result.budget_exceeded,
        "budget_enforcement_failed": result.budget_enforcement_failed,
        "budget_reason": result.budget_reason,
        "observed_turns": result.observed_turns,
        "invocation": result.invocation.to_dict() if result.invocation else None,
        "transcript_event_count": len(result.events),
        "semantic_event_count": len(semantic),
        "message_event_count": sum(item.kind == "agent_message" for item in semantic),
        "tool_call_event_count": sum(item.kind == "tool_called" for item in semantic),
        "tool_result_event_count": sum(item.kind == "tool_result" for item in semantic),
        "turn_count": usage.turn_count if usage is not None else 0,
        "tool_call_count": (
            usage.tool_call_count
            if usage is not None
            else sum(item.kind == "tool_called" for item in semantic)
        ),
        "transcript_kinds": [event.kind for event in result.events],
    }
    if usage is not None:
        payload["usage"] = usage.to_dict()
    if result.cost is not None:
        payload["cost"] = result.cost.to_dict()
    return payload


__all__ = ["transcript_document", "transcript_metadata"]
