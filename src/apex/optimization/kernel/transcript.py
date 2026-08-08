"""Structured metadata extraction for the canonical agent aggregate event."""

from __future__ import annotations

from apex.ports import AgentResult


def transcript_metadata(result: AgentResult) -> dict[str, object]:
    """Project only explicit machine fields useful for RL reconstruction."""

    semantic = tuple(result.semantic_events)
    usage = result.usage
    payload: dict[str, object] = {
        "effort": result.effort,
        "termination_kind": result.termination_kind.value,
        "termination_reason": result.termination_reason,
        "capture_status": result.capture_status.value,
        "candidate_capture_allowed": result.candidate_capture_allowed,
        "observer_stop_sent": result.observer_stop_sent,
        "process_containment_policy_id": (
            result.invocation.process_containment_policy_id
            if result.invocation
            else None
        ),
        "process_containment": (
            result.process_containment.to_dict()
            if result.process_containment is not None
            else None
        ),
        "discarded_stdout_lines": result.discarded_stdout_lines,
        "discarded_stdout_bytes": result.discarded_stdout_bytes,
        "discarded_stdout_sha256": result.discarded_stdout_sha256,
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


__all__ = ["transcript_metadata"]
