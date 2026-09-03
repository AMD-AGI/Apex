"""Structured agent stream parsing without natural-language metric inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from apex.core import ContractError
from apex.ports import (
    AgentCost,
    AgentResult,
    AgentSemanticEvent,
    AgentTranscriptEvent,
    AgentUsage,
)


@dataclass(frozen=True, slots=True)
class ParsedAgentOutput:
    events: tuple[AgentTranscriptEvent, ...]
    semantic_events: tuple[AgentSemanticEvent, ...]
    usage: AgentUsage | None
    cost: AgentCost | None


def parse_agent_output(text: str) -> ParsedAgentOutput:
    """Normalize JSONL objects; non-JSON text never contributes metrics."""

    events = _parse_jsonl_events(text)
    semantic_events = _semantic_events(events)
    return ParsedAgentOutput(
        events,
        semantic_events,
        _usage(events, semantic_events),
        _cost(events),
    )


def agent_transcript_document(result: AgentResult) -> dict[str, object]:
    """Return a deterministic, provider-neutral transcript document."""

    return {
        "schema": "apex.agent-transcript/v3",
        "backend": result.backend.value,
        "model": result.model,
        "effort": result.effort,
        "invocation": result.invocation.to_dict() if result.invocation else None,
        "termination": {
            "kind": result.termination_kind.value,
            "reason": result.termination_reason,
            "capture_status": result.capture_status.value,
            "candidate_capture_allowed": result.candidate_capture_allowed,
            "observer_stop_sent": result.observer_stop_sent,
            "process_containment": (
                result.process_containment.to_dict()
                if result.process_containment is not None
                else None
            ),
            "discarded_stdout_tail": {
                "lines": result.discarded_stdout_lines,
                "bytes": result.discarded_stdout_bytes,
                "sha256": result.discarded_stdout_sha256,
            },
            "credential_redaction_count": result.credential_redaction_count,
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
        "usage": result.usage.to_dict() if result.usage else None,
        "cost": result.cost.to_dict() if result.cost else None,
    }


def _parse_jsonl_events(text: str) -> tuple[AgentTranscriptEvent, ...]:
    events: list[AgentTranscriptEvent] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            events.append(AgentTranscriptEvent(kind="malformed_json", text=line[:1000]))
            continue
        if isinstance(value, dict):
            events.append(
                AgentTranscriptEvent(kind=str(value.get("type", "event")), metadata=value)
            )
    return tuple(events)


def _semantic_events(
    events: Sequence[AgentTranscriptEvent],
) -> tuple[AgentSemanticEvent, ...]:
    normalized: list[AgentSemanticEvent] = []
    for source_index, event in enumerate(events):
        for candidate in _semantic_candidates(event):
            normalized.append(
                AgentSemanticEvent(
                    index=len(normalized),
                    source_event_index=source_index,
                    source_kind=event.kind,
                    **candidate,
                )
            )
    return tuple(normalized)


def _semantic_candidates(event: AgentTranscriptEvent) -> list[dict[str, Any]]:
    metadata = event.metadata
    item = metadata.get("item")
    if isinstance(item, Mapping):
        return _item_candidates(event.kind, item)
    message = metadata.get("message")
    if isinstance(message, Mapping):
        return _message_candidates(event.kind, message)
    return _direct_candidates(event.kind, metadata)


def _item_candidates(source_kind: str, item: Mapping[str, object]) -> list[dict[str, Any]]:
    item_kind = str(item.get("type", "")).lower()
    if item_kind in {"agent_message", "assistant_message", "message"}:
        text = _text_content(item.get("text", item.get("content")))
        return [_message_candidate("assistant", text)] if text is not None else []
    if _is_tool_kind(item_kind):
        return [
            _tool_candidate(
                _tool_phase(source_kind, item),
                item,
                fallback_name=item_kind,
            )
        ]
    return []


def _message_candidates(
    source_kind: str, message: Mapping[str, object]
) -> list[dict[str, Any]]:
    role = str(message.get("role") or source_kind).lower()
    content = message.get("content")
    if isinstance(content, str):
        return [_message_candidate(role, content)] if role in {"assistant", "agent"} else []
    if not isinstance(content, list):
        return []
    candidates: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        block_kind = str(block.get("type", "")).lower()
        if block_kind in {"text", "agent_message", "assistant_message"}:
            text = _text_content(block.get("text", block.get("content")))
            if text is not None and role in {"assistant", "agent"}:
                candidates.append(_message_candidate(role, text))
        elif block_kind in {"tool_use", "tool_call", "tool_called"}:
            candidates.append(_tool_candidate("tool_called", block))
        elif block_kind in {"tool_result", "tool_response"}:
            candidates.append(_tool_candidate("tool_result", block))
    return candidates


def _direct_candidates(
    source_kind: str, metadata: Mapping[str, object]
) -> list[dict[str, Any]]:
    normalized = source_kind.lower().replace(".", "_")
    if normalized in {"agent_message", "assistant_message"}:
        text = _text_content(metadata.get("text", metadata.get("content")))
        return [_message_candidate("assistant", text)] if text is not None else []
    envelope = metadata.get("tool_call")
    if normalized == "tool_call" and isinstance(envelope, Mapping):
        return [_tool_envelope_candidate(source_kind, metadata, envelope)]
    if _is_tool_kind(normalized):
        return [_tool_candidate(_tool_phase(source_kind, metadata), metadata)]
    return []


def _message_candidate(role: str, text: str) -> dict[str, Any]:
    return {"kind": "agent_message", "role": role, "text": text}


def _tool_candidate(
    phase: str,
    value: Mapping[str, object],
    *,
    fallback_name: str = "tool",
) -> dict[str, Any]:
    name = value.get("name", value.get("tool_name", value.get("tool", fallback_name)))
    if value.get("server") and value.get("tool"):
        name = f"{value['server']}.{value['tool']}"
    call_id = value.get("id", value.get("tool_call_id", value.get("tool_use_id")))
    return {
        "kind": phase,
        "tool_name": str(name) if name else fallback_name,
        "tool_call_id": str(call_id) if call_id else None,
        "succeeded": _tool_succeeded(value) if phase == "tool_result" else None,
    }


def _tool_envelope_candidate(
    source_kind: str,
    metadata: Mapping[str, object],
    envelope: Mapping[str, object],
) -> dict[str, Any]:
    name = next((str(key) for key in envelope if str(key)), "tool")
    detail = envelope.get(name)
    values = dict(detail) if isinstance(detail, Mapping) else {}
    values.setdefault("name", name)
    for key in ("id", "tool_call_id", "status", "is_error", "success"):
        if key in metadata:
            values.setdefault(key, metadata[key])
    subtype = str(metadata.get("subtype", ""))
    return _tool_candidate(_tool_phase(f"{source_kind}.{subtype}", values), values)


def _tool_phase(source_kind: str, value: Mapping[str, object]) -> str:
    words = f"{source_kind} {value.get('status', '')}".lower()
    if any(word in words for word in ("completed", "result", "response", "failed", "error")):
        return "tool_result"
    return "tool_called"


def _tool_succeeded(value: Mapping[str, object]) -> bool | None:
    for key in ("succeeded", "success"):
        if isinstance(value.get(key), bool):
            return bool(value[key])
    if isinstance(value.get("is_error"), bool):
        return not bool(value["is_error"])
    exit_code = value.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return exit_code == 0
    status = str(value.get("status", "")).lower()
    if status in {"completed", "success", "succeeded"}:
        return True
    if status in {"failed", "error", "cancelled"}:
        return False
    return None


def _usage(
    events: Sequence[AgentTranscriptEvent],
    semantic_events: Sequence[AgentSemanticEvent],
) -> AgentUsage | None:
    direct, nested = _usage_snapshots(events)
    snapshots = direct or nested
    source_index, values = snapshots[-1] if snapshots else (-1, {})
    input_tokens = _first_int(values, ("input_tokens", "prompt_tokens", "inputTokens"))
    output_tokens = _first_int(values, ("output_tokens", "completion_tokens", "outputTokens"))
    cached = _cached_tokens(values)
    cache_creation = _first_int(values, ("cache_creation_input_tokens",))
    reasoning = _reasoning_tokens(values)
    total = _first_int(values, ("total_tokens", "totalTokens"))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    turns = _explicit_count(events, values, ("num_turns", "turn_count", "turns"))
    tools = _explicit_count(events, values, ("num_tool_calls", "tool_call_count", "tool_calls"))
    turn_sources = _turn_sources(events, semantic_events) if turns is None else ()
    tool_sources = tuple(
        event.source_event_index for event in semantic_events if event.kind == "tool_called"
    )
    turns = len(turn_sources) if turns is None else turns
    tools = len(tool_sources) if tools is None else tools
    tokens = (input_tokens, cached, cache_creation, output_tokens, reasoning, total)
    if not snapshots and not turns and not tools:
        return None
    sources = {source_index} if source_index >= 0 else set()
    sources.update(turn_sources)
    sources.update(tool_sources)
    return AgentUsage(*tokens, turns, tools, tuple(sorted(sources)))


def _usage_snapshots(
    events: Sequence[AgentTranscriptEvent],
) -> tuple[list[tuple[int, Mapping[str, object]]], list[tuple[int, Mapping[str, object]]]]:
    direct: list[tuple[int, Mapping[str, object]]] = []
    summaries: list[tuple[int, Mapping[str, object]]] = []
    nested: list[tuple[int, Mapping[str, object]]] = []
    for index, event in enumerate(events):
        usage = event.metadata.get("usage")
        if isinstance(usage, Mapping):
            direct.append((index, usage))
            if event.kind.lower() == "result":
                summaries.append((index, usage))
        message = event.metadata.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("usage"), Mapping):
            nested.append((index, message["usage"]))
    return summaries or direct, nested


def _cached_tokens(values: Mapping[str, object]) -> int | None:
    direct = _first_int(values, ("cached_input_tokens", "cache_read_input_tokens"))
    details = values.get("input_tokens_details")
    if direct is None and isinstance(details, Mapping):
        return _first_int(details, ("cached_tokens", "cache_read_input_tokens"))
    return direct


def _reasoning_tokens(values: Mapping[str, object]) -> int | None:
    direct = _first_int(values, ("reasoning_tokens", "reasoning_output_tokens"))
    details = values.get("output_tokens_details")
    if direct is None and isinstance(details, Mapping):
        return _first_int(details, ("reasoning_tokens",))
    return direct


def _explicit_count(
    events: Sequence[AgentTranscriptEvent],
    usage: Mapping[str, object],
    keys: tuple[str, ...],
) -> int | None:
    for event in reversed(events):
        value = _first_int(event.metadata, keys)
        if value is not None:
            return value
    return _first_int(usage, keys)


def _turn_sources(
    events: Sequence[AgentTranscriptEvent], semantic: Sequence[AgentSemanticEvent]
) -> tuple[int, ...]:
    completed = tuple(
        index
        for index, event in enumerate(events)
        if event.kind.lower().replace("_", ".") == "turn.completed"
    )
    if completed:
        return completed
    return tuple(
        sorted({event.source_event_index for event in semantic if event.kind == "agent_message"})
    )


def _cost(events: Sequence[AgentTranscriptEvent]) -> AgentCost | None:
    candidates: list[AgentCost] = []
    summaries: list[AgentCost] = []
    for index, event in enumerate(events):
        values = (event.metadata, event.metadata.get("usage"))
        for value in values:
            if isinstance(value, Mapping):
                candidate = _cost_from_mapping(value, index)
                if candidate is not None:
                    candidates.append(candidate)
                    if event.kind.lower() == "result":
                        summaries.append(candidate)
    selected = summaries or candidates
    return selected[-1] if selected else None


def _cost_from_mapping(value: Mapping[str, object], index: int) -> AgentCost | None:
    for key in ("total_cost_usd", "cost_usd"):
        if key in value:
            return _make_cost(value[key], "USD", index, key)
    cost = value.get("cost")
    if isinstance(cost, Mapping) and "amount" in cost:
        return _make_cost(
            cost["amount"], str(cost.get("currency", "USD")), index, "cost"
        )
    if cost is not None and not isinstance(cost, Mapping):
        return _make_cost(cost, str(value.get("currency", "USD")), index, "cost")
    return None


def _make_cost(value: object, currency: str, index: int, key: str) -> AgentCost | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        Decimal(str(value))
        return AgentCost(str(value), currency, index, key)
    except (InvalidOperation, ContractError):
        return None


def _first_int(value: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            return item
    return None


def _text_content(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _is_tool_kind(value: str) -> bool:
    normalized = value.lower().replace(".", "_")
    return any(
        marker in normalized
        for marker in ("tool_call", "tool_use", "tool_result", "command_execution")
    )


__all__ = ["ParsedAgentOutput", "agent_transcript_document", "parse_agent_output"]
