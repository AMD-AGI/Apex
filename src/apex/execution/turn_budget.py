"""Streaming, provider-neutral agent turn-budget enforcement."""

from __future__ import annotations

import json
from collections.abc import Mapping

from apex.ports import AgentTerminationKind, STRUCTURED_TURN_CHECKPOINT_POLICY


TURN_POLICY = STRUCTURED_TURN_CHECKPOINT_POLICY


class AgentTurnBudget:
    """Count complete structured decisions and stop before another model turn."""

    def __init__(self, max_turns: int) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self.observed_turns = 0
        self.termination_kind: AgentTerminationKind | None = None
        self.termination_reason: str | None = None
        self._saw_decision_event = False
        self._saw_turn_evidence = False

    def observe(self, line: str) -> bool:
        """Return true when the process must be terminated for budget exhaustion."""

        try:
            value = _json_object(line)
        except json.JSONDecodeError:
            self._stop(AgentTerminationKind.INVALID_STREAM, "unparseable_structured_event")
            return True
        if value is None:
            return False
        before = self.observed_turns
        explicit = _nonnegative_int(value, "num_turns", "turn_count", "turns")
        decisions = _decision_count(value)
        if decisions:
            self._saw_decision_event = True
            self._saw_turn_evidence = True
        event_type = str(value.get("type", "")).lower().replace("_", ".")
        fallback = int(
            event_type == "turn.completed"
            and not self._saw_decision_event
            and explicit is None
        )
        if fallback:
            self._saw_turn_evidence = True
        inferred = before + decisions + fallback
        self.observed_turns = max(inferred, explicit or 0)
        if explicit is not None:
            self._saw_turn_evidence = True
        if self.observed_turns > self.max_turns:
            self._stop(AgentTerminationKind.TURN_OVERRUN, "max_turns_overrun")
            return True
        if self.observed_turns == self.max_turns:
            self._stop(
                AgentTerminationKind.EXACT_TURN_BOUNDARY,
                "max_turns_exact_boundary",
            )
            return True
        return False

    def finalize(self, *, process_succeeded: bool, observer_stopped: bool) -> None:
        """Reject a nominal success when no structured turn evidence was observable."""

        if observer_stopped and self.termination_kind is None:
            self._stop(AgentTerminationKind.INVALID_STREAM, "turn_observer_failed")
        if process_succeeded and not self._saw_turn_evidence:
            self._stop(
                AgentTerminationKind.INVALID_STREAM,
                "missing_structured_turn_evidence",
            )

    def _stop(self, kind: AgentTerminationKind, reason: str) -> None:
        if self.termination_kind is None:
            self.termination_kind = kind
            self.termination_reason = reason


def _json_object(line: str) -> Mapping[str, object] | None:
    if not line.lstrip().startswith("{"):
        return None
    value = json.loads(line)
    return value if isinstance(value, Mapping) else None


def _decision_count(value: Mapping[str, object]) -> int:
    event_type = str(value.get("type", "")).lower().replace(".", "_")
    message = value.get("message")
    if event_type == "assistant" and isinstance(message, Mapping):
        content = message.get("content")
        return 1
    if event_type in {"assistant_message", "agent_message"}:
        return 1
    if event_type == "item_completed":
        item = value.get("item")
        if isinstance(item, Mapping) and str(item.get("type", "")).lower() in {
            "agent_message",
            "assistant_message",
        }:
            return 1
    if _standalone_tool_request(event_type, value):
        return 1
    return 0


def _standalone_tool_request(event_type: str, value: Mapping[str, object]) -> bool:
    if event_type in {"tool_call", "tool_called", "tool_use"}:
        subtype = str(value.get("subtype", value.get("status", "started"))).lower()
        return subtype not in {"completed", "result", "failed", "error", "cancelled"}
    if event_type in {"item_started", "item_completed"}:
        item = value.get("item")
        if not isinstance(item, Mapping):
            return False
        item_type = str(item.get("type", "")).lower()
        return event_type == "item_started" and any(
            marker in item_type for marker in ("tool", "command_execution")
        )
    return False


def _nonnegative_int(value: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return candidate
    return None


__all__ = ["AgentTurnBudget", "TURN_POLICY"]
