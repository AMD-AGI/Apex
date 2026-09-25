"""Pure reducer for run lifecycle, actions, and evidence events."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Protocol

from apex.core import StateTransitionError, validate_identifier

from .e2e_transitions import E2E_EVENT_HANDLERS
from .state import ActionState, ActionStatus, RunPhase, WorkloadState


DOMAIN_EVENT_TYPES = frozenset(
    {
        "agent_completed",
        "agent_failed",
        "agent_message",
        "candidate_frozen",
        "compile_result",
        "context_packet_created",
        "correctness_result",
        "cost_recorded",
        "decision",
        "delivery_result",
        "dependency_verified",
        "knowledge_outcome_linked",
        "knowledge_read",
        "measurement_result",
        "performance_command_result",
        "prompt_sent",
        "provenance_observed",
        "reward_committed",
        "safety_result",
        "tool_called",
        "tool_result",
        "usage_recorded",
        "experience.deferred",
        "experience.measured",
    }
)


class EventLike(Protocol):
    sequence: int
    event_id: str
    run_id: str
    event_type: str
    payload: Mapping[str, Any]
    parent_event_id: str | None


def reduce_event(state: WorkloadState, event: EventLike) -> WorkloadState:
    """Return the next immutable state or reject an illegal transition."""

    _validate_event_envelope(state, event)
    handlers = {
        "run.started": _run_started,
        "action.queued": _action_queued,
        "action.started": _action_started,
        "action.artifacts_ready": _artifacts_ready,
        "action.verified": _action_verified,
        "action.completed": _action_completed,
        "action.committed": _action_committed,
        "action.failed": _action_failed,
        "action.aborted": _action_aborted,
        "run.succeeded": _run_terminal,
        "run.failed": _run_terminal,
        "run.cancelled": _run_terminal,
        **E2E_EVENT_HANDLERS,
        **{event_type: _domain_event for event_type in DOMAIN_EVENT_TYPES},
    }
    handler = handlers.get(event.event_type)
    if handler is None:
        _reject("Unknown workload event", "unknown_event_type")
    successor = handler(state, event.payload, event.event_type)
    return replace(successor, sequence=event.sequence, last_event_id=event.event_id)


def _domain_event(
    state: WorkloadState,
    _payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    _require_running(state)
    return state


def _validate_event_envelope(state: WorkloadState, event: EventLike) -> None:
    if event.run_id != state.run_id:
        _reject("Event belongs to another run", "event_run_mismatch")
    if event.sequence <= state.sequence:
        _reject("Event sequence did not advance", "event_sequence_stale")
    if event.parent_event_id != state.last_event_id:
        _reject("Event does not extend the current journal head", "event_parent_stale")
    if state.phase in {RunPhase.SUCCEEDED, RunPhase.FAILED, RunPhase.CANCELLED}:
        _reject("Terminal workload cannot transition", "run_is_terminal")


def _run_started(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    if state.phase is not RunPhase.NEW:
        _reject("Run has already started", "run_already_started")
    anchor = _required_string(payload, "initial_anchor_id")
    validate_identifier(anchor, field_name="initial_anchor_id")
    return replace(state, phase=RunPhase.RUNNING, anchor_id=anchor)


def _action_queued(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    _require_running(state)
    if state.pending_action is not None:
        _reject("Another action is pending", "action_already_pending")
    action_id = _required_string(payload, "action_id")
    action_type = _required_string(payload, "action_type")
    parent_anchor = _required_string(payload, "parent_anchor_id")
    generation = _required_int(payload, "parent_anchor_generation")
    validate_identifier(action_id, field_name="action_id")
    validate_identifier(action_type, field_name="action_type")
    if parent_anchor != state.anchor_id or generation != state.anchor_generation:
        _reject("Action was based on a stale anchor", "stale_anchor")
    action = ActionState(
        action_id,
        action_type,
        ActionStatus.QUEUED,
        parent_anchor,
        generation,
    )
    return replace(state, pending_action=action)


def _action_started(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    action = _pending(state, payload, {ActionStatus.QUEUED})
    return replace(state, pending_action=replace(action, status=ActionStatus.STARTED))


def _artifacts_ready(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    action = _pending(state, payload, {ActionStatus.STARTED})
    raw_refs = payload.get("artifact_refs")
    if not isinstance(raw_refs, (list, tuple)) or not raw_refs:
        _reject("At least one artifact receipt is required", "artifacts_missing")
    return replace(
        state,
        pending_action=replace(
            action,
            status=ActionStatus.ARTIFACTS_READY,
            artifact_refs=tuple(str(item) for item in raw_refs),
        ),
    )


def _action_verified(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    action = _pending(state, payload, {ActionStatus.ARTIFACTS_READY})
    return replace(
        state,
        pending_action=replace(
            action,
            status=ActionStatus.VERIFIED,
            verification_id=_required_string(payload, "verification_id"),
        ),
    )


def _action_completed(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    action = _pending(state, payload, {ActionStatus.VERIFIED})
    completed = replace(action, status=ActionStatus.COMMITTED)
    return replace(
        state,
        pending_action=None,
        action_history=(*state.action_history, completed),
    )


def _action_committed(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    action = _pending(state, payload, {ActionStatus.VERIFIED})
    anchor = _required_string(payload, "new_anchor_id")
    patch = _required_string(payload, "accepted_patch_id")
    validate_identifier(anchor, field_name="new_anchor_id")
    validate_identifier(patch, field_name="accepted_patch_id")
    completed = replace(
        action,
        status=ActionStatus.COMMITTED,
        result_anchor_id=anchor,
        accepted_patch_id=patch,
    )
    return replace(
        state,
        anchor_id=anchor,
        anchor_generation=state.anchor_generation + 1,
        accepted_patch_ids=(*state.accepted_patch_ids, patch),
        pending_action=None,
        action_history=(*state.action_history, completed),
    )


def _action_failed(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    action = _pending(state, payload, _ACTIVE_STATUSES)
    completed = replace(
        action,
        status=ActionStatus.FAILED,
        error=_required_string(payload, "error"),
    )
    return replace(
        state,
        pending_action=None,
        action_history=(*state.action_history, completed),
    )


def _action_aborted(
    state: WorkloadState,
    payload: Mapping[str, Any],
    _event_type: str,
) -> WorkloadState:
    action = _pending(state, payload, _ACTIVE_STATUSES)
    completed = replace(
        action,
        status=ActionStatus.ABORTED,
        error=_required_string(payload, "reason"),
    )
    return replace(
        state,
        pending_action=None,
        action_history=(*state.action_history, completed),
    )


def _run_terminal(
    state: WorkloadState,
    payload: Mapping[str, Any],
    event_type: str,
) -> WorkloadState:
    _require_running(state)
    if state.pending_action is not None:
        _reject("Pending action must be resolved before stopping", "pending_action_at_stop")
    if event_type == "run.succeeded" and state.e2e is not None and state.accepted_patch_ids:
        if not state.e2e.final_clean_replay_verified:
            _reject(
                "Accepted E2E patches require a second clean replay before success",
                "second_clean_replay_required",
            )
    phases = {
        "run.succeeded": RunPhase.SUCCEEDED,
        "run.failed": RunPhase.FAILED,
        "run.cancelled": RunPhase.CANCELLED,
    }
    return replace(
        state,
        phase=phases[event_type],
        stop_reason=_required_string(payload, "reason"),
    )


def _pending(
    state: WorkloadState,
    payload: Mapping[str, Any],
    allowed: set[ActionStatus],
) -> ActionState:
    _require_running(state)
    action = state.pending_action
    if action is None:
        _reject("No action is pending", "action_not_pending")
    if action.action_id != _required_string(payload, "action_id"):
        _reject("Event targets another action", "action_id_mismatch")
    if action.status not in allowed:
        _reject("Action transition is illegal", "illegal_action_transition")
    return action


def _require_running(state: WorkloadState) -> None:
    if state.phase is not RunPhase.RUNNING:
        _reject("Run is not active", "run_not_active")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        _reject(f"Event field {key!r} is required", "event_field_missing")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _reject(f"Event field {key!r} must be an integer", "event_field_invalid")
    return value


def _reject(message: str, reason_code: str) -> None:
    raise StateTransitionError(message, reason_code=reason_code)


_ACTIVE_STATUSES = {
    ActionStatus.QUEUED,
    ActionStatus.STARTED,
    ActionStatus.ARTIFACTS_READY,
    ActionStatus.VERIFIED,
}


__all__ = ["DOMAIN_EVENT_TYPES", "EventLike", "reduce_event"]
