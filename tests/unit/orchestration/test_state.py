from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Mapping

import pytest

from apex.core import StateTransitionError
from apex.orchestration import ActionStatus, RunPhase, WorkloadState, reduce_event


@dataclass(frozen=True)
class Event:
    sequence: int
    event_id: str
    run_id: str
    event_type: str
    payload: Mapping[str, Any]
    parent_event_id: str | None


def advance(state: WorkloadState, event_type: str, payload: Mapping[str, Any]) -> WorkloadState:
    event = Event(
        state.sequence + 1,
        f"event-{state.sequence + 1}",
        state.run_id,
        event_type,
        payload,
        state.last_event_id,
    )
    return reduce_event(state, event)


def running_state() -> WorkloadState:
    return advance(
        WorkloadState.initial("run-1"),
        "run.started",
        {"initial_anchor_id": "base-0"},
    )


def test_reducer_runs_complete_action_lifecycle_without_mutating_prior_state() -> None:
    initial = running_state()
    queued = advance(
        initial,
        "action.queued",
        {
            "action_id": "action-1",
            "action_type": "kernel-optimize",
            "parent_anchor_id": "base-0",
            "parent_anchor_generation": 0,
        },
    )
    started = advance(queued, "action.started", {"action_id": "action-1"})
    ready = advance(
        started,
        "action.artifacts_ready",
        {"action_id": "action-1", "artifact_refs": ["sha256:abc"]},
    )
    verified = advance(
        ready,
        "action.verified",
        {"action_id": "action-1", "verification_id": "verification-1"},
    )
    committed = advance(
        verified,
        "action.committed",
        {
            "action_id": "action-1",
            "new_anchor_id": "base-1",
            "accepted_patch_id": "patch-1",
        },
    )

    assert initial.pending_action is None
    assert queued.pending_action is not None
    assert committed.pending_action is None
    assert committed.anchor_id == "base-1"
    assert committed.anchor_generation == 1
    assert committed.action_history[0].status is ActionStatus.COMMITTED


def test_immutable_state_rejects_assignment() -> None:
    state = running_state()
    with pytest.raises(FrozenInstanceError):
        state.sequence = 99  # type: ignore[misc]


def test_illegal_action_transition_is_rejected() -> None:
    state = running_state()
    with pytest.raises(StateTransitionError) as failure:
        advance(state, "action.started", {"action_id": "missing"})
    assert failure.value.reason_code == "action_not_pending"


def test_stale_anchor_is_rejected() -> None:
    state = running_state()
    with pytest.raises(StateTransitionError) as failure:
        advance(
            state,
            "action.queued",
            {
                "action_id": "action-1",
                "action_type": "optimize",
                "parent_anchor_id": "old-anchor",
                "parent_anchor_generation": 0,
            },
        )
    assert failure.value.reason_code == "stale_anchor"


def test_terminal_run_rejects_future_events() -> None:
    terminal = advance(running_state(), "run.succeeded", {"reason": "target met"})
    assert terminal.phase is RunPhase.SUCCEEDED

    with pytest.raises(StateTransitionError) as failure:
        advance(terminal, "run.failed", {"reason": "too late"})
    assert failure.value.reason_code == "run_is_terminal"


def test_event_must_extend_current_parent() -> None:
    state = running_state()
    event = Event(2, "event-2", "run-1", "run.succeeded", {"reason": "done"}, None)
    with pytest.raises(StateTransitionError) as failure:
        reduce_event(state, event)
    assert failure.value.reason_code == "event_parent_stale"
