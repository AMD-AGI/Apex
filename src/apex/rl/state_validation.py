"""Fail-closed validation for optional workload-state RL projections."""

from __future__ import annotations

from typing import Sequence

from apex.core import ContractError, IntegrityError, StateTransitionError
from apex.orchestration import WorkloadState
from apex.orchestration.replay import replay_workload_state
from apex.storage import EventRecord


def validate_workload_state(
    run_id: str,
    records: Sequence[EventRecord],
    state: WorkloadState | None,
) -> None:
    """Require supplied state to equal pure replay of its journal prefix."""

    if state is None:
        return
    if state.run_id != run_id or state.sequence > records[-1].sequence:
        raise IntegrityError(
            "WorkloadState is not anchored to this run",
            "state_run_mismatch",
        )
    if state.sequence:
        event = next(
            (item for item in records if item.sequence == state.sequence),
            None,
        )
        if event is None or event.event_id != state.last_event_id:
            raise IntegrityError(
                "WorkloadState head does not match journal",
                "state_head_mismatch",
            )
    prefix = tuple(item for item in records if item.sequence <= state.sequence)
    try:
        replayed = (
            replay_workload_state(run_id, prefix)
            if prefix
            else WorkloadState.initial(run_id)
        )
    except (ContractError, StateTransitionError) as error:
        raise IntegrityError(
            "WorkloadState cannot be replayed from its canonical journal prefix",
            "state_replay_failed",
        ) from error
    if state != replayed:
        raise IntegrityError(
            "WorkloadState differs from canonical journal replay",
            "state_projection_mismatch",
        )


__all__ = ["validate_workload_state"]
