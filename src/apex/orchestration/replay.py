"""Pure workload-state replay for read-only projections."""

from __future__ import annotations

from collections.abc import Iterable

from apex.core import ContractError, validate_identifier

from .state import WorkloadState
from .transitions import EventLike, reduce_event


def replay_workload_state(
    run_id: str, events: Iterable[EventLike]
) -> WorkloadState:
    """Reduce a verified journal without reading or writing a snapshot."""

    validate_identifier(run_id, field_name="run_id")
    records = tuple(events)
    if not records:
        raise ContractError("Run has no canonical events", "run_not_found")
    state = WorkloadState.initial(run_id)
    for event in records:
        state = reduce_event(state, event)
    return state


__all__ = ["replay_workload_state"]
