"""Atomic append/reduce support for causally linked outcome events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from apex.core import ContractError, IntegrityError, sha256_json

from .state import WorkloadState
from .transitions import EventLike, reduce_event


@dataclass(frozen=True, slots=True)
class AtomicEventInput:
    event_type: str
    payload: Mapping[str, Any]
    idempotency_key: str
    parent_event_id: str | None


class AtomicTransactionReceipt(Protocol):
    events: Sequence[EventLike]


class AtomicJournalPort(Protocol):
    def append_transaction(
        self,
        *,
        run_id: str,
        events: Sequence[AtomicEventInput],
    ) -> AtomicTransactionReceipt: ...

    def get_by_idempotency_key(
        self,
        run_id: str,
        idempotency_key: str,
    ) -> EventLike | None: ...


@dataclass(frozen=True, slots=True)
class ProposedEvent:
    sequence: int
    event_id: str
    run_id: str
    event_type: str
    payload: Mapping[str, Any]
    parent_event_id: str | None


def append_reduced_transaction(
    journal: AtomicJournalPort,
    state: WorkloadState,
    values: Sequence[tuple[str, Mapping[str, Any], str]],
) -> WorkloadState:
    """Validate and atomically append a short causal event chain."""

    if not values:
        raise ContractError("Event transaction is empty", "empty_transaction")
    existing = tuple(
        journal.get_by_idempotency_key(state.run_id, key)
        for _, _, key in values
    )
    if any(item is not None for item in existing):
        if not all(item is not None for item in existing):
            raise IntegrityError(
                "Atomic E2E outcome is partially committed",
                "partial_duplicate_transaction",
            )
        inputs = tuple(
            AtomicEventInput(event_type, payload, key, item.parent_event_id)
            for (event_type, payload, key), item in zip(
                values,
                existing,
                strict=True,
            )
            if item is not None
        )
    else:
        inputs = _new_inputs(state, values)
    receipt = journal.append_transaction(run_id=state.run_id, events=inputs)
    successor = state
    for event in receipt.events:
        if event.sequence > successor.sequence:
            successor = reduce_event(successor, event)
    return successor


def _new_inputs(
    state: WorkloadState,
    values: Sequence[tuple[str, Mapping[str, Any], str]],
) -> tuple[AtomicEventInput, ...]:
    parent = state.last_event_id
    proposed_state = state
    inputs: list[AtomicEventInput] = []
    for offset, (event_type, payload, key) in enumerate(values, start=1):
        event_id = _event_id(state.run_id, key)
        proposal = ProposedEvent(
            state.sequence + offset,
            event_id,
            state.run_id,
            event_type,
            payload,
            parent,
        )
        proposed_state = reduce_event(proposed_state, proposal)
        inputs.append(AtomicEventInput(event_type, payload, key, parent))
        parent = event_id
    return tuple(inputs)


def _event_id(run_id: str, idempotency_key: str) -> str:
    return f"evt-{sha256_json({'run_id': run_id, 'idempotency_key': idempotency_key})}"


__all__ = [
    "AtomicEventInput",
    "AtomicJournalPort",
    "AtomicTransactionReceipt",
    "ProposedEvent",
    "append_reduced_transaction",
]
