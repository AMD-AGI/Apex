"""Small immutable context projections used by E2E search recovery."""

from __future__ import annotations

from typing import Protocol, Sequence

from apex.benchmark import BenchmarkConfigViews
from apex.core import IntegrityError
from apex.storage import EventRecord

from .run_record import E2ERunRecord


class EventIndexView(Protocol):
    events: Sequence[EventRecord]


def initial_anchor_id(index: EventIndexView) -> str:
    """Recover the unique frozen original anchor identity."""

    events = tuple(event for event in index.events if event.event_type == "run.started")
    if len(events) != 1 or not isinstance(events[0].payload.get("initial_anchor_id"), str):
        raise IntegrityError("Initial anchor is missing", "anchor_lineage_mismatch")
    return str(events[0].payload["initial_anchor_id"])


def protocol_hash(record: E2ERunRecord) -> str:
    """Read the frozen measurement protocol from replayed state."""

    search = record.controller.state.e2e
    if search is None:
        raise IntegrityError("E2E state is absent", "recovery_lineage_incomplete")
    return search.measurement_protocol_hash


def views_from_state(record: E2ERunRecord) -> BenchmarkConfigViews:
    """Build the minimal view used to recheck recovered deployment semantics."""

    search = record.controller.state.e2e
    if search is None:
        raise IntegrityError("E2E state is absent", "recovery_lineage_incomplete")
    placeholder = record.root / "run.request.json"
    return BenchmarkConfigViews(
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        "0" * 64,
        search.measurement_protocol_hash,
        "",
        None,
    )


__all__ = ["initial_anchor_id", "protocol_hash", "views_from_state"]
