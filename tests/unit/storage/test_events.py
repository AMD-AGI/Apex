from __future__ import annotations

import sqlite3

import pytest

from apex.core import ContractError, IntegrityError
from apex.storage import EventInput, EventJournal, derive_event_id


def test_transaction_is_atomic_parented_and_replayable(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    first_id = derive_event_id("run-1", "step.one")

    receipt = journal.append_transaction(
        run_id="run-1",
        events=(
            EventInput("step.started", {"attempt": 1}, "step.one"),
            EventInput("step.finished", {"result": "ok"}, "step.two", first_id),
        ),
    )

    assert receipt.first_sequence == 1
    assert receipt.last_sequence == 2
    assert receipt.events[1].parent_event_id == receipt.events[0].event_id
    assert journal.iter_events("run-1") == receipt.events
    journal.verify_run("run-1")


def test_duplicate_transaction_returns_original_receipt_and_conflict_fails(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    kwargs = {
        "run_id": "run-1",
        "event_type": "run.started",
        "payload": {"version": 1},
        "idempotency_key": "run.started",
    }
    original = journal.append(**kwargs)

    assert journal.append(**kwargs) == original
    with pytest.raises(IntegrityError, match="different content") as failure:
        journal.append(**{**kwargs, "payload": {"version": 2}})
    assert failure.value.reason_code == "idempotency_conflict"
    assert len(journal.iter_events("run-1")) == 1


def test_fault_during_batch_rolls_back_all_events(tmp_path) -> None:
    inserts = 0

    def fault(stage: str) -> None:
        nonlocal inserts
        if stage == "after_event_insert":
            inserts += 1
            if inserts == 2:
                raise RuntimeError("simulated process death")

    journal = EventJournal(tmp_path / "events.db", fault_hook=fault)
    first_id = derive_event_id("run-1", "one")

    with pytest.raises(RuntimeError, match="process death"):
        journal.append_transaction(
            run_id="run-1",
            events=(
                EventInput("step.one", {}, "one"),
                EventInput("step.two", {}, "two", first_id),
            ),
        )

    assert journal.iter_events("run-1") == ()


def test_append_only_trigger_rejects_update_and_delete(tmp_path) -> None:
    path = tmp_path / "events.db"
    journal = EventJournal(path)
    journal.append(run_id="run-1", event_type="run.started", payload={}, idempotency_key="start")

    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE events SET event_type = 'tampered'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM events")


def test_checksum_detects_direct_database_tampering(tmp_path) -> None:
    path = tmp_path / "events.db"
    journal = EventJournal(path)
    journal.append(
        run_id="run-1",
        event_type="run.started",
        payload={"trusted": True},
        idempotency_key="start",
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET payload_json = '{\"trusted\":false}'")

    with pytest.raises(IntegrityError) as failure:
        journal.iter_events("run-1")
    assert failure.value.reason_code == "event_checksum_mismatch"


def test_stale_parent_is_rejected_without_advancing_journal(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    head = journal.append(
        run_id="run-1",
        event_type="run.started",
        payload={},
        idempotency_key="start",
    )

    with pytest.raises(ContractError) as failure:
        journal.append(
            run_id="run-1",
            event_type="step.started",
            payload={},
            idempotency_key="step",
            parent_event_id=None,
        )
    assert failure.value.reason_code == "stale_event_parent"
    assert journal.last_event("run-1") == head


def test_transaction_cannot_contain_forward_parent(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    second_id = derive_event_id("run-1", "two")
    with pytest.raises(ContractError):
        journal.append_transaction(
            run_id="run-1",
            events=(
                EventInput("step.one", {}, "one", second_id),
                EventInput("step.two", {}, "two"),
            ),
        )
