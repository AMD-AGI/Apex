"""SQLite-backed canonical append-only event journal."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_json,
    validate_identifier,
)

from ._atomic import fsync_directory


FaultHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class EventInput:
    """Caller-supplied semantic event fields."""

    event_type: str
    payload: Mapping[str, Any]
    idempotency_key: str
    parent_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One immutable, checksummed journal record."""

    sequence: int
    event_id: str
    run_id: str
    event_type: str
    payload: Mapping[str, Any]
    parent_event_id: str | None
    idempotency_key: str
    transaction_id: str
    created_at_ns: int
    checksum: str


@dataclass(frozen=True, slots=True)
class TransactionReceipt:
    """Evidence that a group of events committed atomically."""

    transaction_id: str
    first_sequence: int
    last_sequence: int
    checksum: str
    events: tuple[EventRecord, ...]


class EventJournal:
    """Canonical event history using serialized SQLite transactions."""

    def __init__(self, path: Path, *, fault_hook: FaultHook | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fault_hook = fault_hook
        self._read_only = False
        self._initialize()

    @classmethod
    def open_read_only(cls, path: Path) -> "EventJournal":
        """Open an existing canonical journal without schema or WAL mutation."""

        selected = Path(path)
        if selected.is_symlink() or not selected.is_file():
            raise ContractError(
                "Read-only event journal must be an existing regular file",
                "invalid_event_journal",
            )
        journal = cls.__new__(cls)
        journal.path = selected.resolve(strict=True)
        journal._fault_hook = None
        journal._read_only = True
        return journal

    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
        parent_event_id: str | None = None,
    ) -> EventRecord:
        receipt = self.append_transaction(
            run_id=run_id,
            events=(EventInput(event_type, payload, idempotency_key, parent_event_id),),
        )
        return receipt.events[0]

    def append_transaction(
        self,
        *,
        run_id: str,
        events: Sequence[EventInput],
    ) -> TransactionReceipt:
        if self._read_only:
            raise ContractError("Event journal is read-only", "event_journal_read_only")
        validate_identifier(run_id, field_name="run_id")
        normalized = self._normalize_inputs(events)
        event_ids = tuple(derive_event_id(run_id, item.idempotency_key) for item in normalized)
        transaction_id = _derive_transaction_id(run_id, event_ids)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._existing_by_key(connection, run_id, normalized)
            if existing:
                receipt = self._resolve_duplicate(
                    connection, run_id, normalized, event_ids, transaction_id, existing
                )
                connection.commit()
                return receipt
            self._validate_parents(connection, run_id, normalized, event_ids)
            receipt = self._insert_transaction(
                connection, run_id, normalized, event_ids, transaction_id
            )
            self._fault("before_commit")
            connection.commit()
            return receipt
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def iter_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        verify: bool = True,
    ) -> tuple[EventRecord, ...]:
        validate_identifier(run_id, field_name="run_id")
        connection = self._connect()
        try:
            records = self._records_for_run(connection, run_id)
            if verify:
                self._verify_records(connection, records)
            return tuple(record for record in records if record.sequence > after_sequence)
        finally:
            connection.close()

    def verify_run(self, run_id: str) -> None:
        self.iter_events(run_id, verify=True)

    def last_event(self, run_id: str) -> EventRecord | None:
        records = self.iter_events(run_id, verify=True)
        return records[-1] if records else None

    def get_by_idempotency_key(
        self,
        run_id: str,
        idempotency_key: str,
    ) -> EventRecord | None:
        validate_identifier(run_id, field_name="run_id")
        _validate_idempotency_key(idempotency_key)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM events WHERE run_id = ? AND idempotency_key = ?",
                (run_id, idempotency_key),
            ).fetchone()
            if row is None:
                return None
            record = _record_from_row(row)
            self._receipt_for_transaction(connection, record.transaction_id, verify=True)
            return record
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
        finally:
            connection.close()
        fsync_directory(self.path.parent)

    def _connect(self) -> sqlite3.Connection:
        target = f"{self.path.as_uri()}?mode=ro" if self._read_only else self.path
        connection = sqlite3.connect(
            target,
            isolation_level=None,
            timeout=30,
            uri=self._read_only,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if not self._read_only:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _normalize_inputs(self, events: Sequence[EventInput]) -> tuple[EventInput, ...]:
        if not events:
            raise ContractError("A transaction requires at least one event", "empty_transaction")
        normalized: list[EventInput] = []
        keys: set[str] = set()
        for item in events:
            validate_identifier(item.event_type, field_name="event_type")
            _validate_idempotency_key(item.idempotency_key)
            if item.idempotency_key in keys:
                raise ContractError("Duplicate transaction idempotency key", "duplicate_event_key")
            keys.add(item.idempotency_key)
            payload = _normalize_payload(item.payload)
            normalized.append(EventInput(item.event_type, payload, item.idempotency_key, item.parent_event_id))
        return tuple(normalized)

    def _existing_by_key(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        events: Sequence[EventInput],
    ) -> dict[str, sqlite3.Row]:
        placeholders = ",".join("?" for _ in events)
        keys = tuple(item.idempotency_key for item in events)
        rows = connection.execute(
            f"SELECT * FROM events WHERE run_id = ? AND idempotency_key IN ({placeholders})",
            (run_id, *keys),
        ).fetchall()
        if rows and len(rows) != len(events):
            raise IntegrityError("Partial idempotent transaction exists", "partial_duplicate_transaction")
        return {str(row["idempotency_key"]): row for row in rows}

    def _resolve_duplicate(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        events: Sequence[EventInput],
        event_ids: Sequence[str],
        transaction_id: str,
        existing: Mapping[str, sqlite3.Row],
    ) -> TransactionReceipt:
        for item, event_id in zip(events, event_ids):
            row = existing[item.idempotency_key]
            expected = (event_id, item.event_type, _payload_json(item.payload), item.parent_event_id)
            actual = (row["event_id"], row["event_type"], row["payload_json"], row["parent_event_id"])
            if actual != expected or row["transaction_id"] != transaction_id:
                raise IntegrityError("Idempotency key was reused with different content", "idempotency_conflict")
        return self._receipt_for_transaction(connection, transaction_id, verify=True)

    def _validate_parents(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        events: Sequence[EventInput],
        event_ids: Sequence[str],
    ) -> None:
        head = connection.execute(
            "SELECT event_id FROM events WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        expected_parent = str(head["event_id"]) if head is not None else None
        positions = {event_id: index for index, event_id in enumerate(event_ids)}
        for index, item in enumerate(events):
            parent = item.parent_event_id
            if parent != expected_parent:
                raise ContractError("Event does not extend the current run head", "stale_event_parent")
            if parent is None:
                expected_parent = event_ids[index]
                continue
            if parent in positions:
                if positions[parent] >= index:
                    raise ContractError("Event parent must precede its child", "invalid_event_parent")
            else:
                row = connection.execute("SELECT run_id FROM events WHERE event_id = ?", (parent,)).fetchone()
                if row is None:
                    raise ContractError("Event parent does not exist", "event_parent_missing")
                if row["run_id"] != run_id:
                    raise ContractError("Event parent belongs to another run", "event_parent_run_mismatch")
            expected_parent = event_ids[index]

    def _insert_transaction(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        events: Sequence[EventInput],
        event_ids: Sequence[str],
        transaction_id: str,
    ) -> TransactionReceipt:
        maximum = connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM events").fetchone()[0]
        created = time.time_ns()
        records = tuple(
            _make_record(maximum + index + 1, event_id, run_id, item, transaction_id, created + index)
            for index, (item, event_id) in enumerate(zip(events, event_ids))
        )
        checksum = _transaction_checksum(transaction_id, records)
        connection.execute(
            "INSERT INTO transactions VALUES (?, ?, ?, ?, ?)",
            (transaction_id, records[0].sequence, records[-1].sequence, len(records), checksum),
        )
        for record in records:
            self._insert_event(connection, record)
            self._fault("after_event_insert")
        return TransactionReceipt(transaction_id, records[0].sequence, records[-1].sequence, checksum, records)

    @staticmethod
    def _insert_event(connection: sqlite3.Connection, record: EventRecord) -> None:
        connection.execute(
            """INSERT INTO events (
                sequence, event_id, run_id, event_type, payload_json, parent_event_id,
                idempotency_key, transaction_id, created_at_ns, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.sequence,
                record.event_id,
                record.run_id,
                record.event_type,
                _payload_json(record.payload),
                record.parent_event_id,
                record.idempotency_key,
                record.transaction_id,
                record.created_at_ns,
                record.checksum,
            ),
        )

    def _records_for_run(self, connection: sqlite3.Connection, run_id: str) -> tuple[EventRecord, ...]:
        rows = connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY sequence", (run_id,)
        ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def _verify_records(
        self,
        connection: sqlite3.Connection,
        records: Sequence[EventRecord],
    ) -> None:
        transactions: set[str] = set()
        previous_sequence = 0
        for record in records:
            if _event_checksum(record) != record.checksum:
                raise IntegrityError("Journal event checksum mismatch", "event_checksum_mismatch")
            if record.sequence <= previous_sequence:
                raise IntegrityError("Journal sequence is not increasing", "event_sequence_invalid")
            previous_sequence = record.sequence
            transactions.add(record.transaction_id)
        for transaction_id in transactions:
            self._receipt_for_transaction(connection, transaction_id, verify=True)

    def _receipt_for_transaction(
        self,
        connection: sqlite3.Connection,
        transaction_id: str,
        *,
        verify: bool,
    ) -> TransactionReceipt:
        transaction = connection.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
        ).fetchone()
        rows = connection.execute(
            "SELECT * FROM events WHERE transaction_id = ? ORDER BY sequence", (transaction_id,)
        ).fetchall()
        if transaction is None or not rows:
            raise IntegrityError("Journal transaction receipt is missing", "transaction_receipt_missing")
        records = tuple(_record_from_row(row) for row in rows)
        receipt = TransactionReceipt(
            transaction_id,
            int(transaction["first_sequence"]),
            int(transaction["last_sequence"]),
            str(transaction["checksum"]),
            records,
        )
        if verify:
            _verify_transaction_receipt(receipt, int(transaction["event_count"]))
        return receipt

    def _fault(self, stage: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage)


def derive_event_id(run_id: str, idempotency_key: str) -> str:
    """Derive a stable event identity for retries and parent references."""

    validate_identifier(run_id, field_name="run_id")
    _validate_idempotency_key(idempotency_key)
    return f"evt-{sha256_json({'run_id': run_id, 'idempotency_key': idempotency_key})}"


def _derive_transaction_id(run_id: str, event_ids: Sequence[str]) -> str:
    return f"txn-{sha256_json({'run_id': run_id, 'event_ids': list(event_ids)})}"


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ContractError("Event payload must be a mapping", "invalid_event_payload")
    try:
        return json.loads(canonical_json_bytes(dict(payload)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError("Event payload must be JSON-compatible", "invalid_event_payload") from error


def _payload_json(payload: Mapping[str, Any]) -> str:
    return canonical_json_bytes(dict(payload)).decode("utf-8")


def _validate_idempotency_key(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ContractError("Invalid event idempotency key", "invalid_idempotency_key")


def _make_record(
    sequence: int,
    event_id: str,
    run_id: str,
    item: EventInput,
    transaction_id: str,
    created_at_ns: int,
) -> EventRecord:
    provisional = EventRecord(
        sequence,
        event_id,
        run_id,
        item.event_type,
        item.payload,
        item.parent_event_id,
        item.idempotency_key,
        transaction_id,
        created_at_ns,
        "",
    )
    return EventRecord(
        provisional.sequence,
        provisional.event_id,
        provisional.run_id,
        provisional.event_type,
        provisional.payload,
        provisional.parent_event_id,
        provisional.idempotency_key,
        provisional.transaction_id,
        provisional.created_at_ns,
        _event_checksum(provisional),
    )


def _record_from_row(row: sqlite3.Row) -> EventRecord:
    try:
        payload = json.loads(row["payload_json"])
        if not isinstance(payload, dict):
            raise TypeError("payload is not an object")
        canonical_json_bytes(payload)
        return EventRecord(
            int(row["sequence"]),
            str(row["event_id"]),
            str(row["run_id"]),
            str(row["event_type"]),
            payload,
            row["parent_event_id"],
            str(row["idempotency_key"]),
            str(row["transaction_id"]),
            int(row["created_at_ns"]),
            str(row["checksum"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise IntegrityError("Journal event is malformed", "event_malformed") from error


def _event_checksum(record: EventRecord) -> str:
    return sha256_json(
        {
            "sequence": record.sequence,
            "event_id": record.event_id,
            "run_id": record.run_id,
            "event_type": record.event_type,
            "payload": dict(record.payload),
            "parent_event_id": record.parent_event_id,
            "idempotency_key": record.idempotency_key,
            "transaction_id": record.transaction_id,
            "created_at_ns": record.created_at_ns,
        }
    )


def _transaction_checksum(transaction_id: str, records: Sequence[EventRecord]) -> str:
    return sha256_json(
        {
            "transaction_id": transaction_id,
            "event_checksums": [record.checksum for record in records],
        }
    )


def _verify_transaction_receipt(receipt: TransactionReceipt, event_count: int) -> None:
    records = receipt.events
    valid_bounds = (
        len(records) == event_count
        and records[0].sequence == receipt.first_sequence
        and records[-1].sequence == receipt.last_sequence
        and receipt.last_sequence - receipt.first_sequence + 1 == event_count
    )
    if not valid_bounds:
        raise IntegrityError("Journal transaction boundaries mismatch", "transaction_boundary_mismatch")
    if any(_event_checksum(record) != record.checksum for record in records):
        raise IntegrityError("Journal transaction contains a corrupt event", "event_checksum_mismatch")
    if _transaction_checksum(receipt.transaction_id, records) != receipt.checksum:
        raise IntegrityError("Journal transaction checksum mismatch", "transaction_checksum_mismatch")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    first_sequence INTEGER NOT NULL,
    last_sequence INTEGER NOT NULL,
    event_count INTEGER NOT NULL CHECK (event_count > 0),
    checksum TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    parent_event_id TEXT REFERENCES events(event_id),
    idempotency_key TEXT NOT NULL,
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    created_at_ns INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    UNIQUE (run_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS events_run_sequence ON events(run_id, sequence);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS transactions_no_update BEFORE UPDATE ON transactions
BEGIN SELECT RAISE(ABORT, 'transactions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS transactions_no_delete BEFORE DELETE ON transactions
BEGIN SELECT RAISE(ABORT, 'transactions are append-only'); END;
"""


__all__ = [
    "EventInput",
    "EventJournal",
    "EventRecord",
    "TransactionReceipt",
    "derive_event_id",
]
