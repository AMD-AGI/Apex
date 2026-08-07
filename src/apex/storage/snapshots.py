"""Disposable, integrity-checked state snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_json

from ._atomic import FaultHook, atomic_write_bytes, fsync_directory


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A verified projection at a journal high-water mark."""

    high_water_mark: int
    payload: Mapping[str, Any]
    payload_hash: str


class SnapshotStore:
    """Persist a rebuildable projection outside the canonical journal."""

    def __init__(self, path: Path, *, fault_hook: FaultHook | None = None) -> None:
        self.path = Path(path)
        self._fault_hook = fault_hook

    def save(self, *, high_water_mark: int, payload: Mapping[str, Any]) -> Snapshot:
        if high_water_mark < 0:
            raise ContractError("Snapshot high-water mark cannot be negative", "invalid_snapshot_mark")
        normalized = self._normalize_payload(payload)
        payload_hash = sha256_json(normalized)
        envelope = {
            "schema_version": 1,
            "high_water_mark": high_water_mark,
            "payload": normalized,
            "payload_hash": payload_hash,
        }
        atomic_write_bytes(
            self.path,
            canonical_json_bytes(envelope),
            fault_hook=self._fault_hook,
        )
        return Snapshot(high_water_mark, normalized, payload_hash)

    def load(self) -> Snapshot | None:
        if not self.path.exists():
            return None
        try:
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
            mark = int(envelope["high_water_mark"])
            payload = self._normalize_payload(envelope["payload"])
            payload_hash = str(envelope["payload_hash"])
        except (
            ContractError,
            KeyError,
            OSError,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise IntegrityError("Snapshot is malformed", "snapshot_malformed") from error
        if envelope.get("schema_version") != 1 or mark < 0:
            raise IntegrityError("Snapshot schema is invalid", "snapshot_schema_invalid")
        if sha256_json(payload) != payload_hash:
            raise IntegrityError("Snapshot payload hash does not match", "snapshot_digest_mismatch")
        return Snapshot(mark, payload, payload_hash)

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()
            fsync_directory(self.path.parent)

    @staticmethod
    def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ContractError("Snapshot payload must be a mapping", "invalid_snapshot_payload")
        try:
            return json.loads(canonical_json_bytes(dict(payload)))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ContractError("Snapshot payload must be JSON-compatible", "invalid_snapshot_payload") from error


__all__ = ["Snapshot", "SnapshotStore"]
