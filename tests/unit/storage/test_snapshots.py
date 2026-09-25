from __future__ import annotations

import json

import pytest

from apex.core import IntegrityError
from apex.storage import SnapshotStore


def test_snapshot_round_trip_overwrite_and_delete(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "state" / "snapshot.json")
    first = store.save(high_water_mark=1, payload={"phase": "running"})
    second = store.save(high_water_mark=2, payload={"phase": "done"})

    assert first.payload_hash != second.payload_hash
    assert store.load() == second
    store.delete()
    assert store.load() is None


def test_snapshot_tampering_is_detected(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    store = SnapshotStore(path)
    store.save(high_water_mark=4, payload={"accepted": []})
    envelope = json.loads(path.read_text())
    envelope["payload"]["accepted"] = ["forged"]
    path.write_text(json.dumps(envelope))

    with pytest.raises(IntegrityError) as failure:
        store.load()
    assert failure.value.reason_code == "snapshot_digest_mismatch"


def test_failed_snapshot_replace_preserves_previous_projection(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    stable = SnapshotStore(path)
    previous = stable.save(high_water_mark=1, payload={"phase": "old"})

    def fault(stage: str) -> None:
        if stage == "before_replace":
            raise RuntimeError("crash")

    failing = SnapshotStore(path, fault_hook=fault)
    with pytest.raises(RuntimeError, match="crash"):
        failing.save(high_water_mark=2, payload={"phase": "new"})

    assert stable.load() == previous
