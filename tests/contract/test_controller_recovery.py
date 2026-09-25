from __future__ import annotations

import json

import pytest

from apex.core import ContractError, StateTransitionError
from apex.orchestration import ActionStatus, RunController
from apex.storage import EventJournal, SnapshotStore


def test_deleted_snapshot_rebuilds_to_identical_state(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    snapshots = SnapshotStore(tmp_path / "snapshot.json")
    controller = RunController.create("run-1", journal, snapshots)
    controller.queue_action("action-1", "optimize")
    controller.start_action("action-1")
    expected = controller.state.to_dict()
    original_snapshot = snapshots.path.read_bytes()
    snapshots.delete()

    recovered = RunController.recover("run-1", journal, snapshots)

    assert recovered.state.to_dict() == expected
    assert snapshots.load().payload == expected  # type: ignore[union-attr]
    assert snapshots.path.read_bytes() == original_snapshot


def test_journal_commit_survives_snapshot_crash_and_resumes_pending_action(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    path = tmp_path / "snapshot.json"
    stable = SnapshotStore(path)
    controller = RunController.create("run-1", journal, stable)
    controller.queue_action("action-1", "optimize")
    enabled = False

    def crash(stage: str) -> None:
        if enabled and stage == "before_replace":
            raise RuntimeError("snapshot crash")

    flaky = SnapshotStore(path, fault_hook=crash)
    resumed = RunController.recover("run-1", journal, flaky)
    enabled = True
    with pytest.raises(RuntimeError, match="snapshot crash"):
        resumed.start_action("action-1")

    recovered = RunController.recover("run-1", journal, stable)
    assert recovered.state.pending_action is not None
    assert recovered.state.pending_action.status is ActionStatus.STARTED
    recovered.abort_pending("operator chose a clean retry")
    assert recovered.state.pending_action is None
    assert recovered.state.action_history[-1].status is ActionStatus.ABORTED


def test_corrupt_snapshot_is_ignored_and_replaced_from_journal(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    path = tmp_path / "snapshot.json"
    snapshots = SnapshotStore(path)
    controller = RunController.create("run-1", journal, snapshots)
    controller.queue_action("action-1", "optimize")
    expected = controller.state.to_dict()
    envelope = json.loads(path.read_text())
    envelope["payload"]["anchor_id"] = "forged"
    path.write_text(json.dumps(envelope))

    recovered = RunController.recover("run-1", journal, snapshots)

    assert recovered.state.to_dict() == expected
    assert snapshots.load().payload == expected  # type: ignore[union-attr]


def test_stale_anchor_is_rejected_without_journal_side_effect(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    snapshots = SnapshotStore(tmp_path / "snapshot.json")
    controller = RunController.create("run-1", journal, snapshots, initial_anchor_id="source-0")
    controller.queue_action("action-1", "optimize")
    controller.start_action("action-1")
    controller.mark_artifacts_ready("action-1", ["sha256:a"])
    controller.verify_action("action-1", "verification-1")
    controller.commit_action(
        "action-1",
        new_anchor_id="source-1",
        accepted_patch_id="patch-1",
    )
    sequence = controller.state.sequence

    with pytest.raises(StateTransitionError) as failure:
        controller.queue_action(
            "action-2",
            "optimize",
            parent_anchor_id="source-0",
            parent_anchor_generation=0,
        )

    assert failure.value.reason_code == "stale_anchor"
    assert controller.state.sequence == sequence
    assert journal.last_event("run-1").sequence == sequence  # type: ignore[union-attr]


def test_second_controller_cannot_branch_from_stale_journal_head(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.db")
    snapshots = SnapshotStore(tmp_path / "snapshot.json")
    first = RunController.create("run-1", journal, snapshots)
    stale = RunController.recover("run-1", journal, snapshots)
    first.queue_action("action-1", "optimize")

    with pytest.raises(ContractError) as failure:
        stale.queue_action("action-2", "optimize")

    assert failure.value.reason_code == "stale_event_parent"
    assert len(journal.iter_events("run-1")) == 2
