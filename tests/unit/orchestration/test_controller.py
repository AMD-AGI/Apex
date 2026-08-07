from __future__ import annotations

import pytest

from apex.core import ContractError, StateTransitionError
from apex.orchestration import ActionStatus, RunController, RunPhase
from apex.storage import EventJournal, SnapshotStore


def stores(tmp_path):
    return EventJournal(tmp_path / "events.db"), SnapshotStore(tmp_path / "snapshot.json")


def test_controller_is_sole_writer_for_full_action_lifecycle(tmp_path) -> None:
    journal, snapshots = stores(tmp_path)
    controller = RunController.create("run-1", journal, snapshots, initial_anchor_id="source-0")
    controller.queue_action("action-1", "kernel-optimize")
    controller.start_action("action-1")
    controller.mark_artifacts_ready("action-1", ["sha256:artifact"])
    controller.verify_action("action-1", "verification-1")
    controller.commit_action(
        "action-1",
        new_anchor_id="source-1",
        accepted_patch_id="patch-1",
    )
    final = controller.finish(RunPhase.SUCCEEDED, reason="throughput improved")

    assert final.phase is RunPhase.SUCCEEDED
    assert final.anchor_id == "source-1"
    assert final.accepted_patch_ids == ("patch-1",)
    assert final.action_history[0].status is ActionStatus.COMMITTED
    assert snapshots.load().payload == final.to_dict()  # type: ignore[union-attr]


def test_illegal_transition_is_rejected_before_journal_append(tmp_path) -> None:
    journal, snapshots = stores(tmp_path)
    controller = RunController.create("run-1", journal, snapshots)
    sequence = controller.state.sequence

    with pytest.raises(StateTransitionError):
        controller.start_action("not-queued")

    assert controller.state.sequence == sequence
    assert len(journal.iter_events("run-1")) == 1


def test_retry_of_committed_transition_is_idempotent(tmp_path) -> None:
    journal, snapshots = stores(tmp_path)
    controller = RunController.create("run-1", journal, snapshots)
    controller.queue_action("action-1", "optimize")
    started = controller.start_action("action-1")

    assert controller.start_action("action-1") is started
    assert len(journal.iter_events("run-1")) == 3


def test_run_cannot_finish_with_pending_action(tmp_path) -> None:
    journal, snapshots = stores(tmp_path)
    controller = RunController.create("run-1", journal, snapshots)
    controller.queue_action("action-1", "optimize")

    with pytest.raises(StateTransitionError) as failure:
        controller.finish(RunPhase.SUCCEEDED, reason="premature")
    assert failure.value.reason_code == "pending_action_at_stop"


def test_rebuild_snapshot_uses_only_journal_history(tmp_path) -> None:
    journal, snapshots = stores(tmp_path)
    controller = RunController.create("run-1", journal, snapshots)
    controller.queue_action("action-1", "optimize")
    expected = controller.state
    snapshots.delete()

    assert controller.rebuild_snapshot() == expected
    assert snapshots.load().payload == expected.to_dict()  # type: ignore[union-attr]


def test_domain_evidence_advances_head_without_mutating_anchor(tmp_path) -> None:
    journal, snapshots = stores(tmp_path)
    controller = RunController.create("run-1", journal, snapshots, initial_anchor_id="source-0")
    before = controller.state

    after = controller.record_domain_event(
        "context_packet_created",
        {"attempt_id": "attempt-1", "context_packet_id": "context-1"},
        idempotency_key="attempt-1.context",
    )

    assert after.sequence == before.sequence + 1
    assert after.anchor_id == before.anchor_id
    assert after.anchor_generation == before.anchor_generation
    assert RunController.recover("run-1", journal, snapshots).state == after

    with pytest.raises(ContractError) as failure:
        controller.record_domain_event(
            "agent_claimed_keep",
            {"anchor_id": "forged"},
            idempotency_key="forged",
        )
    assert failure.value.reason_code == "unknown_domain_event"
