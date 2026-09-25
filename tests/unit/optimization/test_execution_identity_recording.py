from __future__ import annotations

import json

import pytest

from apex.core import IntegrityError
from apex.optimization.execution_identity_recording import (
    load_recorded_execution_identity,
    record_apex_execution_identity,
    require_same_execution_identity,
)
from apex.orchestration import RunController
from apex.storage import ArtifactStore, EventJournal, SnapshotStore
from tests.support.execution_identity import execution_identity


def _stores(tmp_path):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    journal = EventJournal(tmp_path / "events" / "run.db")
    controller = RunController.create(
        "run-identity",
        journal,
        SnapshotStore(tmp_path / "state.snapshot.json"),
        initial_anchor_id="anchor-identity",
    )
    return artifacts, journal, controller


def test_execution_identity_is_canonical_event_and_cas_evidence(tmp_path) -> None:
    artifacts, journal, controller = _stores(tmp_path)
    identity = execution_identity(dependency_lock_sha256="e" * 64)

    receipt = record_apex_execution_identity(artifacts, controller, identity)
    events = journal.iter_events("run-identity", verify=True)
    event = events[-1]

    assert json.loads(artifacts.read_bytes(receipt)) == identity.to_dict()
    assert event.event_type == "provenance_observed"
    assert event.payload["kind"] == "apex_execution_identity"
    assert event.payload["execution_identity_sha256"] == identity.receipt_sha256
    assert event.payload["artifacts"][0]["role"] == "apex_execution_identity"
    assert load_recorded_execution_identity(artifacts, events) == identity


def test_execution_identity_mismatch_blocks_resume() -> None:
    original = execution_identity(apex_tree="a" * 40)
    changed = execution_identity(apex_tree="e" * 40)

    require_same_execution_identity(original.receipt_sha256, original)
    with pytest.raises(IntegrityError) as caught:
        require_same_execution_identity(original.receipt_sha256, changed)
    assert caught.value.reason_code == "resume_execution_identity_mismatch"


def test_missing_execution_identity_event_is_rejected(tmp_path) -> None:
    artifacts, journal, _controller = _stores(tmp_path)

    with pytest.raises(IntegrityError) as caught:
        load_recorded_execution_identity(
            artifacts,
            journal.iter_events("run-identity", verify=True),
        )
    assert caught.value.reason_code == "execution_identity_ambiguous"
