from __future__ import annotations

import json

import pytest

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes
from apex.optimization.baseline_recording import (
    record_campaign_baseline,
    validate_resume_campaign_baseline,
)
from apex.orchestration import RunController
from apex.runtime import ReleaseCandidateReceipt
from apex.storage import ArtifactStore, EventJournal, SnapshotStore


def _document() -> dict:
    value = {
        "schema": "apex.release-candidate-receipt/v2",
        "baseline_status": "ready",
        "baseline_blockers": [],
        "status": "blocked",
        "blockers": ["showcases_pending"],
        "static": {"apex_checkout": {"tree": "a" * 40}},
        "evidence": {},
        "qualification_authorities": [],
    }
    value["receipt_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def test_verified_campaign_baseline_is_canonical_event_and_cas_evidence(tmp_path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    journal = EventJournal(tmp_path / "events" / "run.db")
    controller = RunController.create(
        "run-baseline",
        journal,
        SnapshotStore(tmp_path / "state.snapshot.json"),
        initial_anchor_id="anchor-baseline",
    )
    document = _document()

    receipt = record_campaign_baseline(artifacts, controller, document)
    event = journal.iter_events("run-baseline", verify=True)[-1]

    assert json.loads(artifacts.read_bytes(receipt)) == document
    assert event.event_type == "dependency_verified"
    assert event.payload["kind"] == "campaign_baseline"
    assert event.payload["release_candidate_receipt_sha256"] == document["receipt_sha256"]
    assert event.payload["artifacts"][0]["role"] == "campaign_baseline"


def test_blocked_or_self_digest_tampered_baseline_is_not_recorded(tmp_path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    journal = EventJournal(tmp_path / "events" / "run.db")
    controller = RunController.create(
        "run-baseline",
        journal,
        SnapshotStore(tmp_path / "state.snapshot.json"),
        initial_anchor_id="anchor-baseline",
    )
    document = _document()
    document["baseline_status"] = "blocked"

    with pytest.raises(ContractError, match="invalid or blocked"):
        record_campaign_baseline(artifacts, controller, document)

    assert len(journal.iter_events("run-baseline", verify=True)) == 1


def test_resume_requires_the_exact_original_baseline_bytes() -> None:
    document = _document()
    payload = {key: item for key, item in document.items() if key != "receipt_sha256"}
    content = canonical_json_bytes(payload)
    receipt = ReleaseCandidateReceipt(content, sha256_bytes(content))

    validate_resume_campaign_baseline(document, receipt)
    with pytest.raises(ContractError, match="requires the original"):
        validate_resume_campaign_baseline(document, None)

    changed = _document()
    changed["blockers"] = ["different_future_gate"]
    changed_payload = {
        key: item for key, item in changed.items() if key != "receipt_sha256"
    }
    changed_content = canonical_json_bytes(changed_payload)
    changed_receipt = ReleaseCandidateReceipt(
        changed_content,
        sha256_bytes(changed_content),
    )
    with pytest.raises(IntegrityError, match="differs from the original"):
        validate_resume_campaign_baseline(document, changed_receipt)
