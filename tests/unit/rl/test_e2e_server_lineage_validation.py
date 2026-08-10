from __future__ import annotations

from dataclasses import replace

import pytest

from apex.core import IntegrityError
from apex.rl.e2e_server_lineage_validation import validate_e2e_server_lineage
from apex.rl.models import (
    EpisodeArtifact,
    EpisodeEvent,
    EvidenceClass,
    SemanticRole,
)
from apex.storage import ArtifactReceipt


RUN_ID = "e2e-lineage"
ANCHOR = "anchor-original"


def test_projection_rebuilds_reuse_cleanup_without_an_attempt_or_reward() -> None:
    reuse = _measurement(2, "reuse")
    cleanup = _measurement(3, "cleanup")

    replay = validate_e2e_server_lineage(
        RUN_ID, (_started(), reuse, cleanup), None
    )

    assert replay.observation_count == 2
    assert replay.active is None
    assert replay.retired_generations == ("2" * 64,)
    assert cleanup.payload.get("attempt_id") is None
    assert cleanup.payload["reward_eligible"] is False
    assert cleanup.payload["server_lineage"]["cleanup_succeeded"] is True


def test_projection_rejects_cleanup_as_an_independent_attempt() -> None:
    cleanup = _measurement(3, "cleanup")
    cleanup = replace(cleanup, payload={**cleanup.payload, "attempt_id": "attempt-1"})

    with pytest.raises(IntegrityError, match="exact active server generation"):
        validate_e2e_server_lineage(
            RUN_ID, (_started(), _measurement(2, "reuse"), cleanup), None
        )


def test_projection_rejects_duplicate_or_drifting_generation() -> None:
    duplicate = _measurement(3, "cleanup", lineage_sequence=1)
    with pytest.raises(IntegrityError, match="duplicated or out of order"):
        validate_e2e_server_lineage(
            RUN_ID, (_started(), _measurement(2, "reuse"), duplicate), None
        )

    drifted = _measurement(3, "cleanup", server_generation="f" * 64)
    with pytest.raises(IntegrityError, match="exact active server generation"):
        validate_e2e_server_lineage(
            RUN_ID, (_started(), _measurement(2, "reuse"), drifted), None
        )


def test_cleanup_artifacts_cannot_be_referenced_by_reward() -> None:
    cleanup = _measurement(3, "cleanup")
    cleanup_digest = next(
        item.receipt.digest
        for item in cleanup.artifacts
        if item.role == "local_server_lineage"
    )
    reward = _event(
        4,
        "reward_committed",
        SemanticRole.REWARD,
        {"measurement_receipt": cleanup_digest},
    )

    with pytest.raises(IntegrityError, match="used as reward evidence"):
        validate_e2e_server_lineage(
            RUN_ID, (_started(), _measurement(2, "reuse"), cleanup, reward), None
        )


def _started() -> EpisodeEvent:
    return _event(
        1,
        "run.started",
        SemanticRole.CONTROL,
        {"initial_anchor_id": ANCHOR},
    )


def _measurement(
    sequence: int,
    lifecycle: str,
    *,
    lineage_sequence: int | None = None,
    server_generation: str = "2" * 64,
) -> EpisodeEvent:
    line_digest = f"{sequence:064x}"
    attestation_digest = f"{sequence + 100:064x}"
    cleanup = lifecycle == "cleanup"
    reference = {
        "schema": "apex.e2e-local-server-lineage-ref/v1",
        "receipt_sha256": line_digest,
        "lineage_sequence": lineage_sequence or sequence - 1,
        "lifecycle": lifecycle,
        "server_source_generation_sha256": "1" * 64,
        "server_generation_sha256": server_generation,
        "server_identity_sha256": "3" * 64,
        "owner": {
            "kind": "anchor",
            "id": ANCHOR,
            "anchor_id": ANCHOR,
            "anchor_generation": 0,
        },
        "reward_eligible": not cleanup,
        "cleanup_succeeded": cleanup,
    }
    return _event(
        sequence,
        "measurement_result",
        SemanticRole.OUTCOME,
        {
            "action_id": f"{lifecycle}-{sequence}",
            "config_sha256": "4" * 64,
            "reward_eligible": not cleanup,
            "server_lineage": reference,
        },
        artifacts=(
            _artifact("local_server_lineage", line_digest, sequence),
            _artifact(
                "benchmark_execution_attestation", attestation_digest, sequence
            ),
        ),
    )


def _event(
    sequence: int,
    event_type: str,
    role: SemanticRole,
    payload: dict[str, object],
    *,
    artifacts: tuple[EpisodeArtifact, ...] = (),
) -> EpisodeEvent:
    event_id = f"event-{sequence}"
    rebound = tuple(replace(item, event_id=event_id) for item in artifacts)
    return EpisodeEvent(
        sequence=sequence,
        event_id=event_id,
        transaction_id=f"tx-{sequence}",
        parent_event_id=None if sequence == 1 else f"event-{sequence - 1}",
        event_type=event_type,
        semantic_role=role,
        evidence_class=(
            EvidenceClass.MEASURED
            if event_type == "measurement_result"
            else EvidenceClass.UNSPECIFIED
        ),
        payload=payload,
        artifacts=rebound,
        causation_id=None,
        correlation_id=None,
        agent_run_id=None,
    )


def _artifact(role: str, digest: str, sequence: int) -> EpisodeArtifact:
    return EpisodeArtifact(
        role,
        ArtifactReceipt(
            digest,
            1,
            "application/json",
            f"sha256/{digest[:2]}/{digest}",
        ),
        f"event-{sequence}",
    )
