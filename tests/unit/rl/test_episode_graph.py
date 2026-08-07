from __future__ import annotations

from dataclasses import replace
import json

import pytest

from apex.core import IntegrityError
from apex.rl import EpisodeGraphMaterializer, EvidenceClass, SemanticRole
from apex.storage import ArtifactReceipt

from .conftest import append_event, artifact_binding, make_packet


def test_materializes_parent_positive_and_negative_child_episodes(canonical_run):
    materializer = EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    )
    graph = materializer.materialize(
        canonical_run["run_id"],
        workload_state=canonical_run["state"],
        context_packets={
            canonical_run["packet"].context_packet_id: canonical_run["packet"]
        },
    )

    assert graph.parent.kind == "workload"
    assert graph.parent.terminal_status == "succeeded"
    assert graph.workload_state_hash is not None
    assert [child.attempt_id for child in graph.children] == ["attempt-1", "attempt-2"]
    kept, failed = graph.children
    assert kept.status == "success"
    assert kept.verdict == "keep"
    assert kept.trainability == "complete"
    assert kept.context_packet_receipt == canonical_run["packet_receipt"]
    assert kept.scalar_reward == 140.0
    assert kept.policy_ids == ("kernel_robust_v1",)
    assert any(event.semantic_role is SemanticRole.TOOL for event in kept.events)
    assert any(
        event.evidence_class is EvidenceClass.MEASURED for event in kept.events
    )
    assert failed.status == "infrastructure_error"
    assert failed.trainability == "complete"
    assert any(event.semantic_role is SemanticRole.FAILURE for event in failed.events)


def test_materialization_is_byte_stable(canonical_run):
    materializer = EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    )
    first = materializer.materialize(
        canonical_run["run_id"], workload_state=canonical_run["state"]
    )
    second = materializer.materialize(
        canonical_run["run_id"], workload_state=canonical_run["state"]
    )
    assert first.canonical_bytes == second.canonical_bytes
    assert first.graph_id == second.graph_id


def test_rejects_state_not_anchored_to_journal(canonical_run):
    wrong = replace(canonical_run["state"], last_event_id="event-does-not-exist")
    with pytest.raises(IntegrityError, match="head"):
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"], workload_state=wrong)


def test_context_observation_must_be_exact_canonical_packet(canonical_run):
    document = canonical_run["packet"].to_dict()
    noncanonical = canonical_run["artifacts"].put_bytes(
        json.dumps(document, indent=2).encode(), media_type="application/json"
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "context_packet_created",
        {
            "attempt_id": "attempt-noncanonical",
            "artifacts": [artifact_binding("context_packet", noncanonical)],
        },
        "attempt-noncanonical-context",
    )
    with pytest.raises(IntegrityError) as error:
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"])
    assert error.value.reason_code == "invalid_context_packet_artifact"


def test_supplied_context_packet_must_match_cas_bytes(canonical_run):
    different = make_packet(canonical_run["run_id"], cycle=9)
    with pytest.raises(IntegrityError) as error:
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(
            canonical_run["run_id"],
            context_packets={canonical_run["packet"].context_packet_id: different},
        )
    assert error.value.reason_code == "context_packet_mismatch"


def test_declared_missing_cas_receipt_fails_closed(canonical_run):
    missing = ArtifactReceipt(
        digest="f" * 64,
        size=4,
        media_type="application/json",
        relative_path=f"sha256/ff/{'f' * 64}",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "observation_created",
        {
            "attempt_id": "attempt-3",
            "artifacts": [artifact_binding("source", missing)],
        },
        "attempt-3-missing",
    )
    with pytest.raises(IntegrityError, match="missing"):
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"])


def test_missing_reward_lineage_marks_episode_truncated(canonical_run):
    journal = canonical_run["journal"]
    common = {
        "attempt_id": "attempt-3",
        "candidate_id": "candidate-3",
        "split": "train",
        "visibility": "public",
    }
    append_event(
        journal,
        canonical_run["run_id"],
        "context_packet_created",
        {
            **common,
            "artifacts": [
                artifact_binding("context_packet", canonical_run["packet_receipt"])
            ],
        },
        "attempt-3-context",
    )
    candidate = canonical_run["artifacts"].put_bytes(
        b"def candidate(): pass\n", media_type="text/x-python"
    )
    append_event(
        journal,
        canonical_run["run_id"],
        "candidate_frozen",
        {**common, "artifacts": [artifact_binding("candidate", candidate)]},
        "attempt-3-candidate",
    )
    append_event(
        journal,
        canonical_run["run_id"],
        "decision",
        {**common, "verdict": "revert"},
        "attempt-3-decision",
    )
    append_event(
        journal,
        canonical_run["run_id"],
        "reward_committed",
        {
            **common,
            "policy_id": "kernel_robust_v1",
            "evidence_class": "self_reported",
            "scalar_reward": 999,
        },
        "attempt-3-reward",
    )
    graph = EpisodeGraphMaterializer(
        journal, canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])
    child = next(item for item in graph.children if item.attempt_id == "attempt-3")
    assert child.trainability == "truncated"
    assert "reward_not_measured" in child.validation_reasons
    assert "reward_measurement_receipt_missing" in child.validation_reasons
    assert "reward_policy_receipt_missing" in child.validation_reasons


@pytest.mark.parametrize(
    ("outcome_type", "expected"),
    (("compile_result", "compile_failed"), ("correctness_result", "wrong")),
)
def test_failure_labels_distinguish_compile_and_wrong(
    canonical_run, outcome_type, expected
):
    attempt_id = f"attempt-{expected}"
    candidate = canonical_run["artifacts"].put_bytes(
        b"def candidate(): pass\n", media_type="text/x-python"
    )
    common = {
        "attempt_id": attempt_id,
        "candidate_id": f"candidate-{expected}",
        "split": "validation",
        "visibility": "public",
    }
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "context_packet_created",
        {
            **common,
            "artifacts": [
                artifact_binding("context_packet", canonical_run["packet_receipt"])
            ],
        },
        f"{attempt_id}-context",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "candidate_frozen",
        {**common, "artifacts": [artifact_binding("candidate", candidate)]},
        f"{attempt_id}-candidate",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        outcome_type,
        {**common, "passed": False, "evidence_class": "measured"},
        f"{attempt_id}-outcome",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "decision",
        {**common, "verdict": "revert"},
        f"{attempt_id}-decision",
    )
    graph = EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])
    child = next(item for item in graph.children if item.attempt_id == attempt_id)
    assert child.status == expected
