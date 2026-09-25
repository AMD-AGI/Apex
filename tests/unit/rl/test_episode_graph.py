from __future__ import annotations

from dataclasses import replace
import json

import pytest

from apex.core import IntegrityError
from apex.rl import EpisodeArtifact, EpisodeGraphMaterializer, EvidenceClass, SemanticRole
from apex.rl.kernel_measurement_validation import kernel_measurement_evidence_reasons
from apex.storage import ArtifactReceipt

from .conftest import (
    append_event,
    append_event_transaction,
    artifact_binding,
    make_packet,
)


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

    assert graph.parent.kind == "e2e_kernel_only"
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


def test_forged_measurement_execution_role_does_not_authorize_reward(
    canonical_run,
) -> None:
    graph = EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])
    child = graph.children[0]
    forged = canonical_run["artifacts"].put_bytes(
        b'{"looks":"plausible"}', media_type="application/json"
    )
    events = []
    for event in child.events:
        if event.event_type != "reward_committed":
            events.append(event)
            continue
        bindings = tuple(
            EpisodeArtifact("measurement_execution", forged, event.event_id)
            if binding.role == "measurement_execution"
            else binding
            for binding in event.artifacts
        )
        events.append(replace(event, artifacts=bindings))

    reasons = kernel_measurement_evidence_reasons(
        tuple(events), canonical_run["artifacts"]
    )

    assert reasons == {"kernel_measurement_execution_evidence_mismatch"}


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


def test_attempt_lineage_is_never_inferred_from_action_id(canonical_run):
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "action.queued",
        {"action_id": "legacy-attempt", "action_type": "kernel-candidate"},
        "legacy-action-queued",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "action.started",
        {"action_id": "legacy-attempt"},
        "legacy-action-started",
    )

    graph = EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])

    assert "legacy-attempt" not in {child.attempt_id for child in graph.children}


def test_candidate_scoped_event_requires_explicit_attempt_lineage(canonical_run):
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "candidate_frozen",
        {"candidate_id": "candidate-orphan"},
        "candidate-without-attempt",
    )

    with pytest.raises(IntegrityError) as error:
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"])

    assert error.value.reason_code == "attempt_lineage_missing"


def test_conflicting_candidate_id_fails_closed(canonical_run):
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "measurement_result",
        {
            "attempt_id": "attempt-1",
            "candidate_id": "candidate-conflict",
            "evidence_class": "measured",
        },
        "attempt-1-conflicting-candidate",
    )

    with pytest.raises(IntegrityError) as error:
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"])

    assert error.value.reason_code == "candidate_id_mismatch"


def test_candidate_id_cannot_be_reused_by_another_attempt(canonical_run):
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "observation_created",
        {"attempt_id": "attempt-other", "candidate_id": "candidate-1"},
        "other-attempt-same-candidate",
    )

    with pytest.raises(IntegrityError) as error:
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"])

    assert error.value.reason_code == "candidate_id_mismatch"


def test_no_source_e2e_reject_is_complete_and_rewarded(e2e_no_source_run):
    graph = EpisodeGraphMaterializer(
        e2e_no_source_run["journal"], e2e_no_source_run["artifacts"]
    ).materialize(e2e_no_source_run["run_id"])

    assert len(graph.children) == 1
    child = graph.children[0]
    assert child.attempt_id == e2e_no_source_run["attempt_id"]
    assert child.candidate_id is None
    assert child.state_generation == 1
    assert child.verdict == "reject"
    assert child.scalar_reward == 0.0
    assert child.policy_ids == ("e2e_throughput_qos_v1",)
    assert child.status == "no_gain"
    assert child.trainability == "complete"
    outcome_events = tuple(
        event
        for event in child.events
        if event.event_type in {"e2e.candidate_decided", "reward_committed"}
    )
    assert len(outcome_events) == 2
    assert {
        event.transaction_id for event in outcome_events
    } == {e2e_no_source_run["outcome_transaction_id"]}
    assert all(event.to_dict()["transaction_id"] for event in outcome_events)


def _append_e2e_attempt_prefix(run, attempt_id: str) -> dict[str, object]:
    common = {
        "attempt_id": attempt_id,
        "opportunity_id": "opportunity-1",
        "anchor_generation": 0,
        "split": "train",
        "visibility": "public",
    }
    append_event(
        run["journal"],
        run["run_id"],
        "context_packet_created",
        {
            **common,
            "artifacts": [artifact_binding("context_packet", run["packet_receipt"])],
        },
        f"{attempt_id}-context",
    )
    append_event(
        run["journal"],
        run["run_id"],
        "candidate_frozen",
        {
            **common,
            "candidate_id": None,
            "artifacts": [artifact_binding("candidate_manifest", run["manifest"])],
        },
        f"{attempt_id}-candidate",
    )
    return common


def _e2e_outcome_inputs(run, attempt_id: str):
    common = _append_e2e_attempt_prefix(run, attempt_id)
    reason = "agent_made_no_source_change"
    decision = {
        **common,
        "receipt": run["decision"].digest,
        "verdict": "reject",
        "reason": reason,
        "artifacts": [artifact_binding("decision_evidence", run["decision"])],
    }
    grade = run["grade"]
    reward = {
        **common,
        "verdict": "reject",
        "reason_code": reason,
        "policy_id": grade.policy_id,
        "policy_digest": grade.policy_digest,
        "scalar_reward": grade.scalar_reward,
        "reward_vector": grade.to_dict(),
        "evidence_class": "derived",
        "artifacts": [
            artifact_binding("decision_evidence", run["decision"]),
            artifact_binding("e2e_reward_vector", run["grade_receipt"]),
            artifact_binding("reward_policy", run["policy"]),
            artifact_binding("candidate_manifest", run["manifest"]),
        ],
    }
    return decision, reward


def _child_for(run, attempt_id: str):
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )
    return next(item for item in graph.children if item.attempt_id == attempt_id)


def test_independently_appended_e2e_outcomes_are_truncated(e2e_no_source_run):
    attempt_id = "attempt-independent-outcomes"
    decision, reward = _e2e_outcome_inputs(e2e_no_source_run, attempt_id)
    append_event(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        "e2e.candidate_decided",
        decision,
        f"{attempt_id}-decision",
    )
    append_event(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        "reward_committed",
        reward,
        f"{attempt_id}-reward",
    )

    child = _child_for(e2e_no_source_run, attempt_id)

    assert child.trainability == "truncated"
    assert "e2e_outcome_transaction_mismatch" in child.validation_reasons
    assert "e2e_outcome_transaction_shape_invalid" in child.validation_reasons


def test_e2e_outcomes_in_different_transactions_are_truncated(e2e_no_source_run):
    attempt_id = "attempt-split-transactions"
    decision, reward = _e2e_outcome_inputs(e2e_no_source_run, attempt_id)
    append_event_transaction(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        (
            ("e2e.candidate_decided", decision, f"{attempt_id}-decision"),
            ("usage_recorded", {"attempt_id": attempt_id}, f"{attempt_id}-usage"),
        ),
    )
    append_event_transaction(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        (
            ("reward_committed", reward, f"{attempt_id}-reward"),
            ("cost_recorded", {"attempt_id": attempt_id}, f"{attempt_id}-cost"),
        ),
    )

    child = _child_for(e2e_no_source_run, attempt_id)

    assert child.trainability == "truncated"
    assert "e2e_outcome_transaction_mismatch" in child.validation_reasons
    assert "e2e_outcome_transaction_shape_invalid" in child.validation_reasons


def test_extra_event_in_e2e_outcome_transaction_is_truncated(e2e_no_source_run):
    attempt_id = "attempt-extra-transaction-event"
    decision, reward = _e2e_outcome_inputs(e2e_no_source_run, attempt_id)
    append_event_transaction(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        (
            ("e2e.candidate_decided", decision, f"{attempt_id}-decision"),
            ("reward_committed", reward, f"{attempt_id}-reward"),
            ("usage_recorded", {"attempt_id": attempt_id}, f"{attempt_id}-usage"),
        ),
    )

    child = _child_for(e2e_no_source_run, attempt_id)

    assert child.trainability == "truncated"
    assert "e2e_outcome_transaction_mismatch" not in child.validation_reasons
    assert "e2e_outcome_transaction_shape_invalid" in child.validation_reasons


def test_terminal_e2e_attempt_requires_exactly_one_decision(e2e_no_source_run):
    append_event(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        "e2e.candidate_decided",
        {
            "attempt_id": e2e_no_source_run["attempt_id"],
            "opportunity_id": "opportunity-1",
            "receipt": e2e_no_source_run["decision"].digest,
            "verdict": "reject",
            "reason": "agent_made_no_source_change",
        },
        "duplicate-attempt-decision",
    )

    graph = EpisodeGraphMaterializer(
        e2e_no_source_run["journal"], e2e_no_source_run["artifacts"]
    ).materialize(e2e_no_source_run["run_id"])

    child = graph.children[0]
    assert child.trainability == "truncated"
    assert "multiple_decision_events" in child.validation_reasons


def test_terminal_noninfra_e2e_attempt_requires_reward(e2e_no_source_run):
    attempt_id = "attempt-without-reward"
    common = {
        "attempt_id": attempt_id,
        "opportunity_id": "opportunity-1",
        "anchor_generation": 0,
        "split": "train",
        "visibility": "public",
    }
    append_event(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        "e2e.opportunity_selected",
        {
            **common,
            "context_packet_id": e2e_no_source_run["packet"].context_packet_id,
        },
        "unrewarded-selected",
    )
    append_event(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        "context_packet_created",
        {
            **common,
            "artifacts": [
                artifact_binding(
                    "context_packet",
                    e2e_no_source_run["packet_receipt"],
                )
            ],
        },
        "unrewarded-context",
    )
    append_event(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        "candidate_frozen",
        {
            **common,
            "candidate_id": None,
            "artifacts": [
                artifact_binding("candidate_manifest", e2e_no_source_run["manifest"])
            ],
        },
        "unrewarded-candidate",
    )
    append_event(
        e2e_no_source_run["journal"],
        e2e_no_source_run["run_id"],
        "e2e.candidate_decided",
        {
            **common,
            "receipt": e2e_no_source_run["decision"].digest,
            "verdict": "reject",
            "reason": "agent_made_no_source_change",
        },
        "unrewarded-decision",
    )

    graph = EpisodeGraphMaterializer(
        e2e_no_source_run["journal"], e2e_no_source_run["artifacts"]
    ).materialize(e2e_no_source_run["run_id"])

    child = next(item for item in graph.children if item.attempt_id == attempt_id)
    assert child.trainability == "truncated"
    assert "reward_missing" in child.validation_reasons


def test_rejects_state_not_anchored_to_journal(canonical_run):
    wrong = replace(canonical_run["state"], last_event_id="event-does-not-exist")
    with pytest.raises(IntegrityError, match="head"):
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"], workload_state=wrong)


def test_supplied_state_must_match_full_canonical_replay(canonical_run):
    forged = replace(canonical_run["state"], accepted_patch_ids=("patch-forged",))

    with pytest.raises(IntegrityError) as error:
        EpisodeGraphMaterializer(
            canonical_run["journal"], canonical_run["artifacts"]
        ).materialize(canonical_run["run_id"], workload_state=forged)

    assert error.value.reason_code == "state_projection_mismatch"


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
    assert "kernel_reward_stage_invalid" in child.validation_reasons
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
