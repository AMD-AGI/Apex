from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import StateTransitionError
from apex.orchestration import RunController, RunPhase, SearchStage
from apex.storage import EventJournal, SnapshotStore


DIGEST = "a" * 64


def _controller(tmp_path: Path) -> RunController:
    controller = RunController.create(
        "e2e-run", EventJournal(tmp_path / "events.db"), SnapshotStore(tmp_path / "state.json")
    )
    controller.initialize_e2e(
        workload_id="qwen3-next",
        provenance_hash=DIGEST,
        objective_policy_hash="b" * 64,
        accuracy_contract_hash="c" * 64,
        measurement_protocol_hash="d" * 64,
        candidate_limit=3,
        cycle_limit=3,
    )
    return controller


def _through_diagnostics(controller: RunController) -> None:
    controller.commit_e2e_baseline(
        receipt="baseline-receipt",
        metrics={"throughput": 100.0, "ttft_p99_ms": 10.0, "tpot_p99_ms": 2.0},
        quality_passed=True,
    )
    controller.commit_e2e_diagnostics(
        receipt="diagnostic-receipt", opportunity_ids=("opportunity-1", "opportunity-2")
    )


def test_rejected_candidate_is_remembered_and_finalized_as_no_gain(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _through_diagnostics(controller)
    controller.select_e2e_opportunity(
        opportunity_id="opportunity-1", context_packet_id="packet-receipt"
    )
    controller.freeze_e2e_candidate(candidate_id="candidate-1", artifact_ref="candidate-source")
    controller.commit_e2e_micro_verification(
        candidate_id="candidate-1",
        receipt="micro-receipt",
        qualified=False,
        reason="correctness_failed",
    )

    assert controller.state.e2e is not None
    assert controller.state.e2e.stage is SearchStage.UPDATING
    assert controller.state.e2e.decisions[-1].verdict == "reject"
    controller.complete_e2e_update(stop=True, reason="no_gain")
    controller.commit_e2e_final(receipt="final-baseline-replay")
    controller.finish(RunPhase.SUCCEEDED, reason="no_gain")
    assert controller.state.phase is RunPhase.SUCCEEDED
    assert controller.state.anchor_generation == 0
    assert controller.state.accepted_patch_ids == ()


def test_keep_advances_only_current_live_anchor_then_forces_reprofile(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _through_diagnostics(controller)
    controller.select_e2e_opportunity(
        opportunity_id="opportunity-1", context_packet_id="packet-receipt"
    )
    controller.freeze_e2e_candidate(candidate_id="candidate-1", artifact_ref="candidate-source")
    controller.commit_e2e_micro_verification(
        candidate_id="candidate-1", receipt="micro-receipt", qualified=True
    )
    controller.commit_e2e_safety_verification(
        candidate_id="candidate-1", receipt="safety-receipt", finding=False
    )
    controller.commit_e2e_delivery_verification(
        candidate_id="candidate-1", receipt="deployment-receipt"
    )
    controller.decide_e2e_candidate(
        candidate_id="candidate-1",
        receipt="e2e-ab-receipt",
        verdict="keep",
        reason="throughput_improved",
        new_anchor_id="anchor-1",
        accepted_patch_id="patch-1",
    )

    assert controller.state.anchor_id == "anchor-1"
    assert controller.state.anchor_generation == 1
    assert controller.state.e2e is not None
    assert controller.state.e2e.stage is SearchStage.REPROFILING
    controller.commit_e2e_reprofile(
        receipt="diagnostic-anchor-1", opportunity_ids=("opportunity-2",)
    )
    assert controller.state.e2e.bottleneck_generation == 2
    controller.complete_e2e_update(stop=True, reason="target_reached")
    assert controller.state.e2e.stage is SearchStage.FINALIZING


def test_baseline_quality_failure_cannot_enter_diagnosis(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.commit_e2e_baseline(
        receipt="baseline-receipt", metrics={"throughput": 100.0}, quality_passed=False
    )
    assert controller.state.e2e is not None
    assert controller.state.e2e.stage is SearchStage.FINALIZING
    assert controller.state.e2e.exit_reason == "baseline_invalid"
    with pytest.raises(StateTransitionError) as failure:
        controller.commit_e2e_diagnostics(receipt="bad", opportunity_ids=("opportunity-1",))
    assert failure.value.reason_code == "illegal_e2e_transition"


def test_crash_recovery_rebuilds_active_candidate_without_agent_session(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _through_diagnostics(controller)
    controller.select_e2e_opportunity(
        opportunity_id="opportunity-1", context_packet_id="packet-receipt"
    )
    controller.freeze_e2e_candidate(candidate_id="candidate-1", artifact_ref="candidate-source")
    expected = controller.state

    SnapshotStore(tmp_path / "state.json").delete()
    recovered = RunController.recover(
        "e2e-run", EventJournal(tmp_path / "events.db"), SnapshotStore(tmp_path / "state.json")
    )
    assert recovered.state == expected
    assert recovered.state.e2e is not None
    assert recovered.state.e2e.context_packet_id == "packet-receipt"
    assert recovered.state.e2e.active_candidate_id == "candidate-1"
    assert recovered.state.e2e.stage is SearchStage.MICRO_VERIFYING


def test_observation_action_commit_does_not_mutate_anchor(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.queue_action("baseline-action", "benchmark")
    controller.start_action("baseline-action")
    controller.mark_artifacts_ready("baseline-action", ("report-receipt",))
    controller.verify_action("baseline-action", "report-verified")
    controller.complete_action("baseline-action")
    assert controller.state.anchor_id == "anchor-0"
    assert controller.state.anchor_generation == 0
    assert controller.state.action_history[-1].result_anchor_id is None


def test_accepted_patch_cannot_succeed_without_second_clean_replay(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    _through_diagnostics(controller)
    controller.select_e2e_opportunity(
        opportunity_id="opportunity-1", context_packet_id="packet-receipt"
    )
    controller.freeze_e2e_candidate(candidate_id="candidate-1", artifact_ref="source")
    controller.commit_e2e_micro_verification(
        candidate_id="candidate-1", receipt="micro", qualified=True
    )
    controller.commit_e2e_safety_verification(
        candidate_id="candidate-1", receipt="safety", finding=False
    )
    controller.commit_e2e_delivery_verification(
        candidate_id="candidate-1", receipt="delivery"
    )
    controller.decide_e2e_candidate(
        candidate_id="candidate-1",
        receipt="decision",
        verdict="keep",
        reason="accepted",
        new_anchor_id="anchor-1",
        accepted_patch_id="patch-1",
    )
    controller.commit_e2e_reprofile(receipt="diagnostic-2", opportunity_ids=())
    controller.complete_e2e_update(stop=True, reason="budget_exhausted")
    controller.commit_e2e_final(receipt="final", clean_replay_verified=False)

    with pytest.raises(StateTransitionError) as failure:
        controller.finish(RunPhase.SUCCEEDED, reason="succeeded")

    assert failure.value.reason_code == "second_clean_replay_required"
