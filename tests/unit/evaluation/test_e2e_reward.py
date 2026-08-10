from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.evaluation import (
    E2EPairedEstimate,
    E2EPairedVerdict,
    E2ERewardPolicy,
    grade_e2e_outcome,
    replay_e2e_reward,
)


def _verdict(
    *,
    keep: bool,
    reason: str,
    throughput: float,
    accuracy: float = 0.0,
    ttft: float = 0.0,
    tpot: float = 0.0,
    measurement_id: str = "a" * 64,
) -> E2EPairedVerdict:
    throughput_ratio = 1.0 + throughput / 100.0
    ttft_ratio = 1.0 / (1.0 + ttft / 100.0)
    tpot_ratio = 1.0 / (1.0 + tpot / 100.0)
    estimate = E2EPairedEstimate(
        throughput_ratio,
        ttft_ratio,
        tpot_ratio,
        throughput,
        accuracy,
        ttft,
        tpot,
        throughput,
        0.5,
        keep,
        3,
        6,
    )
    return E2EPairedVerdict(keep, reason, estimate, measurement_id)


def _grade(verdict: E2EPairedVerdict):
    return grade_e2e_outcome(
        verdict="keep" if verdict.keep else "revert",
        reason_code=verdict.reason_code,
        candidate_present=True,
        measurement_verdict=verdict,
    )


def test_ten_percent_throughput_gain_and_unchanged_latency_is_136() -> None:
    vector = _grade(_verdict(keep=True, reason="accepted", throughput=10.0))

    assert vector.policy_id == "e2e_throughput_qos_v1"
    assert vector.outcome_class == "accepted"
    assert vector.runtime_component == 20.0
    assert vector.eligible_base_component == 100.0
    assert vector.throughput_component == pytest.approx(16.0)
    assert vector.ttft_p99_component == 0.0
    assert vector.tpot_p99_component == 0.0
    assert vector.scalar_reward == pytest.approx(136.0)
    assert replay_e2e_reward(vector.to_dict()) == pytest.approx(136.0)


def test_unchanged_eligible_runtime_is_120_even_when_not_promoted() -> None:
    vector = _grade(
        _verdict(
            keep=False,
            reason="insufficient_throughput_gain",
            throughput=0.0,
        )
    )

    assert vector.outcome_class == "no_gain"
    assert vector.eligible is True
    assert vector.scalar_reward == 120.0


def test_terminal_scope_uses_same_formula_without_summing_attempts() -> None:
    verdict = _verdict(keep=True, reason="accepted", throughput=10.0)

    vector = grade_e2e_outcome(
        verdict="keep",
        reason_code="accepted",
        candidate_present=True,
        measurement_verdict=verdict,
        scope="task_terminal",
    )

    assert vector.scope == "task_terminal"
    assert vector.scalar_reward == pytest.approx(136.0)
    assert replay_e2e_reward(vector.to_dict()) == pytest.approx(136.0)


def test_terminal_noop_without_selected_candidate_is_120() -> None:
    verdict = _verdict(
        keep=False,
        reason="insufficient_throughput_gain",
        throughput=0.0,
    )
    vector = grade_e2e_outcome(
        verdict="revert",
        reason_code=verdict.reason_code,
        candidate_present=False,
        measurement_verdict=verdict,
        scope="task_terminal",
    )

    assert vector.scalar_reward == 120.0
    assert replay_e2e_reward(vector.to_dict()) == 120.0


@pytest.mark.parametrize(
    ("reason", "accuracy", "ttft", "tpot"),
    (
        ("accuracy_regression", 0.01, 0.0, 0.0),
        ("ttft_p99_regression", 0.0, 5.0001, 0.0),
        ("tpot_p99_regression", 0.0, 0.0, 2.0001),
    ),
)
def test_hard_gate_is_20_and_cannot_be_offset_by_throughput(
    reason: str, accuracy: float, ttft: float, tpot: float
) -> None:
    vector = _grade(
        _verdict(
            keep=False,
            reason=reason,
            throughput=100.0,
            accuracy=accuracy,
            ttft=ttft,
            tpot=tpot,
        )
    )

    assert vector.outcome_class == "hard_gate_regression"
    assert vector.runtime_verified is True
    assert vector.eligible is False
    assert vector.scalar_reward == 20.0


def test_trusted_quality_failure_is_20_with_performance_skipped() -> None:
    vector = grade_e2e_outcome(
        verdict="revert",
        reason_code="quality_gate_failed",
        candidate_present=True,
        performance_skipped="quality_gate",
    )

    assert vector.outcome_class == "hard_gate_regression"
    assert vector.runtime_verified is True
    assert vector.eligible is False
    assert vector.performance_skipped == "quality_gate"
    assert vector.throughput_ratio is None
    assert vector.scalar_reward == 20.0
    assert replay_e2e_reward(vector.to_dict()) == 20.0


def test_quality_failure_cannot_claim_runtime_without_skip_semantics() -> None:
    with pytest.raises(ContractError) as failure:
        grade_e2e_outcome(
            verdict="revert",
            reason_code="quality_gate_failed",
            candidate_present=True,
        )

    assert failure.value.reason_code == "missing_e2e_reward_measurement"


def test_pre_runtime_rejects_are_zero_not_negative_training_penalties() -> None:
    no_source = grade_e2e_outcome(
        verdict="reject",
        reason_code="agent_made_no_source_change",
        candidate_present=False,
    )
    candidate = grade_e2e_outcome(
        verdict="reject",
        reason_code="correctness_failed",
        candidate_present=True,
    )

    assert no_source.outcome_class == "no_source"
    assert candidate.outcome_class == "candidate_rejected"
    assert no_source.scalar_reward == candidate.scalar_reward == 0.0
    assert no_source.runtime_verified is candidate.runtime_verified is False


def test_latency_improvements_have_separate_ten_percent_components() -> None:
    vector = _grade(
        _verdict(
            keep=True,
            reason="accepted",
            throughput=0.5,
            ttft=-50.0,
            tpot=-50.0,
        )
    )

    assert vector.throughput_ratio == pytest.approx(1.005)
    assert vector.ttft_p99_ratio == 2.0
    assert vector.tpot_p99_ratio == 2.0
    assert vector.throughput_component == pytest.approx(0.8)
    assert vector.ttft_p99_component == 20.0
    assert vector.tpot_p99_component == 20.0
    assert vector.scalar_reward == pytest.approx(160.8)


def test_eligible_upper_and_clipped_lower_are_bounded() -> None:
    upper = _grade(
        _verdict(
            keep=True,
            reason="accepted",
            throughput=300.0,
            ttft=-75.0,
            tpot=-75.0,
        )
    )
    lower = _grade(
        _verdict(
            keep=False,
            reason="insufficient_throughput_gain",
            throughput=-75.0,
            ttft=300.0,
            tpot=300.0,
        )
    )

    assert upper.scalar_reward == 320.0
    assert lower.scalar_reward == 70.0


def test_measured_decision_requires_evaluator_verdict() -> None:
    with pytest.raises(ContractError) as failure:
        grade_e2e_outcome(
            verdict="keep",
            reason_code="accepted",
            candidate_present=True,
        )

    assert failure.value.reason_code == "missing_e2e_reward_measurement"


def test_replay_rejects_tampered_components_and_policy() -> None:
    vector = _grade(_verdict(keep=True, reason="accepted", throughput=1.0)).to_dict()
    vector["components"]["total_token_throughput"] = 99.0

    with pytest.raises(ContractError) as component_failure:
        replay_e2e_reward(vector)
    assert component_failure.value.reason_code == "e2e_reward_replay_mismatch"

    clean = grade_e2e_outcome(
        verdict="reject",
        reason_code="correctness_failed",
        candidate_present=True,
    ).to_dict()
    clean["policy_digest"] = "0" * 64
    with pytest.raises(ContractError) as policy_failure:
        replay_e2e_reward(clean, policy=E2ERewardPolicy())
    assert policy_failure.value.reason_code == "e2e_reward_policy_mismatch"


def test_replay_rejects_tampered_gate_and_metric_semantics() -> None:
    vector = _grade(
        _verdict(keep=True, reason="accepted", throughput=10.0)
    ).to_dict()
    vector["eligible"] = False
    vector["components"] = {
        "runtime": 20.0,
        "eligible_base": 0.0,
        "total_token_throughput": 0.0,
        "ttft_p99": 0.0,
        "tpot_p99": 0.0,
        "ge2e": None,
    }
    vector["scalar_reward"] = 20.0

    with pytest.raises(ContractError) as gate_failure:
        replay_e2e_reward(vector)
    assert gate_failure.value.reason_code == "e2e_reward_replay_mismatch"

    clean = _grade(_verdict(keep=True, reason="accepted", throughput=10.0)).to_dict()
    clean["metrics"]["throughput_gain_pct"] = 50.0
    with pytest.raises(ContractError) as metric_failure:
        replay_e2e_reward(clean)
    assert metric_failure.value.reason_code == "e2e_reward_replay_mismatch"
