from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.evaluation import (
    E2ERewardPolicy,
    E2EVerdict,
    grade_e2e_outcome,
    replay_e2e_reward,
)


def _verdict(*, keep: bool, reason: str, throughput: float) -> E2EVerdict:
    return E2EVerdict(
        keep=keep,
        reason_code=reason,
        throughput_gain_pct=throughput,
        accuracy_regression_pct=0.0,
        ttft_p99_regression_pct=0.0,
        tpot_p99_regression_pct=0.0,
        anchor_measurement_id="anchor-measurement",
        candidate_measurement_id="candidate-measurement",
    )


def test_keep_reward_combines_acceptance_with_clipped_throughput_gain() -> None:
    grade = grade_e2e_outcome(
        verdict="keep",
        reason_code="accepted",
        candidate_present=True,
        measurement_verdict=_verdict(
            keep=True,
            reason="accepted",
            throughput=3.0,
        ),
    )

    assert grade.outcome_class == "accepted"
    assert grade.outcome_base == 100.0
    assert grade.throughput_component == 30.0
    assert grade.scalar_reward == 130.0
    assert replay_e2e_reward(grade.to_dict()) == 130.0


def test_hard_gate_regression_cannot_be_offset_by_throughput() -> None:
    grade = grade_e2e_outcome(
        verdict="revert",
        reason_code="accuracy_regression",
        candidate_present=True,
        measurement_verdict=_verdict(
            keep=False,
            reason="accuracy_regression",
            throughput=8.0,
        ),
    )

    assert grade.outcome_class == "hard_gate_regression"
    assert grade.throughput_component == 0.0
    assert grade.scalar_reward == -100.0


def test_source_free_agent_outcome_has_explicit_smaller_penalty() -> None:
    no_source = grade_e2e_outcome(
        verdict="reject",
        reason_code="agent_made_no_source_change",
        candidate_present=False,
    )
    unusable_source = grade_e2e_outcome(
        verdict="reject",
        reason_code="agent_made_no_source_change",
        candidate_present=True,
    )

    assert no_source.outcome_class == "no_source"
    assert no_source.scalar_reward == -20.0
    assert unusable_source.outcome_class == "candidate_rejected"
    assert unusable_source.scalar_reward == -100.0


def test_measured_decision_requires_evaluator_verdict() -> None:
    with pytest.raises(ContractError) as failure:
        grade_e2e_outcome(
            verdict="keep",
            reason_code="accepted",
            candidate_present=True,
        )

    assert failure.value.reason_code == "missing_e2e_reward_measurement"


def test_replay_rejects_tampered_components_and_policy() -> None:
    grade = grade_e2e_outcome(
        verdict="keep",
        reason_code="accepted",
        candidate_present=True,
        measurement_verdict=_verdict(
            keep=True,
            reason="accepted",
            throughput=1.0,
        ),
    ).to_dict()
    grade["components"]["throughput"] = 99.0

    with pytest.raises(ContractError) as component_failure:
        replay_e2e_reward(grade)
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
