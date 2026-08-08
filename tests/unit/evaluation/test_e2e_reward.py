from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.evaluation import (
    E2ERewardPolicy,
    E2EVerdict,
    e2e_comparison_selection_policy,
    grade_e2e_outcome,
    replay_e2e_reward,
    select_conservative_e2e_verdict,
)


def _verdict(
    *,
    keep: bool,
    reason: str,
    throughput: float,
    accuracy: float = 0.0,
    ttft: float = 0.0,
    tpot: float = 0.0,
    anchor_id: str = "anchor-measurement",
    candidate_id: str = "candidate-measurement",
) -> E2EVerdict:
    return E2EVerdict(
        keep=keep,
        reason_code=reason,
        throughput_gain_pct=throughput,
        accuracy_regression_pct=accuracy,
        ttft_p99_regression_pct=ttft,
        tpot_p99_regression_pct=tpot,
        anchor_measurement_id=anchor_id,
        candidate_measurement_id=candidate_id,
    )


def _selected(comparisons: tuple[E2EVerdict, ...]) -> E2EVerdict:
    return comparisons[select_conservative_e2e_verdict(comparisons)]


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


def test_comparison_selection_is_conservative_and_permutation_invariant() -> None:
    no_gain = _verdict(
        keep=False,
        reason="insufficient_throughput_gain",
        throughput=-1.0220958655,
        accuracy=-0.3187,
        tpot=1.1461,
        anchor_id="anchor-a",
        candidate_id="candidate-a",
    )
    hard_gate = _verdict(
        keep=False,
        reason="accuracy_regression",
        throughput=-2.2649212451,
        accuracy=0.5560,
        anchor_id="anchor-b",
        candidate_id="candidate-b",
    )

    for comparisons in ((no_gain, hard_gate), (hard_gate, no_gain)):
        selected = _selected(comparisons)
        grade = grade_e2e_outcome(
            verdict="revert",
            reason_code=selected.reason_code,
            candidate_present=True,
            measurement_verdict=selected,
        )
        assert selected is hard_gate
        assert grade.scalar_reward == pytest.approx(-122.649212451)


def test_comparison_selection_never_hides_a_failure() -> None:
    passing = _verdict(keep=True, reason="accepted", throughput=0.6)
    failing = _verdict(
        keep=False,
        reason="ttft_p99_regression",
        throughput=8.0,
        ttft=5.1,
    )

    assert _selected((passing, failing)) is failing
    assert _selected((failing, passing)) is failing


def test_passing_comparisons_select_the_lower_throughput_gain() -> None:
    lower = _verdict(
        keep=True,
        reason="accepted",
        throughput=0.6,
        anchor_id="anchor-low",
    )
    higher = _verdict(
        keep=True,
        reason="accepted",
        throughput=2.0,
        anchor_id="anchor-high",
    )

    assert _selected((higher, lower)) is lower
    assert _selected((lower, higher)) is lower


def test_comparison_selection_has_stable_semantic_tie_breakers() -> None:
    worse_gain = _verdict(
        keep=False,
        reason="accuracy_regression",
        throughput=0.5,
        accuracy=0.1,
        anchor_id="anchor-z",
    )
    better_gain = _verdict(
        keep=False,
        reason="accuracy_regression",
        throughput=1.5,
        accuracy=0.2,
        anchor_id="anchor-a",
    )
    id_a = _verdict(
        keep=False,
        reason="accuracy_regression",
        throughput=1.0,
        accuracy=0.2,
        anchor_id="anchor-a",
    )
    id_z = _verdict(
        keep=False,
        reason="accuracy_regression",
        throughput=1.0,
        accuracy=0.2,
        anchor_id="anchor-z",
    )

    assert _selected((better_gain, worse_gain)) is worse_gain
    assert _selected((worse_gain, better_gain)) is worse_gain
    assert _selected((id_z, id_a)) is id_a
    assert _selected((id_a, id_z)) is id_a


def test_comparison_selection_policy_is_explicit_and_replayable() -> None:
    document = e2e_comparison_selection_policy()

    assert document["policy_id"] == "conservative_e2e_reward_v1"
    assert document["reward_policy_digest"] == E2ERewardPolicy().digest
    assert document["ordering"][0] == "failure_before_keep"


def test_empty_comparison_set_fails_closed() -> None:
    with pytest.raises(ContractError) as failure:
        select_conservative_e2e_verdict(())

    assert failure.value.reason_code == "invalid_e2e_comparison_set"
