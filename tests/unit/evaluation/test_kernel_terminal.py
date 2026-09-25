from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.evaluation import (
    failed_kernel_terminal_grade,
    load_kernel_terminal_grade,
    no_op_kernel_terminal_grade,
    untrainable_kernel_terminal_grade,
)


def test_terminal_grade_round_trips_exact_policy_vector() -> None:
    grade = no_op_kernel_terminal_grade("attempt-test")

    assert load_kernel_terminal_grade(grade.to_dict()) == grade
    assert grade.scalar_reward == 120.0


@pytest.mark.parametrize(
    ("correctness_reached", "reward"),
    [(False, 0.0), (True, 20.0)],
)
def test_gate_terminal_grades_follow_kernel_formula(
    correctness_reached: bool, reward: float
) -> None:
    grade = failed_kernel_terminal_grade(
        "attempt-test", correctness_reached=correctness_reached
    )

    assert load_kernel_terminal_grade(grade.to_dict()).scalar_reward == reward


def test_untrainable_terminal_has_explicit_reward_null() -> None:
    grade = untrainable_kernel_terminal_grade("measurement_environment_invalid")

    assert load_kernel_terminal_grade(grade.to_dict()) == grade
    assert grade.scalar_reward is None
    assert grade.source_attempt_id is None


@pytest.mark.parametrize("field", ["policy_digest", "scalar_reward", "srobust"])
def test_terminal_loader_rejects_tampered_vector(field: str) -> None:
    value = no_op_kernel_terminal_grade("attempt-test").to_dict()
    value[field] = "0" * 64 if field == "policy_digest" else 999.0

    with pytest.raises(ContractError, match="terminal"):
        load_kernel_terminal_grade(value)
