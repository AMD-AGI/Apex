"""Select the sole task-terminal kernel grade without aggregating attempts."""

from __future__ import annotations

from apex.evaluation import (
    KernelTerminalGrade,
    failed_kernel_terminal_grade,
    no_op_kernel_terminal_grade,
    selected_kernel_terminal_grade,
    untrainable_kernel_terminal_grade,
)

from .attempts import KernelAttemptOutcome


def derive_kernel_terminal_grade(
    outcomes: tuple[KernelAttemptOutcome, ...],
    selected: KernelAttemptOutcome | None,
) -> KernelTerminalGrade:
    if selected is not None and selected.measurement is not None:
        measurement = selected.measurement
        if measurement.reward_eligible and measurement.grade.reward is not None:
            return selected_kernel_terminal_grade(
                selected.attempt_id, measurement.grade
            )
    if selected is not None:
        return untrainable_kernel_terminal_grade("external_evaluation_pending")
    measured = tuple(
        outcome
        for outcome in outcomes
        if outcome.measurement is not None
        and outcome.measurement.reward_eligible
        and outcome.measurement.grade.reward is not None
    )
    if measured:
        source = max(measured, key=lambda item: item.rank)
        return no_op_kernel_terminal_grade(source.attempt_id)
    gate_failures = tuple(
        outcome
        for outcome in outcomes
        if outcome.reason_code in {"compile_failed", "correctness_failed"}
    )
    if gate_failures and len(gate_failures) == len(outcomes):
        correctness = next(
            (
                item
                for item in gate_failures
                if item.reason_code == "correctness_failed"
            ),
            None,
        )
        source = correctness or gate_failures[0]
        return failed_kernel_terminal_grade(
            source.attempt_id,
            correctness_reached=correctness is not None,
        )
    return untrainable_kernel_terminal_grade("terminal_measurement_unavailable")


__all__ = ["derive_kernel_terminal_grade"]
