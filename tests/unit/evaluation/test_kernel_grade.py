from __future__ import annotations

from dataclasses import replace

import pytest

from apex.core import ContractError, TaskStatus
from apex.evaluation import (
    CaseTiming,
    GateVerdict,
    GradeAggregation,
    MeasurementPolicy,
    MeasurementStatus,
    PairedTimingUnit,
    SampleSeries,
    grade_kernel,
    kernel_reward,
)


PASS = GateVerdict(compiled=True, correct=True, integrity_passed=True, tampering_passed=True)


def _series(median: float, tail: float) -> SampleSeries:
    # nearest-rank p99 at N=300 is position 297, so four tail values make
    # the p99 boundary itself part of the synthetic tail.
    return SampleSeries((median,) * 296 + (tail,) * 4)


def _paired_case(
    case_id: str,
    units: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...],
) -> CaseTiming:
    paired = tuple(
        PairedTimingUnit(index, reference, optimized)
        for index, (reference, optimized) in enumerate(units)
    )
    return CaseTiming(
        case_id,
        SampleSeries(tuple(value for unit in paired for value in unit.reference_samples_ms)),
        SampleSeries(tuple(value for unit in paired for value in unit.optimized_samples_ms)),
        paired_units=paired,
    )


def _constant_case(case_id: str, reference: float, optimized: float) -> CaseTiming:
    return _paired_case(
        case_id,
        (
            ((reference,) * 150, (optimized,) * 150),
            ((reference,) * 150, (optimized,) * 150),
        ),
    )


@pytest.mark.parametrize(
    ("speedup", "expected"),
    [
        (0.5, 70.0),
        (0.75, 70.0),
        (0.9, 100.0),
        (1.0, 120.0),
        (1.05, 130.0),
        (1.5, 220.0),
        (2.0, 320.0),
        (3.0, 320.0),
    ],
)
def test_canonical_reward_truth_table(speedup: float, expected: float) -> None:
    assert kernel_reward(PASS, speedup) == pytest.approx(expected)


def test_compile_and_correctness_gates() -> None:
    compile_failed = GateVerdict(False, False, False, False)
    incorrect = GateVerdict(True, False, True, True)

    assert kernel_reward(compile_failed, None) == 0
    assert kernel_reward(incorrect, None) == 20
    assert grade_kernel(compile_failed, []).measurement_status is MeasurementStatus.NOT_RUN_DUE_TO_GATE


def test_tail_regression_limits_grade_even_when_median_improves() -> None:
    grade = grade_kernel(
        PASS,
        [CaseTiming("heavy-tail", reference=_series(10, 20), optimized=_series(8, 40))],
    )

    assert grade.s50 == pytest.approx(1.25)
    assert grade.s99 == pytest.approx(0.5)
    assert grade.srobust == pytest.approx(0.5)
    assert grade.reward == 70


def test_typical_regression_limits_grade_even_when_tail_improves() -> None:
    grade = grade_kernel(
        PASS,
        [CaseTiming("typical", reference=_series(10, 40), optimized=_series(20, 20))],
    )

    assert grade.s50 == pytest.approx(0.5)
    assert grade.s99 == pytest.approx(2.0)
    assert grade.srobust == pytest.approx(0.5)


def test_missing_p99_is_no_measurement_with_null_reward() -> None:
    grade = grade_kernel(
        PASS,
        [CaseTiming("short", SampleSeries((10.0,) * 299), SampleSeries((9.0,) * 299))],
    )

    assert grade.measurement_status is MeasurementStatus.INSUFFICIENT_SAMPLES
    assert grade.task_status is TaskStatus.NO_MEASUREMENT
    assert grade.reward is None
    assert grade.srobust is None


def test_confirmed_safety_finding_blocks_timing_and_reward() -> None:
    gates = GateVerdict(True, True, True, True, safety_finding=True)

    grade = grade_kernel(gates, [CaseTiming("unused", _series(10, 10), _series(9, 9))])

    assert grade.measurement_status is MeasurementStatus.NOT_RUN_DUE_TO_SAFETY
    assert grade.reward is None


def test_workload_weighted_aggregation_uses_case_counts() -> None:
    timings = [
        CaseTiming("hot", _series(10, 10), _series(5, 5), workload_count=9),
        CaseTiming("cold", _series(10, 10), _series(20, 20), workload_count=1),
    ]

    equal = grade_kernel(PASS, timings, aggregation=GradeAggregation.EQUAL_CASE)
    weighted = grade_kernel(PASS, timings, aggregation=GradeAggregation.WORKLOAD_WEIGHTED)

    assert equal.srobust == pytest.approx(1.25)
    assert weighted.srobust == pytest.approx(100 / 65)
    assert weighted.worst_case_srobust == pytest.approx(0.5)


def test_strict_keep_boundary_does_not_change_point_reward() -> None:
    boundary = grade_kernel(PASS, [_constant_case("boundary", 10.0, 10.0 / 1.05)])
    above = grade_kernel(PASS, [_constant_case("above", 10.0, 10.0 / 1.051)])

    assert boundary.srobust == pytest.approx(1.05)
    assert boundary.reward == pytest.approx(130.0)
    assert boundary.threshold_pass is False
    assert boundary.promotion_eligible is False
    assert boundary.promotion_reason_code == "srobust_threshold_not_met"
    assert above.promotion_eligible is True

    with pytest.raises(ContractError) as raised:
        replace(
            boundary,
            task_status=TaskStatus.SUCCEEDED,
            promotion_eligible=True,
            promotion_reason_code="promotion_eligible",
        )
    assert raised.value.reason_code == "invalid_kernel_grade"


def test_high_cv_blocks_promotion_but_not_scalar_reward() -> None:
    reference = (5.0, 15.0) * 75
    optimized = tuple(value / 1.2 for value in reference)
    case = _paired_case("noisy", ((reference, optimized), (reference, optimized)))

    grade = grade_kernel(PASS, [case])

    assert grade.srobust == pytest.approx(1.2)
    assert grade.reward == pytest.approx(160.0)
    assert grade.max_cv is not None and grade.max_cv > 0.10
    assert grade.noise_pass is False
    assert grade.promotion_reason_code == "timing_noise_exceeds_policy"


def test_bootstrap_confidence_lower_bound_blocks_uncertain_point_gain() -> None:
    case = _paired_case(
        "uncertain",
        (
            ((9.0,) * 150, (9.01,) * 150),
            ((10.0,) * 150, (9.0,) * 150),
        ),
    )

    grade = grade_kernel(PASS, [case])

    assert grade.srobust is not None and grade.srobust > 1.05
    assert grade.srobust_ci_lower is not None and grade.srobust_ci_lower < 1.0
    assert grade.confidence_pass is False
    assert grade.promotion_reason_code == "timing_confidence_below_floor"


def test_worst_case_regression_blocks_aggregate_win() -> None:
    grade = grade_kernel(
        PASS,
        [
            _constant_case("large-win", 10.0, 5.0),
            _constant_case("small-loss", 10.0, 10.0 / 0.99),
        ],
    )

    assert grade.srobust is not None and grade.srobust > 1.05
    assert grade.confidence_pass is True
    assert grade.worst_case_srobust == pytest.approx(0.99)
    assert grade.worst_case_pass is False
    assert grade.promotion_reason_code == "worst_case_regression"


def test_missing_paired_units_retains_point_grade_without_confident_keep() -> None:
    grade = grade_kernel(
        PASS,
        [CaseTiming("legacy-math", _series(10.0, 10.0), _series(8.0, 8.0))],
    )

    assert grade.measurement_status is MeasurementStatus.VALID
    assert grade.reward == pytest.approx(170.0)
    assert grade.srobust_ci_lower is None
    assert grade.promotion_eligible is False
    assert grade.promotion_reason_code == "timing_confidence_unavailable"
