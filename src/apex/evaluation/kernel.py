"""Canonical robust p50/p99 kernel grade and correctness gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from apex.core import ContractError, TaskStatus, sha256_json

from .statistics import (
    MeasurementPolicy,
    MeasurementStatus,
    PairedTimingUnit,
    BootstrapDistribution,
    Quantiles,
    SampleSeries,
    bootstrap_interval,
    coefficient_of_variation,
    paired_block_bootstrap,
    quantiles,
)


class GradeAggregation(str, Enum):
    EQUAL_CASE = "equal_case"
    WORKLOAD_WEIGHTED = "workload_weighted"


@dataclass(frozen=True, slots=True)
class GateVerdict:
    compiled: bool
    correct: bool
    integrity_passed: bool
    tampering_passed: bool
    safety_finding: bool = False

    @property
    def correctness_gate_passed(self) -> bool:
        return self.compiled and self.correct and self.integrity_passed and self.tampering_passed


@dataclass(frozen=True, slots=True)
class CaseTiming:
    case_id: str
    reference: SampleSeries
    optimized: SampleSeries
    workload_count: int = 1
    paired_units: tuple[PairedTimingUnit, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or self.workload_count <= 0:
            raise ContractError("case identity/count is invalid", "invalid_case")
        if self.reference.method_identity != self.optimized.method_identity:
            raise ContractError(
                "Reference and optimized timing methods differ",
                "measurement_method_mismatch",
            )
        if len(self.reference.values_ms) != len(self.optimized.values_ms):
            raise ContractError(
                "Reference and optimized sample counts differ",
                "measurement_method_mismatch",
            )
        if self.paired_units:
            if len({unit.unit_id for unit in self.paired_units}) != len(self.paired_units):
                raise ContractError("Paired timing unit IDs differ", "invalid_paired_timing_unit")
            reference = tuple(
                sample for unit in self.paired_units for sample in unit.reference_samples_ms
            )
            optimized = tuple(
                sample for unit in self.paired_units for sample in unit.optimized_samples_ms
            )
            if reference != self.reference.values_ms or optimized != self.optimized.values_ms:
                raise ContractError(
                    "Paired timing units do not bind the raw series",
                    "invalid_paired_timing_unit",
                )


@dataclass(frozen=True, slots=True)
class CaseGrade:
    case_id: str
    reference: Quantiles
    optimized: Quantiles
    s50: float
    s99: float
    srobust: float
    workload_count: int
    reference_cv: float
    optimized_cv: float
    s50_ci_lower: float | None
    s50_ci_upper: float | None
    s99_ci_lower: float | None
    s99_ci_upper: float | None
    srobust_ci_lower: float | None
    srobust_ci_upper: float | None
    bootstrap_unit_count: int
    bootstrap_repetitions: int


@dataclass(frozen=True, slots=True)
class _CaseComputation:
    grade: CaseGrade
    bootstrap: BootstrapDistribution | None


@dataclass(frozen=True, slots=True)
class KernelGrade:
    policy_id: str
    measurement_status: MeasurementStatus
    task_status: TaskStatus
    gates: GateVerdict
    cases: tuple[CaseGrade, ...]
    aggregation: GradeAggregation
    s50: float | None
    s99: float | None
    srobust: float | None
    worst_case_srobust: float | None
    reward: float | None
    max_cv: float | None
    s50_ci_lower: float | None
    s50_ci_upper: float | None
    s99_ci_lower: float | None
    s99_ci_upper: float | None
    srobust_ci_lower: float | None
    srobust_ci_upper: float | None
    confidence_level: float
    bootstrap_seed: int
    bootstrap_repetitions: int
    min_bootstrap_units: int
    keep_srobust_threshold: float
    confidence_srobust_floor: float
    worst_case_srobust_floor: float
    max_cv_threshold: float
    threshold_pass: bool
    confidence_pass: bool
    noise_pass: bool
    worst_case_pass: bool
    promotion_eligible: bool
    promotion_reason_code: str
    reward_bounds: tuple[float, float] = (0.0, 320.0)
    reason_code: str | None = None

    def __post_init__(self) -> None:
        _validate_kernel_grade(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "measurement_status": self.measurement_status.value,
            "task_status": self.task_status.value,
            "gates": {
                "compiled": self.gates.compiled,
                "correct": self.gates.correct,
                "integrity_passed": self.gates.integrity_passed,
                "tampering_passed": self.gates.tampering_passed,
                "safety_finding": self.gates.safety_finding,
            },
            "cases": [
                {
                    "case_id": case.case_id,
                    "reference": _quantiles_dict(case.reference),
                    "optimized": _quantiles_dict(case.optimized),
                    "s50": case.s50,
                    "s99": case.s99,
                    "srobust": case.srobust,
                    "workload_count": case.workload_count,
                    "reference_cv": case.reference_cv,
                    "optimized_cv": case.optimized_cv,
                    "s50_ci_lower": case.s50_ci_lower,
                    "s50_ci_upper": case.s50_ci_upper,
                    "s99_ci_lower": case.s99_ci_lower,
                    "s99_ci_upper": case.s99_ci_upper,
                    "srobust_ci_lower": case.srobust_ci_lower,
                    "srobust_ci_upper": case.srobust_ci_upper,
                    "bootstrap_unit_count": case.bootstrap_unit_count,
                    "bootstrap_repetitions": case.bootstrap_repetitions,
                }
                for case in self.cases
            ],
            "aggregation": self.aggregation.value,
            "s50": self.s50,
            "s99": self.s99,
            "srobust": self.srobust,
            "worst_case_srobust": self.worst_case_srobust,
            "reward": self.reward,
            "max_cv": self.max_cv,
            "s50_ci_lower": self.s50_ci_lower,
            "s50_ci_upper": self.s50_ci_upper,
            "s99_ci_lower": self.s99_ci_lower,
            "s99_ci_upper": self.s99_ci_upper,
            "srobust_ci_lower": self.srobust_ci_lower,
            "srobust_ci_upper": self.srobust_ci_upper,
            "confidence_level": self.confidence_level,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "min_bootstrap_units": self.min_bootstrap_units,
            "keep_srobust_threshold": self.keep_srobust_threshold,
            "confidence_srobust_floor": self.confidence_srobust_floor,
            "worst_case_srobust_floor": self.worst_case_srobust_floor,
            "max_cv_threshold": self.max_cv_threshold,
            "threshold_pass": self.threshold_pass,
            "confidence_pass": self.confidence_pass,
            "noise_pass": self.noise_pass,
            "worst_case_pass": self.worst_case_pass,
            "promotion_eligible": self.promotion_eligible,
            "promotion_reason_code": self.promotion_reason_code,
            "reward_bounds": list(self.reward_bounds),
            "reason_code": self.reason_code,
        }


def _quantiles_dict(value: Quantiles) -> dict[str, object]:
    return {
        "p50_ms": value.p50_ms,
        "p99_ms": value.p99_ms,
        "sample_count": value.sample_count,
        "sample_unit": value.sample_unit,
        "quantile_method": value.quantile_method,
        "artifact_sha256": value.artifact_sha256,
    }


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def kernel_reward(gates: GateVerdict, srobust: float | None) -> float | None:
    """Apply the sole public ``kernel_robust_v1`` scalar formula."""

    if not gates.compiled:
        return 0.0
    if not gates.correctness_gate_passed:
        return 20.0
    if gates.safety_finding or srobust is None or not math.isfinite(srobust):
        return None
    return 20.0 + 100.0 + 200.0 * _clip(srobust - 1.0, -0.25, 1.0)


def _validate_kernel_grade(grade: KernelGrade) -> None:
    if grade.policy_id != "kernel_robust_v1":
        raise ContractError("Kernel grade policy is unsupported", "invalid_kernel_grade")
    if (
        not 0.95 <= grade.confidence_level < 1.0
        or grade.bootstrap_seed < 0
        or grade.bootstrap_repetitions < 100
        or grade.min_bootstrap_units < 2
        or grade.keep_srobust_threshold < 1.05
        or grade.confidence_srobust_floor < 1.0
        or grade.worst_case_srobust_floor < 1.0
        or not 0 < grade.max_cv_threshold <= 0.10
    ):
        raise ContractError("Kernel grade policy fields are invalid", "invalid_kernel_grade")
    if grade.measurement_status is MeasurementStatus.VALID:
        _validate_valid_kernel_grade(grade)
        return
    points = (
        grade.s50,
        grade.s99,
        grade.srobust,
        grade.worst_case_srobust,
        grade.max_cv,
        grade.s50_ci_lower,
        grade.s50_ci_upper,
        grade.s99_ci_lower,
        grade.s99_ci_upper,
        grade.srobust_ci_lower,
        grade.srobust_ci_upper,
    )
    if grade.cases or any(value is not None for value in points) or grade.promotion_eligible:
        raise ContractError("Unmeasured grade carries promotion evidence", "invalid_kernel_grade")


def _validate_valid_kernel_grade(grade: KernelGrade) -> None:
    if not grade.gates.correctness_gate_passed or grade.gates.safety_finding:
        raise ContractError("Valid grade has failed evaluator gates", "invalid_kernel_grade")
    for case in grade.cases:
        values = (case.s50, case.s99, case.srobust, case.reference_cv, case.optimized_cv)
        counts = (case.reference.sample_count, case.optimized.sample_count)
        if (
            any(not math.isfinite(value) for value in values)
            or min(case.s50, case.s99, case.srobust) <= 0
            or min(case.reference_cv, case.optimized_cv) < 0
            or case.srobust != min(case.s50, case.s99)
            or counts[0] != counts[1]
            or min(counts) < 300
        ):
            raise ContractError("Kernel case grade is inconsistent", "invalid_kernel_grade")
    points = (grade.s50, grade.s99, grade.srobust, grade.worst_case_srobust, grade.max_cv)
    if not grade.cases or any(value is None or not math.isfinite(value) for value in points):
        raise ContractError("Measured grade is incomplete", "invalid_kernel_grade")
    assert grade.s50 is not None and grade.s99 is not None and grade.srobust is not None
    assert grade.worst_case_srobust is not None and grade.max_cv is not None
    expected_s50, expected_s99 = _aggregate(grade.cases, grade.aggregation)
    expected_reward = kernel_reward(grade.gates, grade.srobust)
    if not (
        math.isclose(grade.s50, expected_s50)
        and math.isclose(grade.s99, expected_s99)
        and grade.srobust == min(grade.s50, grade.s99)
        and grade.worst_case_srobust == min(case.srobust for case in grade.cases)
        and grade.max_cv == max(max(case.reference_cv, case.optimized_cv) for case in grade.cases)
        and expected_reward is not None
        and grade.reward is not None
        and math.isclose(grade.reward, expected_reward)
    ):
        raise ContractError("Measured grade values are inconsistent", "invalid_kernel_grade")
    _validate_grade_promotion(grade)


def _validate_grade_promotion(grade: KernelGrade) -> None:
    assert grade.srobust is not None and grade.worst_case_srobust is not None
    assert grade.max_cv is not None
    intervals = (
        grade.s50_ci_lower, grade.s50_ci_upper, grade.s99_ci_lower,
        grade.s99_ci_upper, grade.srobust_ci_lower, grade.srobust_ci_upper,
    )
    if any(value is None for value in intervals) and any(value is not None for value in intervals):
        raise ContractError("Kernel grade intervals are partial", "invalid_kernel_grade")
    if all(value is not None for value in intervals):
        pairs = ((intervals[0], intervals[1]), (intervals[2], intervals[3]), (intervals[4], intervals[5]))
        if any(lower > upper for lower, upper in pairs):
            raise ContractError("Kernel grade intervals are reversed", "invalid_kernel_grade")
    threshold = grade.srobust > grade.keep_srobust_threshold
    confidence = bool(grade.srobust_ci_lower is not None and grade.srobust_ci_lower > grade.confidence_srobust_floor)
    noise = grade.max_cv <= grade.max_cv_threshold
    worst = grade.worst_case_srobust >= grade.worst_case_srobust_floor
    reason = _promotion_reason(
        threshold=threshold, confidence=confidence, noise=noise,
        worst_case=worst, ci_available=grade.srobust_ci_lower is not None,
    )
    if (
        (grade.threshold_pass, grade.confidence_pass, grade.noise_pass, grade.worst_case_pass)
        != (threshold, confidence, noise, worst)
        or grade.promotion_eligible != (reason == "promotion_eligible")
        or grade.promotion_reason_code != reason
        or grade.task_status is not (TaskStatus.SUCCEEDED if grade.promotion_eligible else TaskStatus.NO_GAIN)
    ):
        raise ContractError("Kernel grade promotion fields are inconsistent", "invalid_kernel_grade")


def _case_grade(case: CaseTiming, policy: MeasurementPolicy) -> _CaseComputation:
    reference = quantiles(case.reference, policy)
    optimized = quantiles(case.optimized, policy)
    s50 = reference.p50_ms / optimized.p50_ms
    s99 = reference.p99_ms / optimized.p99_ms
    bootstrap = paired_block_bootstrap(
        case.paired_units,
        policy,
        seed=policy.bootstrap_seed ^ int(sha256_json({"case_id": case.case_id})[:16], 16),
    )
    intervals = _bootstrap_intervals(bootstrap, policy)
    return _CaseComputation(
        CaseGrade(
            case_id=case.case_id,
            reference=reference,
            optimized=optimized,
            s50=s50,
            s99=s99,
            srobust=min(s50, s99),
            workload_count=case.workload_count,
            reference_cv=coefficient_of_variation(case.reference.values_ms),
            optimized_cv=coefficient_of_variation(case.optimized.values_ms),
            s50_ci_lower=intervals[0],
            s50_ci_upper=intervals[1],
            s99_ci_lower=intervals[2],
            s99_ci_upper=intervals[3],
            srobust_ci_lower=intervals[4],
            srobust_ci_upper=intervals[5],
            bootstrap_unit_count=len(case.paired_units),
            bootstrap_repetitions=bootstrap.repetitions if bootstrap else 0,
        ),
        bootstrap,
    )


def _bootstrap_intervals(
    bootstrap: BootstrapDistribution | None,
    policy: MeasurementPolicy,
) -> tuple[float | None, ...]:
    if bootstrap is None:
        return (None, None, None, None, None, None)
    return (
        *bootstrap_interval(bootstrap.s50, policy),
        *bootstrap_interval(bootstrap.s99, policy),
        *bootstrap_interval(bootstrap.srobust, policy),
    )


def _aggregate(cases: Sequence[CaseGrade], policy: GradeAggregation) -> tuple[float, float]:
    if policy is GradeAggregation.EQUAL_CASE:
        return (
            sum(case.s50 for case in cases) / len(cases),
            sum(case.s99 for case in cases) / len(cases),
        )
    reference_p50 = sum(case.workload_count * case.reference.p50_ms for case in cases)
    optimized_p50 = sum(case.workload_count * case.optimized.p50_ms for case in cases)
    reference_p99 = sum(case.workload_count * case.reference.p99_ms for case in cases)
    optimized_p99 = sum(case.workload_count * case.optimized.p99_ms for case in cases)
    return reference_p50 / optimized_p50, reference_p99 / optimized_p99


def _aggregate_bootstrap(
    computations: Sequence[_CaseComputation],
    aggregation: GradeAggregation,
    policy: MeasurementPolicy,
) -> tuple[float | None, ...]:
    if any(item.bootstrap is None for item in computations):
        return (None, None, None, None, None, None)
    s50_values: list[float] = []
    s99_values: list[float] = []
    for repetition in range(policy.bootstrap_repetitions):
        if aggregation is GradeAggregation.EQUAL_CASE:
            s50 = sum(item.bootstrap.s50[repetition] for item in computations if item.bootstrap) / len(computations)
            s99 = sum(item.bootstrap.s99[repetition] for item in computations if item.bootstrap) / len(computations)
        else:
            s50, s99 = _weighted_bootstrap(computations, repetition)
        s50_values.append(s50)
        s99_values.append(s99)
    robust_values = [min(s50, s99) for s50, s99 in zip(s50_values, s99_values, strict=True)]
    return (
        *bootstrap_interval(s50_values, policy),
        *bootstrap_interval(s99_values, policy),
        *bootstrap_interval(robust_values, policy),
    )


def _weighted_bootstrap(
    computations: Sequence[_CaseComputation], repetition: int
) -> tuple[float, float]:
    items = tuple((item.grade, item.bootstrap) for item in computations)
    ref50 = sum(grade.workload_count * sample.reference_p50_ms[repetition] for grade, sample in items if sample)
    opt50 = sum(grade.workload_count * sample.optimized_p50_ms[repetition] for grade, sample in items if sample)
    ref99 = sum(grade.workload_count * sample.reference_p99_ms[repetition] for grade, sample in items if sample)
    opt99 = sum(grade.workload_count * sample.optimized_p99_ms[repetition] for grade, sample in items if sample)
    return ref50 / opt50, ref99 / opt99


def _promotion_reason(
    *, threshold: bool, confidence: bool, noise: bool, worst_case: bool, ci_available: bool
) -> str:
    if not threshold:
        return "srobust_threshold_not_met"
    if not noise:
        return "timing_noise_exceeds_policy"
    if not ci_available:
        return "timing_confidence_unavailable"
    if not confidence:
        return "timing_confidence_below_floor"
    if not worst_case:
        return "worst_case_regression"
    return "promotion_eligible"


def _unmeasured_grade(
    gates: GateVerdict,
    aggregation: GradeAggregation,
    status: MeasurementStatus,
    reason: str,
    policy: MeasurementPolicy,
) -> KernelGrade:
    reward = kernel_reward(gates, None)
    task_status = TaskStatus.REJECTED if not gates.correctness_gate_passed else TaskStatus.NO_MEASUREMENT
    return KernelGrade(
        policy_id="kernel_robust_v1",
        measurement_status=status,
        task_status=task_status,
        gates=gates,
        cases=(),
        aggregation=aggregation,
        s50=None,
        s99=None,
        srobust=None,
        worst_case_srobust=None,
        reward=reward,
        max_cv=None,
        s50_ci_lower=None,
        s50_ci_upper=None,
        s99_ci_lower=None,
        s99_ci_upper=None,
        srobust_ci_lower=None,
        srobust_ci_upper=None,
        confidence_level=policy.bootstrap_confidence_level,
        bootstrap_seed=policy.bootstrap_seed,
        bootstrap_repetitions=policy.bootstrap_repetitions,
        min_bootstrap_units=policy.min_bootstrap_units,
        keep_srobust_threshold=policy.keep_srobust_threshold,
        confidence_srobust_floor=policy.confidence_srobust_floor,
        worst_case_srobust_floor=policy.worst_case_srobust_floor,
        max_cv_threshold=policy.max_cv,
        threshold_pass=False,
        confidence_pass=False,
        noise_pass=False,
        worst_case_pass=False,
        promotion_eligible=False,
        promotion_reason_code=reason,
        reason_code=reason,
    )


def _valid_grade(
    gates: GateVerdict,
    computations: Sequence[_CaseComputation],
    aggregation: GradeAggregation,
    policy: MeasurementPolicy,
) -> KernelGrade:
    cases = tuple(item.grade for item in computations)
    s50, s99 = _aggregate(cases, aggregation)
    srobust = min(s50, s99)
    worst_case = min(case.srobust for case in cases)
    intervals = _aggregate_bootstrap(computations, aggregation, policy)
    max_cv = max(max(case.reference_cv, case.optimized_cv) for case in cases)
    threshold_pass = srobust > policy.keep_srobust_threshold
    ci_available = intervals[4] is not None
    confidence_pass = bool(
        ci_available and intervals[4] > policy.confidence_srobust_floor
    )
    noise_pass = max_cv <= policy.max_cv
    worst_case_pass = worst_case >= policy.worst_case_srobust_floor
    reason = _promotion_reason(
        threshold=threshold_pass,
        confidence=confidence_pass,
        noise=noise_pass,
        worst_case=worst_case_pass,
        ci_available=ci_available,
    )
    promotion = reason == "promotion_eligible"
    return KernelGrade(
        policy_id="kernel_robust_v1",
        measurement_status=MeasurementStatus.VALID,
        task_status=TaskStatus.SUCCEEDED if promotion else TaskStatus.NO_GAIN,
        gates=gates,
        cases=cases,
        aggregation=aggregation,
        s50=s50,
        s99=s99,
        srobust=srobust,
        worst_case_srobust=worst_case,
        reward=kernel_reward(gates, srobust),
        max_cv=max_cv,
        s50_ci_lower=intervals[0],
        s50_ci_upper=intervals[1],
        s99_ci_lower=intervals[2],
        s99_ci_upper=intervals[3],
        srobust_ci_lower=intervals[4],
        srobust_ci_upper=intervals[5],
        confidence_level=policy.bootstrap_confidence_level,
        bootstrap_seed=policy.bootstrap_seed,
        bootstrap_repetitions=policy.bootstrap_repetitions,
        min_bootstrap_units=policy.min_bootstrap_units,
        keep_srobust_threshold=policy.keep_srobust_threshold,
        confidence_srobust_floor=policy.confidence_srobust_floor,
        worst_case_srobust_floor=policy.worst_case_srobust_floor,
        max_cv_threshold=policy.max_cv,
        threshold_pass=threshold_pass,
        confidence_pass=confidence_pass,
        noise_pass=noise_pass,
        worst_case_pass=worst_case_pass,
        promotion_eligible=promotion,
        promotion_reason_code=reason,
        reason_code=None if promotion else reason,
    )


def grade_kernel(
    gates: GateVerdict,
    timings: Sequence[CaseTiming],
    *,
    measurement_policy: MeasurementPolicy | None = None,
    aggregation: GradeAggregation = GradeAggregation.EQUAL_CASE,
) -> KernelGrade:
    """Grade matching cases, failing closed on missing or invalid p99 evidence."""

    policy = measurement_policy or MeasurementPolicy()
    if not gates.correctness_gate_passed:
        return _unmeasured_grade(
            gates, aggregation, MeasurementStatus.NOT_RUN_DUE_TO_GATE, "correctness_or_integrity_gate", policy
        )
    if gates.safety_finding:
        return _unmeasured_grade(
            gates, aggregation, MeasurementStatus.NOT_RUN_DUE_TO_SAFETY, "confirmed_safety_finding", policy
        )
    if not timings:
        return _unmeasured_grade(gates, aggregation, MeasurementStatus.UNSUPPORTED, "missing_timing_cases", policy)
    case_ids = [case.case_id for case in timings]
    if len(case_ids) != len(set(case_ids)):
        return _unmeasured_grade(gates, aggregation, MeasurementStatus.INVALID, "duplicate_case_id", policy)

    try:
        computations = tuple(_case_grade(case, policy) for case in timings)
    except ContractError as error:
        if error.reason_code == "insufficient_samples":
            status = MeasurementStatus.INSUFFICIENT_SAMPLES
        elif error.reason_code in {"unsupported_sample_unit", "needs_better_timer"}:
            status = MeasurementStatus.UNSUPPORTED
        else:
            status = MeasurementStatus.INVALID
        return _unmeasured_grade(gates, aggregation, status, error.reason_code, policy)

    return _valid_grade(gates, computations, aggregation, policy)
