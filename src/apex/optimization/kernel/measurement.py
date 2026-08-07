"""Trusted standalone kernel measurement parsing and result projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apex.evaluation import (
    GateVerdict,
    GradeAggregation,
    KernelGrade,
    KernelMeasurementArtifact,
    MeasurementPolicy,
    MeasurementStatus,
    grade_kernel,
    load_kernel_measurement_report,
)
from apex.intake import KernelMeasurementSpec, ResolvedTaskSpec


@dataclass(frozen=True, slots=True)
class KernelMeasurementEvaluation:
    artifact: KernelMeasurementArtifact
    grade: KernelGrade

    @property
    def reward_eligible(self) -> bool:
        return self.grade.measurement_status is MeasurementStatus.VALID

    @property
    def improved(self) -> bool:
        return self.reward_eligible and self.grade.promotion_eligible

    def task_result_fields(self) -> dict[str, Any]:
        grade = self.grade
        return {
            "measurement_status": grade.measurement_status.value,
            "measurement_report_sha256": self.artifact.sha256,
            "grade_policy_id": grade.policy_id,
            "s50": grade.s50,
            "s99": grade.s99,
            "srobust": grade.srobust,
            "worst_case_srobust": grade.worst_case_srobust,
            "reward": grade.reward,
            "max_cv": grade.max_cv,
            "s50_ci_lower": grade.s50_ci_lower,
            "s50_ci_upper": grade.s50_ci_upper,
            "s99_ci_lower": grade.s99_ci_lower,
            "s99_ci_upper": grade.s99_ci_upper,
            "srobust_ci_lower": grade.srobust_ci_lower,
            "srobust_ci_upper": grade.srobust_ci_upper,
            "confidence_level": grade.confidence_level,
            "threshold_pass": grade.threshold_pass,
            "confidence_pass": grade.confidence_pass,
            "noise_pass": grade.noise_pass,
            "worst_case_pass": grade.worst_case_pass,
            "promotion_eligible": grade.promotion_eligible,
            "promotion_reason_code": grade.promotion_reason_code,
        }


def evaluate_kernel_measurement(
    resolved: ResolvedTaskSpec,
    *,
    candidate_root: Path,
) -> KernelMeasurementEvaluation | None:
    """Recompute the canonical grade from a fresh evaluator-owned report."""

    contract = resolved.task.measurement
    if contract is None:
        return None
    report_path = candidate_root.joinpath(*contract.report_path.split("/"))
    aggregation = GradeAggregation(contract.aggregation)
    policy = _measurement_policy(contract)
    artifact = load_kernel_measurement_report(
        report_path,
        aggregation=aggregation,
        measurement_policy=policy,
    )
    grade = grade_kernel(
        GateVerdict(
            compiled=True,
            correct=True,
            integrity_passed=True,
            tampering_passed=True,
            safety_finding=False,
        ),
        artifact.cases,
        measurement_policy=artifact.policy,
        aggregation=artifact.aggregation,
    )
    return KernelMeasurementEvaluation(artifact, grade)


def _measurement_policy(contract: KernelMeasurementSpec) -> MeasurementPolicy:
    return MeasurementPolicy(
        policy_id=contract.policy_id,
        min_valid_samples=contract.min_valid_samples,
        min_tail_observations=contract.min_tail_observations,
        sample_unit=contract.sample_unit,
        quantile_method=contract.quantile_method,
        warmup_samples=contract.warmup_samples,
        keep_srobust_threshold=contract.keep_srobust_threshold,
        confidence_srobust_floor=contract.confidence_srobust_floor,
        worst_case_srobust_floor=contract.worst_case_srobust_floor,
        max_cv=contract.max_cv,
        bootstrap_confidence_level=contract.bootstrap_confidence_level,
        bootstrap_seed=contract.bootstrap_seed,
        bootstrap_repetitions=contract.bootstrap_repetitions,
        min_bootstrap_units=contract.min_bootstrap_units,
    )


__all__ = ["KernelMeasurementEvaluation", "evaluate_kernel_measurement"]
