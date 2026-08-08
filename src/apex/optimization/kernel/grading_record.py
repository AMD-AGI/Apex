"""Canonical event projections for kernel timing, reward, and promotion evidence."""

from __future__ import annotations

from apex.evaluation import KernelGrade, KernelMeasurementArtifact, MeasurementPolicy


def measurement_payload(
    artifact: KernelMeasurementArtifact,
    grade: KernelGrade,
) -> dict[str, object]:
    protocol = artifact.protocol
    return {
        "measurement_schema": "apex.kernel-measurement/v1",
        "measurement_status": grade.measurement_status.value,
        "measurement_policy_id": artifact.policy.policy_id,
        "grade_policy_id": grade.policy_id,
        "aggregation": grade.aggregation.value,
        "timing_method": artifact.timing_method,
        "timer_resolution_ns": protocol.timer_resolution_ns,
        "inner_repeats": protocol.inner_repeats,
        "measurement_method_sha256": protocol.measurement_method_sha256,
        "abba_seed": protocol.abba_seed,
        "abba_block_count": len(artifact.blocks),
        "warmup_samples": artifact.warmup_samples,
        "case_count": len(grade.cases),
        **grade_statistics(grade),
        "reward": grade.reward,
        "reason_code": grade.reason_code,
    }


def grade_statistics(grade: KernelGrade) -> dict[str, object]:
    return {
        "s50": grade.s50,
        "s99": grade.s99,
        "srobust": grade.srobust,
        "worst_case_srobust": grade.worst_case_srobust,
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


def reward_vector(grade: KernelGrade) -> dict[str, object]:
    return {
        "compile": grade.gates.compiled,
        "correctness": grade.gates.correct,
        "integrity": grade.gates.integrity_passed,
        "anti_tampering": grade.gates.tampering_passed,
        "safety": {"finding": grade.gates.safety_finding},
        **{f"kernel_{key}": value for key, value in grade_statistics(grade).items()},
        "kernel_robust_reward": grade.reward,
    }


def kernel_reward_policy_source(policy: MeasurementPolicy) -> dict[str, object]:
    return {
        "schema": "apex.kernel-reward-policy/v1",
        "measurement_schema": "apex.kernel-measurement/v1",
        "policy_id": "kernel_robust_v1",
        "measurement_policy": policy.to_dict(),
        "timing_protocol": {
            "execution_receipt": "apex.kernel-measurement-execution/v1",
            "writer": "trusted_evaluator_adapter",
            "phase": "measurement",
            "protected_harness_digest_required": True,
            "ordering": "seeded_paired_abba_blocks",
            "inner_repeats": 1,
            "timer_resolution_required": True,
            "measurement_method_sha256_required": True,
            "gpu_health_before_after_each_block_required": True,
        },
        "formula": (
            "20*Icompile + Icorrect*(100 + "
            "200*clip(min(S50,S99)-1,-0.25,1.00))"
        ),
        "promotion": {
            "point_threshold": "Srobust > keep_srobust_threshold",
            "confidence": "Srobust_CI_lower > confidence_srobust_floor",
            "noise": "max_case_cv <= max_cv",
            "worst_case": "worst_case_srobust >= worst_case_srobust_floor",
        },
    }


__all__ = [
    "grade_statistics",
    "kernel_reward_policy_source",
    "measurement_payload",
    "reward_vector",
]
