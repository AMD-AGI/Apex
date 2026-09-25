"""Translate standalone kernel state into the generic safety-gate contracts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from apex.core import ContractError, IntegrityError, sha256_json
from apex.evaluation.safety import (
    ArtifactKind,
    FindingStatus,
    FrozenCandidate,
    InstrumentationControl,
    KernelLanguage,
    PhaseIsolationReceipt,
    SafetyGateResult,
    TaskSafetyProfile,
    VerificationPlan,
    VerificationPolicy,
    decide_safety,
)
from apex.intake import ResolvedTaskSpec


def materialize_safety_candidate(
    candidate_root: Path,
    *,
    destination: Path,
    profile: TaskSafetyProfile,
) -> FrozenCandidate:
    if destination.exists():
        raise ContractError(
            "safety candidate destination exists", "safety_candidate_exists"
        )
    destination.mkdir(parents=True)
    for relative in profile.submission_paths:
        source = candidate_root.joinpath(*relative.split("/"))
        target = destination.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, 0o444)
    frozen = FrozenCandidate.capture(destination, profile)
    directories = (path for path in destination.rglob("*") if path.is_dir())
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        os.chmod(directory, 0o555)
    os.chmod(destination, 0o555)
    frozen.verify()
    return frozen


def safety_profile(resolved: ResolvedTaskSpec) -> TaskSafetyProfile:
    language = {
        "python": KernelLanguage.PYTHON,
        "triton": KernelLanguage.TRITON,
        "hip": KernelLanguage.HIP,
    }[resolved.task.language]
    artifact_kind = (
        ArtifactKind.SOURCE_AOT
        if language is KernelLanguage.HIP
        else ArtifactKind.PYTHON_JIT
    )
    instrumentation = (
        InstrumentationControl.RECOMPILE
        if language is KernelLanguage.HIP
        else InstrumentationControl.COMPILER_CONTROLLED
    )
    capability = {
        KernelLanguage.PYTHON: "python_dispatch",
        KernelLanguage.TRITON: "compiler_controlled_jit",
        KernelLanguage.HIP: "fixed_recipe_recompile",
    }[language]
    return TaskSafetyProfile(
        language=language,
        artifact_kind=artifact_kind,
        instrumentation_control=instrumentation,
        submission_paths=tuple(sorted(resolved.task.editable_files)),
        target_symbols=tuple(sorted(set(resolved.task.target_functions))),
        adapter_capabilities=(capability,),
    )


def baseline_source_digest(resolved: ResolvedTaskSpec) -> str:
    return sha256_json(
        {
            "schema_version": "apex.baseline-kernel-source/v1",
            "files": dict(sorted(resolved.baseline_file_hashes.items())),
        }
    )


def validate_safety_result(
    plan: VerificationPlan,
    isolation: PhaseIsolationReceipt,
    result: SafetyGateResult,
    policy: VerificationPolicy,
) -> None:
    exact = (
        result.run_id == plan.run_id
        and result.candidate_id == plan.candidate_id
        and result.anchor_generation == plan.anchor_generation
        and result.plan_fingerprint == plan.fingerprint
        and result.policy_fingerprint == plan.policy_fingerprint
        and result.source_digest == plan.source_digest
        and result.candidate_digest == plan.candidate_digest
        and result.deployed_digest == plan.deployed_digest
        and result.isolation_receipt_fingerprint == isolation.fingerprint
    )
    expected_decision = decide_safety(
        result.evaluations,
        policy=policy,
        blocking_errors=result.gate_errors,
    )
    if not exact or expected_decision != result.decision:
        raise IntegrityError(
            "safety adapter returned evidence for a different plan or policy",
            "invalid_safety_result",
        )


def safety_rejection_reason(result: SafetyGateResult) -> str:
    if any(
        evaluation.finding is FindingStatus.FOUND
        for evaluation in result.evaluations
    ):
        return "confirmed_safety_finding"
    if any(
        reason.startswith("required_safety_incomplete")
        for reason in result.decision.reason_codes
    ):
        return "required_safety_incomplete"
    return "safety_gate_rejected"


def task_safety_fields(
    result: SafetyGateResult,
    receipt_digest: str,
) -> dict[str, object]:
    if result.safety_certified:
        status = "certified"
    elif any(
        evaluation.finding is FindingStatus.FOUND
        for evaluation in result.evaluations
    ):
        status = "rejected_finding"
    elif any(
        reason.startswith("required_safety_incomplete")
        for reason in result.decision.reason_codes
    ):
        status = "required_incomplete"
    elif not result.decision.promotion_eligible:
        status = "gate_rejected"
    elif not result.evaluations:
        status = "not_configured"
    elif all(
        evaluation.finding is FindingStatus.CLEAN
        for evaluation in result.evaluations
    ):
        status = "clean_unqualified"
    else:
        status = "advisory_incomplete"
    return {
        "safety_status": status,
        "safety_certified": result.safety_certified,
        "safety_result_fingerprint": result.fingerprint,
        "safety_receipt_digest": receipt_digest,
    }


__all__ = [
    "baseline_source_digest",
    "materialize_safety_candidate",
    "safety_profile",
    "safety_rejection_reason",
    "task_safety_fields",
    "validate_safety_result",
]
