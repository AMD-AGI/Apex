"""Trusted standalone kernel measurement execution and grade projection."""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apex.core import ApexError, ContractError, IntegrityError, sha256_file, sha256_json
from apex.evaluation import (
    GateVerdict,
    GradeAggregation,
    KernelGrade,
    KernelMeasurementArtifact,
    KernelMeasurementExecutionReceipt,
    MeasurementPolicy,
    MeasurementStatus,
    grade_kernel,
    load_kernel_measurement_report,
)
from apex.intake import KernelMeasurementSpec, ResolvedTaskSpec
from apex.ports import KernelMeasurementPort, KernelMeasurementRequest

from .verification import candidate_source_digest


@dataclass(frozen=True, slots=True)
class KernelMeasurementEvaluation:
    artifact: KernelMeasurementArtifact
    execution: KernelMeasurementExecutionReceipt
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


@dataclass(frozen=True, slots=True)
class KernelMeasurementCapture:
    artifact: KernelMeasurementArtifact
    execution: KernelMeasurementExecutionReceipt


def evaluate_kernel_measurement(
    resolved: ResolvedTaskSpec,
    *,
    candidate_root: Path,
    run_id: str,
    attempt_id: str,
    output_root: Path,
    evaluator: KernelMeasurementPort | None,
) -> KernelMeasurementEvaluation:
    """Measure through a trusted port and recompute the grade from its sealed output."""

    return grade_kernel_measurement(
        capture_kernel_measurement(
            resolved,
            candidate_root=candidate_root,
            run_id=run_id,
            attempt_id=attempt_id,
            output_root=output_root,
            evaluator=evaluator,
        )
    )


def capture_kernel_measurement(
    resolved: ResolvedTaskSpec,
    *,
    candidate_root: Path,
    run_id: str,
    attempt_id: str,
    output_root: Path,
    evaluator: KernelMeasurementPort | None,
) -> KernelMeasurementCapture:
    """Capture and validate raw evidence without committing a grade or reward."""

    contract = resolved.task.measurement
    if contract is None:
        raise ContractError(
            "No trusted standalone kernel measurement contract is configured",
            "measurement_contract_missing",
        )
    if evaluator is None:
        raise ContractError(
            "No trusted standalone kernel measurement adapter is configured",
            "measurement_evaluator_unavailable",
        )
    if evaluator.adapter_id != contract.adapter_id:
        raise ContractError(
            "Configured kernel measurement adapter does not match the frozen task",
            "measurement_adapter_mismatch",
        )
    evaluator_method = _trusted_adapter_method(evaluator, contract)
    policy = _measurement_policy(contract)
    policy_sha256 = sha256_json(policy.to_dict())
    source_sha256 = candidate_source_digest(
        candidate_root, resolved.task.editable_files
    )
    harness_sha256 = _validate_harness(resolved, candidate_root)
    report_path = _prepare_output(output_root, candidate_root)
    request = _request(
        resolved,
        run_id=run_id,
        attempt_id=attempt_id,
        candidate_root=candidate_root,
        report_path=report_path,
        source_sha256=source_sha256,
        harness_sha256=harness_sha256,
        policy_sha256=policy_sha256,
        measurement_method_sha256=evaluator_method,
    )
    execution = _execute(evaluator, request, resolved, output_root)
    artifact = load_kernel_measurement_report(
        report_path,
        aggregation=GradeAggregation(contract.aggregation),
        measurement_policy=policy,
    )
    _validate_artifact_binding(artifact, execution, contract)
    return KernelMeasurementCapture(artifact, execution)


def grade_kernel_measurement(
    capture: KernelMeasurementCapture,
) -> KernelMeasurementEvaluation:
    """Recompute the canonical grade from one already validated raw capture."""

    artifact = capture.artifact
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
    return KernelMeasurementEvaluation(artifact, capture.execution, grade)


def load_kernel_measurement_capture(
    resolved: ResolvedTaskSpec,
    *,
    report_path: Path,
    execution: KernelMeasurementExecutionReceipt,
) -> KernelMeasurementCapture:
    """Reload a CAS report and independently revalidate its frozen policy binding."""

    contract = resolved.task.measurement
    if contract is None:
        raise ContractError(
            "No trusted standalone kernel measurement contract is configured",
            "measurement_contract_missing",
        )
    policy = _measurement_policy(contract)
    if execution.measurement_policy_sha256 != sha256_json(policy.to_dict()):
        raise IntegrityError(
            "Measurement execution receipt binds another grading policy",
            "measurement_policy_mismatch",
        )
    artifact = load_kernel_measurement_report(
        report_path,
        aggregation=GradeAggregation(contract.aggregation),
        measurement_policy=policy,
    )
    _validate_artifact_binding(artifact, execution, contract)
    return KernelMeasurementCapture(artifact, execution)


def _execute(
    evaluator: KernelMeasurementPort,
    request: KernelMeasurementRequest,
    resolved: ResolvedTaskSpec,
    output_root: Path,
) -> KernelMeasurementExecutionReceipt:
    started = time.monotonic_ns()
    try:
        output = evaluator.measure(request)
    except ApexError:
        raise
    except Exception as error:
        raise ContractError(
            "Trusted kernel measurement adapter failed",
            "measurement_adapter_failed",
        ) from error
    returned = time.monotonic_ns()
    _validate_output(output.writer_id, output.report_path, request, output_root)
    report_sha256 = sha256_file(request.report_path)
    report_size = request.report_path.stat().st_size
    observed = time.monotonic_ns()
    observed_source = candidate_source_digest(
        request.candidate_root, resolved.task.editable_files
    )
    observed_harness = _validate_harness(resolved, request.candidate_root)
    if observed_source != request.candidate_source_sha256:
        raise IntegrityError(
            "Candidate source changed during evaluator measurement",
            "candidate_changed_during_measurement",
        )
    if observed_harness != request.harness_sha256:
        raise IntegrityError(
            "Protected harness changed during evaluator measurement",
            "measurement_harness_changed",
        )
    completed = time.monotonic_ns()
    return KernelMeasurementExecutionReceipt(
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        writer_id=output.writer_id,
        candidate_source_sha256=request.candidate_source_sha256,
        harness_sha256=request.harness_sha256,
        measurement_method_sha256=request.measurement_method_sha256,
        measurement_policy_sha256=request.measurement_policy_sha256,
        report_sha256=report_sha256,
        report_size=report_size,
        phase_started_monotonic_ns=started,
        adapter_returned_monotonic_ns=returned,
        output_observed_monotonic_ns=observed,
        phase_completed_monotonic_ns=completed,
    )


def _request(
    resolved: ResolvedTaskSpec,
    *,
    run_id: str,
    attempt_id: str,
    candidate_root: Path,
    report_path: Path,
    source_sha256: str,
    harness_sha256: str,
    policy_sha256: str,
    measurement_method_sha256: str,
) -> KernelMeasurementRequest:
    contract = resolved.task.measurement
    assert contract is not None
    return KernelMeasurementRequest(
        run_id=run_id,
        attempt_id=attempt_id,
        adapter_id=contract.adapter_id,
        candidate_root=candidate_root.resolve(strict=True),
        report_path=report_path,
        harness_paths=tuple(
            candidate_root.joinpath(*relative.split("/")).resolve(strict=True)
            for relative in contract.harness_files
        ),
        runner_argv=contract.runner.argv,
        runner_cwd=(
            candidate_root
            if contract.runner.cwd == "."
            else candidate_root.joinpath(*contract.runner.cwd.split("/"))
        ).resolve(strict=True),
        runner_env=dict(contract.runner.env),
        runner_timeout_seconds=contract.runner.timeout_seconds,
        candidate_source_sha256=source_sha256,
        harness_sha256=harness_sha256,
        measurement_method_sha256=measurement_method_sha256,
        measurement_policy_sha256=policy_sha256,
    )


def _prepare_output(output_root: Path, candidate_root: Path) -> Path:
    candidate = candidate_root.resolve(strict=True)
    requested = output_root.resolve(strict=False)
    if requested.is_relative_to(candidate) or candidate.is_relative_to(requested):
        raise IntegrityError(
            "Measurement output overlaps the candidate workspace",
            "candidate_visible_measurement_output",
        )
    try:
        output_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    except OSError as error:
        raise IntegrityError(
            "Evaluator measurement output cannot be created",
            "measurement_output_unavailable",
        ) from error
    os.chmod(output_root, 0o700)
    report_path = (output_root / "raw-report.json").resolve(strict=False)
    if report_path.exists() or report_path.is_symlink():
        raise IntegrityError(
            "Evaluator measurement report existed before the phase",
            "stale_measurement_report",
        )
    return report_path


def _validate_output(
    writer_id: str,
    report_path: Path,
    request: KernelMeasurementRequest,
    output_root: Path,
) -> None:
    if writer_id != request.adapter_id:
        raise IntegrityError(
            "Kernel measurement writer identity does not match the frozen adapter",
            "measurement_writer_mismatch",
        )
    if report_path != request.report_path:
        raise IntegrityError(
            "Kernel measurement adapter returned an unexpected report path",
            "measurement_report_path_mismatch",
        )
    try:
        metadata = os.lstat(report_path)
        resolved = report_path.resolve(strict=True)
    except OSError as error:
        raise IntegrityError(
            "Kernel measurement adapter did not produce a report",
            "measurement_report_missing",
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or resolved.parent != output_root.resolve(strict=True)
    ):
        raise IntegrityError(
            "Kernel measurement adapter output is not an evaluator-owned regular file",
            "unsafe_measurement_report",
        )


def _validate_harness(resolved: ResolvedTaskSpec, root: Path) -> str:
    if not resolved.harness_file_hashes or resolved.harness_sha256 is None:
        raise IntegrityError(
            "Frozen kernel measurement harness evidence is missing",
            "measurement_harness_evidence_missing",
        )
    observed: dict[str, str] = {}
    resolved_root = root.resolve(strict=True)
    for relative, expected in sorted(resolved.harness_file_hashes.items()):
        path = root.joinpath(*relative.split("/"))
        try:
            metadata = os.lstat(path)
            target = path.resolve(strict=True)
        except OSError as error:
            raise IntegrityError(
                "Protected kernel measurement harness is missing",
                "measurement_harness_changed",
            ) from error
        digest = sha256_file(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not target.is_relative_to(resolved_root)
            or digest != expected
        ):
            raise IntegrityError(
                "Protected kernel measurement harness differs from frozen bytes",
                "measurement_harness_changed",
            )
        observed[relative] = digest
    aggregate = sha256_json(
        {
            "schema": "apex.kernel-measurement-harness/v1",
            "files": observed,
        }
    )
    if aggregate != resolved.harness_sha256:
        raise IntegrityError(
            "Protected kernel measurement harness digest is invalid",
            "measurement_harness_changed",
        )
    return aggregate


def _validate_artifact_binding(
    artifact: KernelMeasurementArtifact,
    execution: KernelMeasurementExecutionReceipt,
    contract: KernelMeasurementSpec,
) -> None:
    declared = artifact.protocol.measurement_method_sha256.removeprefix("sha256:")
    frozen = contract.measurement_method_sha256.removeprefix("sha256:")
    if (
        declared != frozen
        or execution.measurement_method_sha256.removeprefix("sha256:") != frozen
    ):
        raise IntegrityError(
            "Raw report measurement method differs from the frozen evaluator method",
            "measurement_method_mismatch",
        )
    if artifact.sha256 != execution.report_sha256:
        raise IntegrityError(
            "Raw report differs from the evaluator execution receipt",
            "measurement_report_changed",
        )


def _trusted_adapter_method(
    evaluator: KernelMeasurementPort,
    contract: KernelMeasurementSpec,
) -> str:
    observed = getattr(evaluator, "measurement_method_sha256", "")
    if not isinstance(observed, str):
        observed = ""
    normalized = observed.removeprefix("sha256:")
    frozen = contract.measurement_method_sha256.removeprefix("sha256:")
    if (
        normalized != frozen
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise IntegrityError(
            "Trusted adapter method differs from the frozen measurement method",
            "measurement_method_mismatch",
        )
    return observed


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


__all__ = [
    "KernelMeasurementCapture",
    "KernelMeasurementEvaluation",
    "capture_kernel_measurement",
    "evaluate_kernel_measurement",
    "grade_kernel_measurement",
    "load_kernel_measurement_capture",
]
