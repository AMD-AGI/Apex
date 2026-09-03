"""Lease-bracketed standalone timing with deferred evidence commitment."""

from __future__ import annotations

from dataclasses import dataclass

from apex.core import ApexError
from apex.ports import KernelMeasurementPort
from apex.runtime import require_gpu_measurement_guard
from apex.storage import ArtifactReceipt

from .attempts import AttemptSession, PreparedCandidate
from .gpu_recording import record_gpu_measurement_bracket
from .measurement import KernelMeasurementEvaluation, evaluate_kernel_measurement
from .verification import CandidateVerifier


@dataclass(frozen=True, slots=True)
class LeaseMeasurementExecution:
    performance_passed: bool
    performance_receipt: ArtifactReceipt
    bracket_receipt: ArtifactReceipt
    measurement: KernelMeasurementEvaluation | None
    measurement_error: str | None


def execute_lease_measurement(
    attempt: AttemptSession,
    prepared: PreparedCandidate,
    *,
    verifier: CandidateVerifier,
    evaluator: KernelMeasurementPort | None,
    capture_measurement: bool,
) -> LeaseMeasurementExecution:
    """Run timing under a bracket; write no grade, reward, or delivery inside it."""

    run = attempt.run
    measurement = None
    measurement_error = None
    with require_gpu_measurement_guard(
        run.gpu_lease_guard, attempt.attempt_id
    ) as bracket:
        performance = verifier.performance(
            run.resolved,
            candidate_root=attempt.candidate.root,
            expected_source_digest=prepared.normal_source_digest,
        )
        performance_receipt = run.record.record_command(
            attempt.attempt_id, performance
        )
        if performance.passed and capture_measurement:
            try:
                measurement = evaluate_kernel_measurement(
                    run.resolved,
                    candidate_root=attempt.candidate.root,
                    run_id=run.run_id,
                    attempt_id=attempt.attempt_id,
                    output_root=run.run_root / "measurements" / attempt.attempt_id,
                    evaluator=evaluator,
                )
            except ApexError as error:
                measurement_error = error.reason_code
    bracket_receipt = record_gpu_measurement_bracket(
        run.record, bracket.receipt, attempt_id=attempt.attempt_id
    )
    return LeaseMeasurementExecution(
        performance.passed,
        performance_receipt,
        bracket_receipt,
        measurement,
        measurement_error,
    )


__all__ = ["LeaseMeasurementExecution", "execute_lease_measurement"]
