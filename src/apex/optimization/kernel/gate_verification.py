"""Trusted compile/correctness gate execution for one kernel attempt."""

from __future__ import annotations

from apex.core import TaskStatus
from apex.runtime import require_gpu_lease_heartbeat
from apex.storage import ArtifactReceipt

from .attempts import (
    AttemptSession,
    CompileCorrectnessReceipts,
    KernelAttemptOutcome,
    PreparedCandidate,
)
from .lifecycle import close_prepared
from .gpu_recording import record_gpu_lease_heartbeat
from .reward_recording import record_attempt_gate_reward
from .verification import CandidateVerifier


def verify_compile_correctness(
    verifier: CandidateVerifier,
    attempt: AttemptSession,
    prepared: PreparedCandidate,
    prior: tuple[ArtifactReceipt, ...],
) -> CompileCorrectnessReceipts | KernelAttemptOutcome:
    """Run evaluator-owned gates and commit attributable low rewards on failure."""

    run = attempt.run
    require_gpu_lease_heartbeat(run.gpu_lease_guard)
    compiled = verifier.compile(
        run.resolved,
        candidate_root=attempt.candidate.root,
        expected_source_digest=prepared.normal_source_digest,
    )
    compile_receipt = run.record.record_command(attempt.attempt_id, compiled)
    compile_heartbeat = require_gpu_lease_heartbeat(run.gpu_lease_guard)
    compile_lease = record_gpu_lease_heartbeat(
        run.record,
        compile_heartbeat,
        attempt_id=attempt.attempt_id,
        phase="compile",
    )
    evidence = (
        *prior, *prepared.candidate_receipts, compile_receipt, compile_lease
    )
    if not compiled.passed:
        record_attempt_gate_reward(
            run.record,
            attempt.attempt_id,
            stage="compile",
            command_receipt=compile_receipt,
        )
        return close_prepared(
            attempt,
            prepared,
            TaskStatus.REJECTED,
            "compile_failed",
            evidence,
            closure="reject",
        )
    require_gpu_lease_heartbeat(run.gpu_lease_guard)
    correctness = verifier.correctness(
        run.resolved,
        candidate_root=attempt.candidate.root,
        expected_source_digest=prepared.normal_source_digest,
    )
    correctness_receipt = run.record.record_command(attempt.attempt_id, correctness)
    correctness_heartbeat = require_gpu_lease_heartbeat(run.gpu_lease_guard)
    correctness_lease = record_gpu_lease_heartbeat(
        run.record,
        correctness_heartbeat,
        attempt_id=attempt.attempt_id,
        phase="correctness",
    )
    evidence = (*evidence, correctness_receipt, correctness_lease)
    if not correctness.passed:
        record_attempt_gate_reward(
            run.record,
            attempt.attempt_id,
            stage="correctness",
            command_receipt=correctness_receipt,
        )
        return close_prepared(
            attempt,
            prepared,
            TaskStatus.REJECTED,
            "correctness_failed",
            evidence,
            closure="reject",
        )
    return CompileCorrectnessReceipts(
        compile_receipt,
        correctness_receipt,
        evidence,
    )


__all__ = ["verify_compile_correctness"]
