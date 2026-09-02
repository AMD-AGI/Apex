"""Immutable standalone attempt outcomes and deterministic best selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import AgentBackendName, TaskStatus
from apex.evaluation import EvaluationContractReceipt
from apex.evaluation.safety import FrozenCandidate, SafetyGateResult, VerificationPlan
from apex.intake import ResolvedTaskSpec, TaskSpec
from apex.runtime import GpuLease, GpuLeaseReceipt
from apex.storage import ArtifactReceipt

from .context import KernelContext
from .measurement import KernelMeasurementEvaluation
from .run_record import KernelRunRecord
from .workspace import CandidateWorkspace


@dataclass(frozen=True, slots=True)
class KernelOptimizeRequest:
    task: TaskSpec
    result_json: Path
    backend_override: AgentBackendName | None = None
    model_override: str | None = None
    effort_override: str | None = None


@dataclass(frozen=True, slots=True)
class RunSession:
    request: KernelOptimizeRequest
    resolved: ResolvedTaskSpec
    run_id: str
    run_root: Path
    record: KernelRunRecord
    evaluation_contract: EvaluationContractReceipt
    evaluation_contract_artifact: ArtifactReceipt
    gpu_lease: GpuLeaseReceipt
    gpu_lease_artifact: ArtifactReceipt
    gpu_lease_guard: GpuLease


@dataclass(frozen=True, slots=True)
class AttemptSession:
    run: RunSession
    attempt_id: str
    cycle: int
    candidate: CandidateWorkspace
    context: KernelContext


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    normal_source_digest: str
    changed_files: tuple[str, ...]
    candidate_receipts: tuple[ArtifactReceipt, ...]
    safety_candidate: FrozenCandidate
    safety_plan: VerificationPlan


@dataclass(frozen=True, slots=True)
class CompileCorrectnessReceipts:
    compile: ArtifactReceipt
    correctness: ArtifactReceipt
    evidence: tuple[ArtifactReceipt, ...]


@dataclass(frozen=True, slots=True)
class SafetyEvidence:
    result: SafetyGateResult
    receipt: ArtifactReceipt
    evidence: tuple[ArtifactReceipt, ...]


@dataclass(frozen=True, slots=True)
class MeasurementEvidence:
    measurement: KernelMeasurementEvaluation | None
    receipt: ArtifactReceipt | None
    evidence: tuple[ArtifactReceipt, ...]


@dataclass(frozen=True, slots=True)
class KernelAttemptOutcome:
    """One closed attempt; selection never mutates or overwrites this record."""

    attempt_id: str
    cycle: int
    status: TaskStatus
    reason_code: str
    strategy_fingerprint: str
    evidence_receipts: tuple[str, ...]
    candidate_root: Path | None = None
    changed_files: tuple[str, ...] = ()
    safety_result: SafetyGateResult | None = None
    safety_receipt_digest: str | None = None
    measurement: KernelMeasurementEvaluation | None = None
    measurement_fields: Mapping[str, Any] | None = None
    eligible: bool = False
    stop_search: bool = False
    decision_recorded: bool = False

    def __post_init__(self) -> None:
        if self.cycle < 0 or not self.reason_code or not self.evidence_receipts:
            raise ValueError("kernel attempt outcome is incomplete")
        if self.eligible and (
            self.candidate_root is None
            or not self.changed_files
            or self.safety_result is None
            or self.safety_receipt_digest is None
        ):
            raise ValueError("eligible kernel attempt lacks verified candidate evidence")
        if self.decision_recorded and self.eligible:
            raise ValueError("eligible attempt cannot be decided before best selection")

    @property
    def rank(self) -> tuple[int, float, float, int]:
        """Prefer trusted robust grades, then use a stable earliest-attempt tie break."""

        if self.measurement is None:
            return (0, float("-inf"), float("-inf"), -self.cycle)
        grade = self.measurement.grade
        return (
            1,
            grade.srobust if grade.srobust is not None else float("-inf"),
            grade.reward if grade.reward is not None else float("-inf"),
            -self.cycle,
        )

    def result_measurement_fields(self) -> dict[str, Any]:
        if self.measurement is not None:
            return self.measurement.task_result_fields()
        return dict(self.measurement_fields or {})


def select_best(outcomes: tuple[KernelAttemptOutcome, ...]) -> KernelAttemptOutcome | None:
    """Return the best eligible attempt without using agent or stdout claims."""

    eligible = tuple(item for item in outcomes if item.eligible)
    return max(eligible, key=lambda item: item.rank) if eligible else None


def representative_failure(
    outcomes: tuple[KernelAttemptOutcome, ...],
) -> KernelAttemptOutcome:
    """Choose the most informative terminal result when no candidate is eligible."""

    if not outcomes:
        raise ValueError("kernel search produced no attempts")
    priority = {
        TaskStatus.NO_GAIN: 5,
        TaskStatus.NO_MEASUREMENT: 4,
        TaskStatus.REJECTED: 3,
        TaskStatus.TIMEOUT: 2,
        TaskStatus.INFRASTRUCTURE_ERROR: 1,
    }
    return max(
        outcomes,
        key=lambda item: (priority.get(item.status, 0), item.rank, -item.cycle),
    )


__all__ = [
    "AttemptSession",
    "CompileCorrectnessReceipts",
    "KernelAttemptOutcome",
    "KernelOptimizeRequest",
    "MeasurementEvidence",
    "PreparedCandidate",
    "RunSession",
    "SafetyEvidence",
    "representative_failure",
    "select_best",
]
