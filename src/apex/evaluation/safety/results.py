"""Normalized safety evidence and gate result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from apex.core import ContractError, canonical_json_bytes, sha256_json

from .plan import ToolRuntimeIdentity, validate_sha256
from .policy import SafetyDecision
from .profile import CapabilityStatus, normalize_relative_path


RESULT_SCHEMA_VERSION = "apex.safety-gate-result/v1"
TOOL_REPORT_SCHEMA_VERSION = "apex.safety-tool-report/v1"


class ExecutionStatus(str, Enum):
    NOT_RUN = "not_run"
    COMPLETED = "completed"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return self.value


class FindingStatus(str, Enum):
    NOT_EVALUATED = "not_evaluated"
    CLEAN = "clean"
    FOUND = "found"
    INCONCLUSIVE = "inconclusive"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class LineageReceipt:
    """Exact identity/digest chain accepted from an evaluator-owned report."""

    identity: ToolRuntimeIdentity
    source_digest: str
    candidate_digest: str
    deployed_digest: str
    positive_control_digest: str
    plan_fingerprint: str

    def __post_init__(self) -> None:
        validate_sha256(self.source_digest, field="source_digest")
        validate_sha256(self.candidate_digest, field="candidate_digest")
        validate_sha256(self.deployed_digest, field="deployed_digest")
        validate_sha256(self.positive_control_digest, field="positive_control_digest")
        validate_sha256(self.plan_fingerprint, field="plan_fingerprint")

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "source_digest": self.source_digest,
            "candidate_digest": self.candidate_digest,
            "deployed_digest": self.deployed_digest,
            "positive_control_digest": self.positive_control_digest,
            "plan_fingerprint": self.plan_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class EvidenceArtifactReceipt:
    """Checksummed evaluator artifact.  Safety artifacts are diagnostic-only."""

    role: str
    path: str
    digest: str
    size: int
    producer: str = "evaluator"
    timing_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.role.strip() or self.producer != "evaluator":
            raise ContractError("invalid safety artifact receipt", "invalid_safety_evidence")
        object.__setattr__(self, "path", normalize_relative_path(self.path, field="artifact_path"))
        validate_sha256(self.digest, field="artifact_digest")
        if self.size < 0 or self.timing_eligible:
            raise ContractError(
                "sanitizer artifacts are diagnostic-only and can never provide timing",
                reason_code="sanitizer_timing_forbidden",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path,
            "digest": self.digest,
            "size": self.size,
            "producer": self.producer,
            "timing_eligible": False,
        }


@dataclass(frozen=True, slots=True)
class ToolEvaluation:
    """Capability/execution/finding remain independent in all outputs."""

    tool: str
    capability: CapabilityStatus
    execution: ExecutionStatus
    finding: FindingStatus
    reason_codes: tuple[str, ...]
    lineage: LineageReceipt | None = None
    artifacts: tuple[EvidenceArtifactReceipt, ...] = ()
    exit_code: int | None = None
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_digest: str | None = None
    stderr_digest: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        try:
            object.__setattr__(self, "capability", CapabilityStatus(str(self.capability)))
            object.__setattr__(self, "execution", ExecutionStatus(str(self.execution)))
            object.__setattr__(self, "finding", FindingStatus(str(self.finding)))
        except ValueError as exc:
            raise ContractError("invalid safety result state", "invalid_safety_evidence") from exc
        if self.finding is FindingStatus.CLEAN:
            if (
                self.capability is not CapabilityStatus.READY
                or self.execution is not ExecutionStatus.COMPLETED
                or self.lineage is None
                or not self.artifacts
                or self.stdout_truncated
                or self.stderr_truncated
            ):
                raise ContractError(
                    "clean requires complete ready execution and exact lineage",
                    reason_code="false_clean_safety_evidence",
                )
        if self.capability is CapabilityStatus.NOT_APPLICABLE and (
            self.execution is not ExecutionStatus.NOT_RUN
            or self.finding is not FindingStatus.NOT_EVALUATED
        ):
            raise ContractError(
                "not_applicable is not a clean execution",
                reason_code="not_applicable_is_not_clean",
            )
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ContractError("invalid safety duration", "invalid_safety_evidence")
        for digest in (self.stdout_digest, self.stderr_digest):
            if digest is not None:
                validate_sha256(digest, field="log_digest")

    @property
    def diagnostic_only(self) -> bool:
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "capability": self.capability.value,
            "execution": self.execution.value,
            "finding": self.finding.value,
            "reason_codes": list(self.reason_codes),
            "lineage": self.lineage.to_dict() if self.lineage is not None else None,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "process": {
                "exit_code": self.exit_code,
                "timed_out": self.timed_out,
                "stdout_truncated": self.stdout_truncated,
                "stderr_truncated": self.stderr_truncated,
                "stdout_digest": self.stdout_digest,
                "stderr_digest": self.stderr_digest,
                "duration_seconds": self.duration_seconds,
            },
            "diagnostic_only": True,
        }


@dataclass(frozen=True, slots=True)
class SafetyGateResult:
    schema_version: str
    run_id: str
    candidate_id: str
    anchor_generation: int
    plan_fingerprint: str
    policy_fingerprint: str
    source_digest: str
    candidate_digest: str
    deployed_digest: str
    isolation_receipt_fingerprint: str
    evaluations: tuple[ToolEvaluation, ...]
    decision: SafetyDecision
    gate_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ContractError("unsupported safety result schema", "unsupported_safety_schema")
        if self.anchor_generation < 0:
            raise ContractError("invalid safety result generation", "invalid_safety_evidence")
        validate_sha256(self.plan_fingerprint, field="plan_fingerprint")
        validate_sha256(self.policy_fingerprint, field="policy_fingerprint")
        validate_sha256(self.source_digest, field="source_digest")
        validate_sha256(self.candidate_digest, field="candidate_digest")
        validate_sha256(self.deployed_digest, field="deployed_digest")
        validate_sha256(self.isolation_receipt_fingerprint, field="isolation_receipt_fingerprint")
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        object.__setattr__(self, "gate_errors", tuple(self.gate_errors))
        tools = tuple(evaluation.tool for evaluation in self.evaluations)
        if tools != tuple(sorted(set(tools))):
            raise ContractError("safety evaluations must be unique and sorted", "invalid_safety_evidence")

    @property
    def safety_certified(self) -> bool:
        return self.decision.safety_certified

    @property
    def forbidden_timing_digests(self) -> tuple[str, ...]:
        return tuple(
            sorted({artifact.digest for evaluation in self.evaluations for artifact in evaluation.artifacts})
        )

    def assert_performance_artifact_allowed(self, digest: str) -> None:
        validate_sha256(digest, field="performance_artifact_digest")
        if digest in self.forbidden_timing_digests:
            raise ContractError(
                "instrumented sanitizer artifacts cannot be used for timing",
                reason_code="sanitizer_timing_forbidden",
            )

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._body())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "anchor_generation": self.anchor_generation,
            "plan_fingerprint": self.plan_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "source_digest": self.source_digest,
            "candidate_digest": self.candidate_digest,
            "deployed_digest": self.deployed_digest,
            "isolation_receipt_fingerprint": self.isolation_receipt_fingerprint,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "decision": self.decision.to_dict(),
            "gate_errors": list(self.gate_errors),
            "forbidden_timing_digests": list(self.forbidden_timing_digests),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "result_fingerprint": self.fingerprint}


def parse_execution_status(value: object) -> ExecutionStatus:
    try:
        return ExecutionStatus(str(value))
    except ValueError as exc:
        raise ContractError("invalid execution status", "invalid_safety_evidence") from exc


def parse_finding_status(value: object) -> FindingStatus:
    try:
        return FindingStatus(str(value))
    except ValueError as exc:
        raise ContractError("invalid finding status", "invalid_safety_evidence") from exc


__all__ = [
    "EvidenceArtifactReceipt",
    "ExecutionStatus",
    "FindingStatus",
    "LineageReceipt",
    "RESULT_SCHEMA_VERSION",
    "SafetyGateResult",
    "TOOL_REPORT_SCHEMA_VERSION",
    "ToolEvaluation",
    "parse_execution_status",
    "parse_finding_status",
]
