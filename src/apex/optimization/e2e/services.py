"""Narrow evaluator ports and receipts used by the E2E candidate state machine."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.core import ContractError, TaskStatus, ValidationLevel, sha256_file
from apex.evaluation import E2EMeasurement, KernelGrade, MeasurementStatus
from apex.evaluation.safety import (
    ArtifactKind,
    FindingStatus,
    FrozenCandidate,
    InstrumentationControl,
    KernelLanguage,
    PhaseIsolationReceipt,
    SafetyGate,
    SafetyGateRequest,
    SafetyGateResult,
    TaskSafetyProfile,
    VerificationPlan,
    VerificationPolicy,
)
from apex.runtime import RunProvenance

from .candidate import E2ECandidate, materialize_frozen_sources, validate_frozen_sources
from .kernel_lane import KernelOpportunity


_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class MicroQualificationRequest:
    run_id: str
    candidate: E2ECandidate
    opportunity: KernelOpportunity
    artifact_root: Path
    anchor_generation: int
    gpu_device_scope: str


@dataclass(frozen=True, slots=True)
class MicroQualification:
    """Micro verdict derived from the canonical kernel grader, or honestly deferred."""

    candidate_id: str
    grade: KernelGrade | None
    evidence: Mapping[str, Any]
    qualification_mode: str = "strict_micro"
    deferred_candidate_valid: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ContractError("Micro candidate identity is empty", "invalid_micro_qualification")
        if self.qualification_mode not in {"strict_micro", "e2e_quality_deferred"}:
            raise ContractError(
                "Unknown micro qualification mode", "invalid_micro_qualification"
            )
        if self.qualification_mode == "e2e_quality_deferred":
            if self.grade is not None:
                raise ContractError(
                    "Deferred qualification cannot carry a kernel grade",
                    "invalid_micro_qualification",
                )
            return
        if not isinstance(self.grade, KernelGrade) or self.deferred_candidate_valid:
            raise ContractError(
                "Strict micro qualification requires one canonical kernel grade",
                "invalid_micro_qualification",
            )

    @property
    def compiled(self) -> bool | None:
        return self.grade.gates.compiled if self.grade else None

    @property
    def correct(self) -> bool | None:
        return self.grade.gates.correct if self.grade else None

    @property
    def integrity_passed(self) -> bool:
        return self.grade.gates.integrity_passed if self.grade else self.deferred_candidate_valid

    @property
    def performance_valid(self) -> bool:
        return bool(self.grade and self.grade.measurement_status is MeasurementStatus.VALID)

    @property
    def s50(self) -> float | None:
        return self.grade.s50 if self.grade else None

    @property
    def s99(self) -> float | None:
        return self.grade.s99 if self.grade else None

    @property
    def srobust(self) -> float | None:
        return self.grade.srobust if self.grade else None

    @property
    def sample_count(self) -> int:
        counts = [case.reference.sample_count for case in self.grade.cases] if self.grade else []
        return min(counts) if counts else 0

    @property
    def reason_code(self) -> str:
        if self.grade is not None:
            return self.grade.promotion_reason_code
        observed = self.evidence.get("reason_code")
        if isinstance(observed, str) and observed:
            return observed
        return "e2e_quality_deferred" if self.deferred_candidate_valid else "invalid_frozen_candidate"

    @property
    def qualified(self) -> bool:
        if self.qualification_mode == "e2e_quality_deferred":
            return self.deferred_candidate_valid
        return bool(self.grade and self.grade.promotion_eligible)

    @property
    def kernel_reward_available(self) -> bool:
        """Whether a trusted raw-sample micro grader can issue kernel reward."""

        return bool(
            self.qualification_mode == "strict_micro"
            and self.grade
            and self.grade.measurement_status is MeasurementStatus.VALID
            and self.grade.reward is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "qualification_mode": self.qualification_mode,
            "deferred_candidate_valid": self.deferred_candidate_valid,
            "qualified": self.qualified,
            "reason_code": self.reason_code,
            "kernel_reward_available": self.kernel_reward_available,
            "grade": self.grade.to_dict() if self.grade else None,
            "evidence": dict(self.evidence),
        }


class MicroQualificationPort(Protocol):
    def supports(self, opportunity: KernelOpportunity) -> bool: ...

    def verify(self, request: MicroQualificationRequest) -> MicroQualification: ...


@dataclass(frozen=True, slots=True)
class SafetyQualificationRequest:
    run_id: str
    candidate: E2ECandidate
    opportunity: KernelOpportunity
    artifact_root: Path
    anchor_generation: int


@dataclass(frozen=True, slots=True)
class SafetyQualification:
    candidate_id: str
    allowed_to_measure: bool
    promotion_eligible: bool
    safety_certified: bool
    finding: bool
    reason_codes: tuple[str, ...]
    evidence: Mapping[str, Any]

    @property
    def qualified(self) -> bool:
        return self.allowed_to_measure and self.promotion_eligible and not self.finding

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CandidateSafetyPort(Protocol):
    @property
    def policy_fingerprint(self) -> str | None: ...

    def verify(self, request: SafetyQualificationRequest) -> SafetyQualification: ...


class _NoToolRunner:
    def run(self, _request: object) -> object:
        raise AssertionError("no-tool safety policy must not invoke a runner")


class NoToolSafetyVerifier:
    """Real freeze/isolation gate for deployments without configured sanitizers."""

    @property
    def policy_fingerprint(self) -> str:
        """Bind formal delivery to the exact evaluator-owned no-tool policy."""

        return VerificationPolicy.no_tools().fingerprint

    def verify(self, request: SafetyQualificationRequest) -> SafetyQualification:
        candidate = request.candidate
        if candidate.candidate_id is None or candidate.candidate_source_sha256 is None:
            raise ContractError("Safety requires a frozen candidate", "invalid_frozen_candidate")
        validate_frozen_sources(candidate)
        artifact_root = _prepare_safety_artifact_root(request.artifact_root)
        snapshot_root = materialize_frozen_sources(
            candidate, artifact_root / "frozen-candidate"
        )
        evidence_root = artifact_root / "evidence"
        language = {
            "python": KernelLanguage.PYTHON,
            "triton": KernelLanguage.TRITON,
        }.get(request.opportunity.language, KernelLanguage.UNKNOWN)
        profile = TaskSafetyProfile(
            language=language,
            artifact_kind=ArtifactKind.PYTHON_JIT,
            instrumentation_control=(
                InstrumentationControl.COMPILER_CONTROLLED
                if language is KernelLanguage.TRITON
                else InstrumentationControl.NONE
            ),
            submission_paths=tuple(sorted(candidate.editable_files)),
            target_symbols=(request.opportunity.runtime_name,),
        )
        frozen = FrozenCandidate.capture(snapshot_root, profile)
        policy = VerificationPolicy.no_tools()
        plan = VerificationPlan.create(
            run_id=request.run_id,
            candidate_id=candidate.candidate_id,
            anchor_generation=request.anchor_generation,
            profile=profile,
            policy=policy,
            source_digest=candidate.baseline_source_sha256,
            candidate_digest=frozen.candidate_digest,
            deployed_digest=frozen.candidate_digest,
        )
        isolation = PhaseIsolationReceipt(
            run_id=request.run_id,
            plan_fingerprint=plan.fingerprint,
            anchor_generation=request.anchor_generation,
            candidate_digest=frozen.candidate_digest,
            frozen_root=str(snapshot_root),
            evaluator_artifact_root=str(evidence_root),
            agent_process_tree_terminated=True,
            credentials_revoked=True,
            tool_channels_revoked=True,
            report_directory_hidden_from_agent=True,
            candidate_read_only=True,
        )
        result: SafetyGateResult = SafetyGate(_NoToolRunner()).evaluate(
            SafetyGateRequest(
                plan=plan,
                policy=policy,
                frozen_candidate=frozen,
                isolation_receipt=isolation,
                artifact_root=evidence_root,
                current_run_id=request.run_id,
                current_candidate_id=candidate.candidate_id,
                current_anchor_generation=request.anchor_generation,
                current_deployed_digest=frozen.candidate_digest,
            )
        )
        return SafetyQualification(
            candidate_id=candidate.candidate_id,
            allowed_to_measure=result.decision.allowed_to_measure,
            promotion_eligible=result.decision.promotion_eligible,
            safety_certified=result.decision.safety_certified,
            finding=any(item.finding is FindingStatus.FOUND for item in result.evaluations),
            reason_codes=result.decision.reason_codes,
            evidence=result.to_dict(),
        )


def _prepare_safety_artifact_root(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise ContractError("Safety artifact root is unsafe", "invalid_safety_artifact_root")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise ContractError("Safety artifact root is unsafe", "invalid_safety_artifact_root")
    return path.resolve()


@dataclass(frozen=True, slots=True)
class CandidateDeploymentRequest:
    run_id: str
    candidate: E2ECandidate
    opportunity: KernelOpportunity
    provenance: RunProvenance
    benchmark_measurement: Path
    benchmark_diagnostic: Path
    workload_semantics_sha256: str
    artifact_root: Path
    anchor_generation: int
    safety: SafetyQualification
    benchmark_replay: Path | None = None
    accepted_stack: tuple[AcceptedCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class DeploymentConfigDigests:
    """Exact benchmark-config bytes derived for one immutable image."""

    measurement: str
    diagnostic: str
    replay: str

    def __post_init__(self) -> None:
        if any(
            not _SHA256.fullmatch(value)
            for value in (self.measurement, self.diagnostic, self.replay)
        ):
            raise ContractError(
                "Deployment config digest is invalid",
                "invalid_deployment_config_digest",
            )

    @classmethod
    def capture(
        cls, measurement: Path, diagnostic: Path, replay: Path
    ) -> "DeploymentConfigDigests":
        return cls(
            sha256_file(measurement),
            sha256_file(diagnostic),
            sha256_file(replay),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateDeployment:
    """Primary source-rebuild/overlay deployment bound to exact candidate bytes."""

    candidate_id: str
    deployed: bool
    reason_code: str
    measurement_config: Path
    diagnostic_config: Path
    replay_config: Path
    workload_semantics_sha256: str
    deployed_source_sha256: str
    deployed_image_id: str | None
    validation_level: ValidationLevel
    engagement_verified: bool
    evidence: Mapping[str, Any]
    infrastructure_failure: bool = False
    config_sha256: DeploymentConfigDigests | None = None

    def __post_init__(self) -> None:
        derived = self.evidence.get("derived_image")
        evidence_image_id = (
            derived.get("image_id") if isinstance(derived, Mapping) else None
        )
        if self.deployed:
            if (
                self.deployed_image_id is None
                or not _IMAGE_ID.fullmatch(self.deployed_image_id)
                or evidence_image_id != self.deployed_image_id
                or self.config_sha256 is None
                or self.evidence.get("config_sha256")
                != self.config_sha256.to_dict()
            ):
                raise ContractError(
                    "Deployment lacks immutable image/config identity",
                    "invalid_deployment_identity",
                )
        elif self.deployed_image_id is not None or self.config_sha256 is not None:
            raise ContractError(
                "Failed deployment cannot claim deployed identities",
                "invalid_deployment_identity",
            )

    @property
    def qualified(self) -> bool:
        return self.deployed and self.engagement_verified

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["measurement_config"] = str(self.measurement_config)
        value["diagnostic_config"] = str(self.diagnostic_config)
        value["replay_config"] = str(self.replay_config)
        value["validation_level"] = self.validation_level.value
        value["config_sha256"] = (
            self.config_sha256.to_dict() if self.config_sha256 else None
        )
        return value


class CandidateDeploymentPort(Protocol):
    def supports(self, opportunity: KernelOpportunity, provenance: RunProvenance) -> bool: ...

    def deploy(self, request: CandidateDeploymentRequest) -> CandidateDeployment: ...

    def rollback(self, deployment: CandidateDeployment) -> None: ...


@dataclass(frozen=True, slots=True)
class AcceptedCandidate:
    candidate: E2ECandidate
    opportunity: KernelOpportunity
    micro: MicroQualification
    safety: SafetyQualification
    deployment: CandidateDeployment
    primary_measurement: E2EMeasurement
    decision_receipt: str


@dataclass(frozen=True, slots=True)
class FinalDeliveryRequest:
    run_id: str
    accepted: tuple[AcceptedCandidate, ...]
    provenance: RunProvenance
    benchmark_original: Path
    benchmark_measurement: Path
    benchmark_diagnostic: Path
    benchmark_replay: Path
    baseline: E2EMeasurement
    final: E2EMeasurement
    artifact_root: Path
    agent_backend: str | None = None
    agent_model: str | None = None
    accuracy_policy_sha256: str | None = None
    performance_policy_sha256: str | None = None
    safety_policy_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FinalDeliveryResult:
    """Second-environment delivery verdict; formal success has one shape only."""

    verified: bool
    status: TaskStatus
    reason_code: str
    validation_level: ValidationLevel
    clean_replay_verified: bool
    bundle_path: str | None
    bundle_digest: str | None
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        formal = (
            self.verified
            and self.clean_replay_verified
            and self.status is TaskStatus.SUCCEEDED
            and self.validation_level is ValidationLevel.SOURCE_REBUILD_VERIFIED
            and bool(self.bundle_path)
            and bool(self.bundle_digest)
        )
        if self.verified != formal or (self.status is TaskStatus.SUCCEEDED) != formal:
            raise ContractError(
                "Formal E2E success requires source rebuild and second clean replay",
                "invalid_final_delivery_verdict",
            )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["validation_level"] = self.validation_level.value
        return value


class FinalDeliveryPort(Protocol):
    def finalize(self, request: FinalDeliveryRequest) -> FinalDeliveryResult: ...


class UnavailableMicroQualifier:
    def supports(self, _opportunity: KernelOpportunity) -> bool:
        return False

    def verify(self, _request: MicroQualificationRequest) -> MicroQualification:
        raise ContractError("No trusted micro verifier is configured", "micro_verifier_unavailable")


class UnavailableDeployment:
    def supports(self, _opportunity: KernelOpportunity, _provenance: RunProvenance) -> bool:
        return False

    def deploy(self, _request: CandidateDeploymentRequest) -> CandidateDeployment:
        raise ContractError("No source deployment adapter is configured", "delivery_adapter_unavailable")

    def rollback(self, _deployment: CandidateDeployment) -> None:
        return None


class UnavailableFinalDelivery:
    def finalize(self, request: FinalDeliveryRequest) -> FinalDeliveryResult:
        status = (
            TaskStatus.PROVENANCE_UNRESOLVED
            if not request.provenance.source_delivery_ready
            else TaskStatus.VERIFICATION_FAILED
        )
        reason = (
            "source_provenance_unresolved"
            if status is TaskStatus.PROVENANCE_UNRESOLVED
            else "final_delivery_adapter_unavailable"
        )
        level = (
            ValidationLevel.RUNTIME_OVERLAY_VERIFIED
            if any(
                item.deployment.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
                for item in request.accepted
            )
            else ValidationLevel.NONE
        )
        return FinalDeliveryResult(False, status, reason, level, False, None, None, {})


__all__ = [
    "AcceptedCandidate",
    "CandidateDeployment",
    "CandidateDeploymentPort",
    "CandidateDeploymentRequest",
    "CandidateSafetyPort",
    "FinalDeliveryPort",
    "FinalDeliveryRequest",
    "FinalDeliveryResult",
    "MicroQualification",
    "MicroQualificationPort",
    "MicroQualificationRequest",
    "NoToolSafetyVerifier",
    "SafetyQualification",
    "SafetyQualificationRequest",
    "UnavailableDeployment",
    "UnavailableFinalDelivery",
    "UnavailableMicroQualifier",
]
