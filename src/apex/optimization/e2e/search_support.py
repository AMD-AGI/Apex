"""Small immutable records and validation helpers for the E2E search loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apex.benchmark import BenchmarkConfigViews, NormalizedBenchmarkResult
from apex.core import ContractError, IntegrityError, sha256_file
from apex.orchestration import SearchStage
from apex.storage import ArtifactReceipt

from .benchmarking import Diagnosis
from .candidate import E2ECandidate
from .kernel_lane import KernelOpportunity
from .outcomes import commit_e2e_reject
from .run_record import E2ERunRecord
from .services import (
    AcceptedCandidate,
    CandidateDeployment,
    MicroQualification,
    SafetyQualification,
)


_AGENT_TEARDOWN_INFRASTRUCTURE_FAILURES = {
    "agent_process_cleanup_failed",
    "agent_process_containment_unverified",
}


@dataclass(frozen=True, slots=True)
class QualifiedAttempt:
    attempt_id: str
    opportunity: KernelOpportunity
    candidate: E2ECandidate
    candidate_receipt: ArtifactReceipt
    micro: MicroQualification
    micro_receipt: ArtifactReceipt
    safety: SafetyQualification
    safety_receipt: ArtifactReceipt
    deployment: CandidateDeployment
    delivery_receipt: ArtifactReceipt


def raise_agent_teardown_infrastructure(
    candidate: E2ECandidate,
    receipt: ArtifactReceipt,
) -> None:
    if candidate.reason_code not in _AGENT_TEARDOWN_INFRASTRUCTURE_FAILURES:
        return
    result = candidate.agent_result
    raise IntegrityError(
        "Agent process teardown could not be verified",
        candidate.reason_code,
        {
            "attempt_id": candidate.attempt_id,
            "candidate_manifest_receipt": receipt.digest,
            "capture_status": result.capture_status.value,
            "termination_kind": result.termination_kind.value,
            "termination_reason": result.termination_reason,
            "process_containment": (
                result.process_containment.to_dict()
                if result.process_containment is not None
                else None
            ),
        },
    )


def opportunity_map(diagnosis: Diagnosis) -> dict[str, KernelOpportunity]:
    return {item.opportunity_id: item for item in diagnosis.plan.opportunities}


def candidate_configs(candidate: AcceptedCandidate) -> tuple[Path, Path, Path]:
    deployment = candidate.deployment
    return (
        deployment.measurement_config,
        deployment.diagnostic_config,
        deployment.replay_config,
    )


def candidate_id(candidate: E2ECandidate) -> str:
    if candidate.candidate_id is None:
        raise ContractError("Candidate is not frozen", "invalid_frozen_candidate")
    return candidate.candidate_id


def source_key(opportunity: KernelOpportunity) -> str:
    if opportunity.source_root is None or opportunity.source_path is None:
        raise ContractError("Kernel source is unresolved", "source_unresolved")
    root = opportunity.source_root.resolve(strict=True)
    relative = opportunity.source_path.resolve(strict=True).relative_to(root)
    return f"{root}:{relative.as_posix()}"


def search_stage(record: E2ERunRecord) -> SearchStage:
    search = record.controller.state.e2e
    if search is None:
        raise ContractError("E2E state is not initialized", "e2e_not_initialized")
    return search.stage


def validate_deployment(
    deployment: CandidateDeployment,
    candidate: E2ECandidate,
    views: BenchmarkConfigViews,
) -> None:
    if candidate.candidate_id != deployment.candidate_id:
        raise ContractError("Deployment targets another candidate", "candidate_id_mismatch")
    if deployment.workload_semantics_sha256 != views.workload_semantics_sha256:
        raise ContractError("Deployment changed workload semantics", "benchmark_semantics_changed")
    if not deployment.deployed:
        return
    if deployment.deployed_source_sha256 != candidate.candidate_source_sha256:
        raise ContractError("Deployed source differs from frozen candidate", "candidate_lineage_mismatch")
    for path in (
        deployment.measurement_config,
        deployment.diagnostic_config,
        deployment.replay_config,
    ):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ContractError("Deployment config is missing or unsafe", "invalid_replay_config")
    digests = deployment.config_sha256
    if digests is None:
        raise ContractError(
            "Deployment config digests are missing",
            "invalid_deployment_config_digest",
        )
    for path, expected in zip(
        (
            deployment.measurement_config,
            deployment.diagnostic_config,
            deployment.replay_config,
        ),
        (digests.measurement, digests.diagnostic, digests.replay),
        strict=True,
    ):
        if sha256_file(path) != expected:
            raise IntegrityError(
                "Deployment config differs from its immutable receipt",
                "deployment_config_digest_mismatch",
            )


def validate_candidate_runtime(
    result: NormalizedBenchmarkResult,
    deployment: CandidateDeployment,
) -> None:
    """Bind the actual Magpie serving container to the deployed immutable image."""

    runtime = result.serving_runtime
    expected = deployment.deployed_image_id
    config_digests = deployment.config_sha256
    measurement_digest = (
        config_digests.measurement if config_digests is not None else None
    )
    if (
        expected is None
        or runtime.required is not True
        or runtime.requested_image != expected
        or runtime.resolved_image_id != expected
        or (result.succeeded and not runtime.passed)
    ):
        raise IntegrityError(
            "Candidate benchmark did not execute the deployed immutable image",
            "candidate_runtime_image_mismatch",
            {
                "expected_image_id": expected,
                "requested_image": runtime.requested_image,
                "resolved_image_id": runtime.resolved_image_id,
                "serving_runtime_required": runtime.required,
                "serving_runtime_passed": runtime.passed,
                "serving_runtime_error": runtime.error,
            },
        )
    if (
        measurement_digest is None
        or sha256_file(deployment.measurement_config) != measurement_digest
        or runtime.input_config_sha256 != measurement_digest
    ):
        raise IntegrityError(
            "Candidate benchmark did not execute the deployed config bytes",
            "candidate_runtime_config_mismatch",
            {
                "expected_config_sha256": measurement_digest,
                "runtime_config_sha256": runtime.input_config_sha256,
            },
        )


def qualified_receipts(
    attempt: QualifiedAttempt,
    benchmark_receipt: ArtifactReceipt,
) -> dict[str, str]:
    return {
        "micro_receipt": attempt.micro_receipt.digest,
        "safety_receipt": attempt.safety_receipt.digest,
        "delivery_receipt": attempt.delivery_receipt.digest,
        "benchmark_receipt": benchmark_receipt.digest,
    }


def qualified_artifacts(
    attempt: QualifiedAttempt,
    benchmark_receipt: ArtifactReceipt,
) -> tuple[tuple[str, ArtifactReceipt], ...]:
    return (
        ("micro_qualification", attempt.micro_receipt),
        ("safety_qualification", attempt.safety_receipt),
        ("primary_delivery", attempt.delivery_receipt),
        ("normalized_benchmark", benchmark_receipt),
    )


def promotion_receipts(
    attempt: QualifiedAttempt,
    promotion_receipt: ArtifactReceipt,
) -> dict[str, str]:
    return {
        "micro_receipt": attempt.micro_receipt.digest,
        "safety_receipt": attempt.safety_receipt.digest,
        "delivery_receipt": attempt.delivery_receipt.digest,
        "promotion_pair_receipt": promotion_receipt.digest,
    }


def promotion_artifacts(
    attempt: QualifiedAttempt,
    promotion_receipt: ArtifactReceipt,
) -> tuple[tuple[str, ArtifactReceipt], ...]:
    return (
        ("micro_qualification", attempt.micro_receipt),
        ("safety_qualification", attempt.safety_receipt),
        ("primary_delivery", attempt.delivery_receipt),
        ("matched_promotion_pair", promotion_receipt),
    )


def commit_qualified_reject(
    record: E2ERunRecord,
    attempt: QualifiedAttempt,
    benchmark_receipt: ArtifactReceipt,
    reason: str,
) -> None:
    """Close a fully qualified attempt whose E2E measurement cannot win."""

    commit_e2e_reject(
        record,
        attempt_id=attempt.attempt_id,
        opportunity_id=attempt.opportunity.opportunity_id,
        candidate_id=candidate_id(attempt.candidate),
        candidate_manifest=attempt.candidate_receipt,
        reason=reason,
        evidence_receipts=qualified_receipts(attempt, benchmark_receipt),
        evidence_artifacts=qualified_artifacts(attempt, benchmark_receipt),
    )


__all__ = [
    "QualifiedAttempt",
    "candidate_configs",
    "candidate_id",
    "commit_qualified_reject",
    "opportunity_map",
    "promotion_artifacts",
    "promotion_receipts",
    "qualified_artifacts",
    "qualified_receipts",
    "raise_agent_teardown_infrastructure",
    "search_stage",
    "source_key",
    "validate_candidate_runtime",
    "validate_deployment",
]
