"""Small integrity helpers shared by the E2E composition root."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from apex.benchmark import BenchmarkConfigViews
from apex.core import ContractError, TaskStatus, sha256_json
from apex.intake import E2EOptimizeSpec
from apex.orchestration import RunPhase
from apex.runtime import GpuLeaseReceipt, MagpieConfigContract
from apex.storage import ArtifactReceipt
from apex.storage import EventRecord

from .oracles import CorrectnessOracleRegistry
from .recovery import RecoveredRunRequest


def relocate_views(
    staged: BenchmarkConfigViews, destination: Path
) -> BenchmarkConfigViews:
    return replace(
        staged,
        original=destination / staged.original.name,
        measurement=destination / staged.measurement.name,
        diagnostic=destination / staged.diagnostic.name,
        replay=destination / staged.replay.name,
    )


def verify_resume_gpu_scope(
    request: RecoveredRunRequest, gpu_lease: GpuLeaseReceipt
) -> None:
    if gpu_lease.execution_scope != request.gpu_device_scope:
        raise ContractError(
            "GPU execution scope differs from the interrupted run",
            "resume_gpu_scope_mismatch",
        )


def verify_resume_gpu_lease(
    request: RecoveredRunRequest,
    gpu_lease: GpuLeaseReceipt,
    events: tuple[EventRecord, ...],
) -> None:
    """Require the same devices under a newly acquired, never-reused lease."""

    verify_resume_gpu_scope(request, gpu_lease)
    acquired = tuple(
        event
        for event in events
        if event.payload.get("kind") == "gpu_lease"
    )
    if len(acquired) != 1:
        raise ContractError(
            "Interrupted run has no unique original GPU lease",
            "resume_gpu_lease_history_invalid",
        )
    previous = acquired[0].payload.get("lease_digest")
    if not isinstance(previous, str) or previous == gpu_lease.digest:
        raise ContractError(
            "Resume cannot reuse the interrupted run's old GPU lease",
            "resume_gpu_lease_reused",
        )


def require_optimizable_contract(resolved: MagpieConfigContract) -> None:
    if resolved.status != "config_compatible":
        raise ContractError(
            "Magpie resolved plan requires a capability upgrade",
            "capability_upgrade_required",
            {"blockers": list(resolved.capability_receipt["blockers"])},
        )
    require_docker_one_shot_contract(resolved)
    if resolved.capability_receipt["optimization_applicable"] is not True:
        raise ContractError(
            "Magpie cleanup lifecycle is not an independent optimization task",
            "cleanup_lifecycle_not_optimizable",
        )


def require_docker_one_shot_contract(resolved: MagpieConfigContract) -> None:
    """Keep the V2 product boundary ahead of provenance, GPU, and agent work."""

    identity = resolved.plan["identity"]
    run_mode = str(identity["run_mode"])
    lifecycle = str(resolved.plan["lifecycle"])
    if run_mode != "docker" or lifecycle != "one_shot":
        raise ContractError(
            "Apex E2E V2 supports Docker one-shot workloads only",
            "e2e_docker_only",
            {"run_mode": run_mode, "lifecycle": lifecycle},
        )


def objective_hash(spec: E2EOptimizeSpec) -> str:
    return sha256_json(spec.to_dict()["goal"])


def accuracy_hash(
    views: BenchmarkConfigViews,
    spec: E2EOptimizeSpec,
    correctness_oracles: CorrectnessOracleRegistry | None = None,
) -> str:
    policy = {
        "schema": "apex.e2e-quality-policy-binding.v1",
        "quality_tasks": views.quality_tasks,
        "evaluator_policy_sha256": views.evaluator_policy_sha256,
        "regression_gates": asdict(spec.goal.gates),
    }
    if correctness_oracles is not None:
        policy["correctness_oracle_policy_sha256"] = (
            correctness_oracles.policy_sha256
        )
    return sha256_json(policy)


def artifact_binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def verify_terminal_phase(phase: RunPhase, status: TaskStatus) -> None:
    successful = {TaskStatus.SUCCEEDED, TaskStatus.NO_GAIN}
    if (phase is RunPhase.SUCCEEDED) != (status in successful):
        raise ContractError(
            "Terminal result conflicts with run state",
            "e2e_result_binding_mismatch",
        )


__all__ = [
    "accuracy_hash",
    "artifact_binding",
    "objective_hash",
    "relocate_views",
    "require_docker_one_shot_contract",
    "require_optimizable_contract",
    "verify_resume_gpu_scope",
    "verify_resume_gpu_lease",
    "verify_terminal_phase",
]
