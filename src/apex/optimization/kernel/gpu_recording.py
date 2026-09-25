"""Canonical standalone-kernel GPU lifecycle evidence recording."""

from __future__ import annotations

from apex.core import IntegrityError, canonical_json_bytes
from apex.runtime import GpuLeaseHeartbeatReceipt, GpuMeasurementBracketReceipt
from apex.storage import ArtifactReceipt
from apex.runtime.assigned_gpu import AssignedGpuLease, AssignedGpuMeasurement
from apex.runtime import require_gpu_lease_heartbeat


def check_kernel_gpu(run) -> None:
    if not isinstance(run.gpu_lease_guard, AssignedGpuLease):
        require_gpu_lease_heartbeat(run.gpu_lease_guard)


def kernel_gpu_checkpoint(run, attempt_id: str, phase: str) -> ArtifactReceipt:
    # Caller-assigned runs retain one startup record, not per-phase host probes.
    if isinstance(run.gpu_lease_guard, AssignedGpuLease):
        run.record.controller.record_domain_event(
            "dependency_verified",
            {**run.record.attempt_payload(attempt_id), "kind": "assigned_gpu", "phase": phase,
             "artifacts": [{"role": "assigned_gpu", "receipt": run.gpu_lease_artifact.to_dict()}]},
            idempotency_key=f"attempt.{attempt_id}.assigned_gpu.{phase}",
        )
        return run.gpu_lease_artifact
    return record_gpu_lease_heartbeat(
        run.record, require_gpu_lease_heartbeat(run.gpu_lease_guard),
        attempt_id=attempt_id, phase=phase,
    )


def record_gpu_measurement_bracket(
    record,
    bracket: GpuMeasurementBracketReceipt | AssignedGpuMeasurement,
    *,
    attempt_id: str,
    allocation: ArtifactReceipt | None = None,
) -> ArtifactReceipt:
    if bracket.run_id != record.run_id or bracket.action_id != attempt_id:
        raise IntegrityError(
            "GPU measurement bracket targets another attempt",
            "gpu_measurement_bracket_mismatch",
        )
    receipt = record.artifacts.put_bytes(
        canonical_json_bytes(bracket.to_dict()), media_type="application/json"
    )
    record.controller.record_domain_event(
        "dependency_verified",
        {
            **record.attempt_payload(attempt_id),
            "kind": "gpu_measurement_bracket",
            "lease_digest": bracket.lease_digest,
            "bracket_digest": bracket.digest,
            "artifacts": [
                {"role": "gpu_measurement_bracket", "receipt": receipt.to_dict()},
                *([{"role": "assigned_gpu", "receipt": allocation.to_dict()}] if allocation else []),
            ],
        },
        idempotency_key=f"attempt.{attempt_id}.gpu_measurement_bracket",
    )
    return receipt


def record_gpu_lease_heartbeat(
    record,
    heartbeat: GpuLeaseHeartbeatReceipt,
    *,
    attempt_id: str,
    phase: str,
) -> ArtifactReceipt:
    """Bind a post-gate renewal before any low gate reward is committed."""

    if heartbeat.run_id != record.run_id or heartbeat.reason != "manual":
        raise IntegrityError(
            "GPU lease heartbeat targets another run or phase",
            "gpu_lease_heartbeat_mismatch",
        )
    receipt = record.artifacts.put_bytes(
        canonical_json_bytes(heartbeat.to_dict()), media_type="application/json"
    )
    record.controller.record_domain_event(
        "dependency_verified",
        {
            **record.attempt_payload(attempt_id),
            "kind": "gpu_lease_heartbeat",
            "phase": phase,
            "lease_digest": heartbeat.lease_digest,
            "heartbeat_digest": heartbeat.digest,
            "artifacts": [
                {"role": "gpu_lease_heartbeat", "receipt": receipt.to_dict()}
            ],
        },
        idempotency_key=f"attempt.{attempt_id}.gpu_lease_heartbeat.{phase}",
    )
    return receipt


__all__ = ["record_gpu_lease_heartbeat", "record_gpu_measurement_bracket"]
