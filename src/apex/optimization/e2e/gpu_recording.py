"""Canonical E2E GPU heartbeat recording outside scoring observations."""

from __future__ import annotations

from apex.core import IntegrityError, canonical_json_bytes
from apex.runtime import GpuLeaseHeartbeatReceipt
from apex.storage import ArtifactReceipt


def record_gpu_lease_heartbeat(
    record,
    heartbeat: GpuLeaseHeartbeatReceipt,
    *,
    action_id: str,
) -> ArtifactReceipt:
    if heartbeat.run_id != record.run_id or heartbeat.reason != "manual":
        raise IntegrityError(
            "E2E GPU heartbeat targets another run",
            "gpu_lease_heartbeat_mismatch",
        )
    receipt = record.artifacts.put_bytes(
        canonical_json_bytes(heartbeat.to_dict()), media_type="application/json"
    )
    record.controller.record_domain_event(
        "dependency_verified",
        {
            "kind": "gpu_lease_heartbeat",
            "action_id": action_id,
            "lease_digest": heartbeat.lease_digest,
            "heartbeat_digest": heartbeat.digest,
            "artifacts": [
                {"role": "gpu_lease_heartbeat", "receipt": receipt.to_dict()}
            ],
        },
        idempotency_key=f"gpu_lease_heartbeat.{action_id}",
    )
    return receipt


__all__ = ["record_gpu_lease_heartbeat"]
