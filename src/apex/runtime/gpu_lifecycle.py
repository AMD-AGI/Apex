"""Typed renewal and measurement-boundary evidence for a cooperative GPU lease."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from apex.core import ContractError, sha256_json

from .gpu_doctor import GpuDoctorReceipt
from .gpu_ownership import GpuOwnershipReceipt


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_HEARTBEAT_REASONS = frozenset(
    {"acquired", "manual", "measurement_pre", "measurement_post"}
)


@dataclass(frozen=True, slots=True)
class GpuLeaseOwnerIdentity:
    """PID-reuse-resistant identity of the Apex process holding the lock."""

    pid: int
    uid: int
    start_time_ticks: int
    cmdline_sha256: str

    def __post_init__(self) -> None:
        if (
            self.pid <= 0
            or self.uid < 0
            or self.start_time_ticks <= 0
            or not _DIGEST.fullmatch(self.cmdline_sha256)
        ):
            raise ContractError(
                "GPU lease owner identity is incomplete",
                "invalid_gpu_lease_owner_identity",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "uid": self.uid,
            "start_time_ticks": self.start_time_ticks,
            "cmdline_sha256": self.cmdline_sha256,
        }


@dataclass(frozen=True, slots=True)
class GpuLeaseHeartbeatReceipt:
    """One fail-closed renewal after rechecking owner, devices, and health."""

    schema_version: int
    run_id: str
    lease_digest: str
    sequence: int
    reason: str
    observed_unix_seconds: float
    valid_until_unix_seconds: float
    ttl_seconds: float
    owner: GpuLeaseOwnerIdentity
    ownership: GpuOwnershipReceipt
    doctor: GpuDoctorReceipt

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or not self.run_id
            or not _DIGEST.fullmatch(self.lease_digest)
            or self.sequence <= 0
            or self.reason not in _HEARTBEAT_REASONS
            or not _finite_nonnegative(self.observed_unix_seconds)
            or not _finite_positive(self.ttl_seconds)
            or not _finite_nonnegative(self.valid_until_unix_seconds)
            or self.valid_until_unix_seconds
            != self.observed_unix_seconds + self.ttl_seconds
            or self.ownership.foreign_owners
            or self.doctor.ownership != self.ownership
            or not self.doctor.formal_measurement_ready
        ):
            raise ContractError(
                "GPU lease heartbeat is inconsistent",
                "invalid_gpu_lease_heartbeat",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "lease_digest": self.lease_digest,
            "sequence": self.sequence,
            "reason": self.reason,
            "observed_unix_seconds": self.observed_unix_seconds,
            "valid_until_unix_seconds": self.valid_until_unix_seconds,
            "ttl_seconds": self.ttl_seconds,
            "owner": self.owner.to_dict(),
            "ownership": self.ownership.to_dict(),
            "doctor": self.doctor.to_dict(),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class GpuMeasurementBracketReceipt:
    """Pre/post heartbeats enclosing one exact formal measurement action."""

    schema_version: int
    run_id: str
    action_id: str
    lease_digest: str
    started_unix_seconds: float
    finished_unix_seconds: float
    pre: GpuLeaseHeartbeatReceipt
    post: GpuLeaseHeartbeatReceipt

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or not self.run_id
            or not self.action_id
            or not _DIGEST.fullmatch(self.lease_digest)
            or not _finite_nonnegative(self.started_unix_seconds)
            or not _finite_nonnegative(self.finished_unix_seconds)
            or self.finished_unix_seconds < self.started_unix_seconds
            or self.pre.reason != "measurement_pre"
            or self.post.reason != "measurement_post"
            or self.pre.run_id != self.run_id
            or self.post.run_id != self.run_id
            or self.pre.lease_digest != self.lease_digest
            or self.post.lease_digest != self.lease_digest
            or self.post.sequence != self.pre.sequence + 1
            or self.pre.owner != self.post.owner
            or self.pre.ownership.selected_devices
            != self.post.ownership.selected_devices
            or self.pre.ownership.device_inventory
            != self.post.ownership.device_inventory
            or self.pre.observed_unix_seconds > self.started_unix_seconds
            or self.post.observed_unix_seconds < self.finished_unix_seconds
            or self.finished_unix_seconds > self.pre.valid_until_unix_seconds
        ):
            raise ContractError(
                "GPU measurement lease bracket is incomplete",
                "invalid_gpu_measurement_bracket",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "action_id": self.action_id,
            "lease_digest": self.lease_digest,
            "started_unix_seconds": self.started_unix_seconds,
            "finished_unix_seconds": self.finished_unix_seconds,
            "pre": self.pre.to_dict(),
            "post": self.post.to_dict(),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class GpuMeasurementGuard(Protocol):
    receipt: GpuMeasurementBracketReceipt

    def __enter__(self) -> "GpuMeasurementGuard": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


def require_gpu_measurement_guard(
    lease: object, action_id: str
) -> GpuMeasurementGuard:
    """Reject legacy/fake lease objects before any formal measurement starts."""

    factory = getattr(lease, "measurement", None)
    if not callable(factory):
        raise ContractError(
            "GPU lease lacks the required measurement lifecycle",
            "gpu_lease_lifecycle_unavailable",
        )
    guard = factory(action_id)
    if not hasattr(guard, "__enter__") or not hasattr(guard, "__exit__"):
        raise ContractError(
            "GPU measurement guard is invalid",
            "gpu_lease_lifecycle_unavailable",
        )
    return guard


def require_gpu_lease_heartbeat(
    lease: object, reason: str = "manual"
) -> GpuLeaseHeartbeatReceipt:
    """Require a typed live renewal before a non-measurement GPU gate proceeds."""

    heartbeat = getattr(lease, "heartbeat", None)
    if not callable(heartbeat):
        raise ContractError(
            "GPU lease lacks the required heartbeat lifecycle",
            "gpu_lease_lifecycle_unavailable",
        )
    receipt = heartbeat(reason)
    if not isinstance(receipt, GpuLeaseHeartbeatReceipt):
        raise ContractError(
            "GPU lease returned an untyped heartbeat",
            "gpu_lease_lifecycle_unavailable",
        )
    return receipt


def _finite_positive(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value > 0


def _finite_nonnegative(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value >= 0


__all__ = [
    "GpuLeaseHeartbeatReceipt",
    "GpuLeaseOwnerIdentity",
    "GpuMeasurementBracketReceipt",
    "GpuMeasurementGuard",
    "require_gpu_measurement_guard",
    "require_gpu_lease_heartbeat",
]
