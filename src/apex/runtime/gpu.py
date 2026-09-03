"""Run-scoped physical GPU lease with fail-closed ownership evidence."""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from apex.core import ContractError, canonical_json_bytes, sha256_bytes, sha256_json

from .gpu_ownership import (
    GpuOwnershipReceipt,
)
from .gpu_doctor import (
    GpuDoctorInspector,
    GpuDoctorReceipt,
    LinuxGpuDoctorInspector,
)
from .gpu_topology import selector_scope
from .gpu_lifecycle import (
    GpuLeaseHeartbeatReceipt,
    GpuLeaseOwnerIdentity,
    GpuMeasurementBracketReceipt,
    GpuMeasurementGuard,
)


@dataclass(frozen=True, slots=True)
class GpuLeaseReceipt:
    schema_version: int
    run_id: str
    execution_scope: str
    physical_scope: str
    owner_pid: int
    acquired_unix_seconds: float
    lock_path: str
    ownership: GpuOwnershipReceipt
    doctor: GpuDoctorReceipt
    lock_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        paths = self.lock_paths or (self.lock_path,)
        object.__setattr__(self, "lock_paths", paths)
        unique_ids = tuple(
            sorted(device.unique_id for device in self.ownership.selected_devices)
        )
        if (
            self.schema_version != 3
            or self.owner_pid <= 0
            or self.acquired_unix_seconds < 0
            or not Path(self.lock_path).is_absolute()
            or self.lock_path != paths[0]
            or len(paths) != len(unique_ids)
            or len(paths) != len(set(paths))
            or len(unique_ids) != len(set(unique_ids))
            or any(not Path(path).is_absolute() for path in paths)
            or self.execution_scope != self.ownership.execution_scope
            or self.physical_scope != self.ownership.physical_scope
            or self.ownership.foreign_owners
            or self.doctor.ownership != self.ownership
            or not self.doctor.formal_measurement_ready
        ):
            raise ContractError(
                "GPU lease receipt is inconsistent with ownership evidence",
                "invalid_gpu_lease_receipt",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "execution_scope": self.execution_scope,
            "physical_scope": self.physical_scope,
            "owner_pid": self.owner_pid,
            "acquired_unix_seconds": self.acquired_unix_seconds,
            "lock_path": self.lock_path,
            "lock_paths": list(self.lock_paths),
            "ownership": self.ownership.to_dict(),
            "doctor": self.doctor.to_dict(),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class GpuLease(Protocol):
    receipt: GpuLeaseReceipt

    def __enter__(self) -> "GpuLease": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    def heartbeat(self, reason: str = "manual") -> GpuLeaseHeartbeatReceipt: ...

    def measurement(self, action_id: str) -> GpuMeasurementGuard: ...


class GpuLeaseManager(Protocol):
    def acquire(
        self, run_id: str, *, requested_devices: str | None = None
    ) -> GpuLease: ...


class LocalGpuLease:
    """A nonblocking advisory lock shared by cooperating Apex processes."""

    def __init__(
        self,
        run_id: str,
        *,
        lock_root: Path | None = None,
        requested_devices: str | None = None,
        doctor_inspector: GpuDoctorInspector | None = None,
        ttl_seconds: float = 10_800.0,
        clock: Callable[[], float] = time.time,
        owner_identity_provider: Callable[[], GpuLeaseOwnerIdentity] | None = None,
    ) -> None:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("GPU lease TTL must be positive")
        self._run_id = run_id
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._owner_identity_provider = owner_identity_provider or _current_owner
        self._preflight_owner = self._owner_identity_provider()
        self._selector_scope = resolve_gpu_device_scope(requested_devices)
        self._doctor_inspector = doctor_inspector or LinuxGpuDoctorInspector()
        preflight_doctor = self._doctor_inspector.inspect(
            self._selector_scope, allowed_pids=(os.getpid(),)
        )
        preflight = preflight_doctor.ownership
        _reject_foreign_owners(preflight)
        _reject_doctor(preflight_doctor)
        self._execution_scope = preflight.execution_scope
        self._physical_scope = preflight.physical_scope
        self._preflight = preflight
        self._preflight_doctor = preflight_doctor
        root = lock_root or Path("/tmp/apex-gpu-leases")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._locks = _physical_lock_paths(root, preflight)
        self._descriptors: tuple[int, ...] = ()
        self._heartbeat_receipt: GpuLeaseHeartbeatReceipt | None = None
        self._heartbeat_sequence = 0
        self.receipt = GpuLeaseReceipt(
            3,
            run_id,
            self._execution_scope,
            self._physical_scope,
            os.getpid(),
            0.0,
            str(self._locks[0][1]),
            preflight,
            preflight_doctor,
            tuple(str(path) for _, path in self._locks),
        )

    def __enter__(self) -> "LocalGpuLease":
        acquired_descriptors: list[int] = []
        try:
            for unique_id, path in self._locks:
                acquired_descriptors.append(
                    _acquire_physical_lock(unique_id, path, self._physical_scope)
                )
            doctor = self._doctor_inspector.inspect(
                self._selector_scope, allowed_pids=(os.getpid(),)
            )
            _reject_doctor(doctor)
            ownership = doctor.ownership
            if ownership.selected_devices != self._preflight.selected_devices:
                raise ContractError(
                    "Physical GPU mapping changed while acquiring the lease",
                    "gpu_physical_mapping_changed",
                )
            if (
                ownership.device_inventory != self._preflight.device_inventory
                or ownership.selector_inputs != self._preflight.selector_inputs
                or ownership.hsa_inventory != self._preflight.hsa_inventory
            ):
                raise ContractError(
                    "GPU inventory or visibility changed while acquiring the lease",
                    "gpu_physical_mapping_changed",
                )
            _reject_foreign_owners(ownership)
            owner = self._owner_identity_provider()
            if owner != self._preflight_owner:
                raise ContractError(
                    "GPU lease owner process changed during acquisition",
                    "gpu_lease_owner_changed",
                )
            acquired_at = self._clock()
            self.receipt = GpuLeaseReceipt(
                3,
                self._run_id,
                self._execution_scope,
                self._physical_scope,
                os.getpid(),
                acquired_at,
                str(self._locks[0][1]),
                ownership,
                doctor,
                tuple(str(path) for _, path in self._locks),
            )
            self._descriptors = tuple(acquired_descriptors)
            self._heartbeat_receipt = self._new_heartbeat(
                "acquired", doctor, owner, acquired_at
            )
            self._write_metadata()
            return self
        except BaseException:
            _release_descriptors(acquired_descriptors)
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        descriptors = self._descriptors
        if not descriptors:
            return
        self._descriptors = ()
        failures = _release_descriptors(descriptors)
        if failures and exc_type is None:
            raise ContractError(
                "One or more physical GPU locks could not be released cleanly",
                "gpu_lease_release_failed",
                {"failures": failures},
            )

    def heartbeat(self, reason: str = "manual") -> GpuLeaseHeartbeatReceipt:
        """Renew only after proving this process still owns the same healthy GPUs."""

        if not self._descriptors or self._heartbeat_receipt is None:
            raise ContractError("GPU lease is not active", "gpu_lease_inactive")
        now = self._clock()
        previous = self._heartbeat_receipt
        if now < previous.observed_unix_seconds:
            raise ContractError("GPU lease clock regressed", "gpu_lease_clock_regressed")
        if now > previous.valid_until_unix_seconds:
            raise ContractError(
                "GPU lease expired before its next heartbeat",
                "gpu_lease_expired",
                {"expired_unix_seconds": previous.valid_until_unix_seconds},
            )
        owner = self._owner_identity_provider()
        if owner != self._preflight_owner:
            raise ContractError(
                "GPU lease owner process identity changed",
                "gpu_lease_owner_changed",
            )
        doctor = self._doctor_inspector.inspect(
            self._selector_scope, allowed_pids=(owner.pid,)
        )
        self._validate_renewal(doctor)
        receipt = self._new_heartbeat(reason, doctor, owner, now)
        self._heartbeat_receipt = receipt
        self._write_metadata()
        return receipt

    def measurement(self, action_id: str) -> GpuMeasurementGuard:
        """Create a fail-closed pre/post bracket for one formal measurement."""

        if not action_id:
            raise ValueError("measurement action_id is required")
        return _LocalMeasurementGuard(self, action_id)

    def _validate_renewal(self, doctor: GpuDoctorReceipt) -> None:
        _reject_doctor(doctor)
        ownership = doctor.ownership
        _reject_foreign_owners(ownership)
        if (
            ownership.selected_devices != self._preflight.selected_devices
            or ownership.device_inventory != self._preflight.device_inventory
            or ownership.selector_inputs != self._preflight.selector_inputs
            or ownership.hsa_inventory != self._preflight.hsa_inventory
        ):
            raise ContractError(
                "GPU identity changed during an active lease",
                "gpu_lease_device_identity_changed",
            )

    def _new_heartbeat(
        self,
        reason: str,
        doctor: GpuDoctorReceipt,
        owner: GpuLeaseOwnerIdentity,
        observed: float,
    ) -> GpuLeaseHeartbeatReceipt:
        self._heartbeat_sequence += 1
        return GpuLeaseHeartbeatReceipt(
            1,
            self._run_id,
            self.receipt.digest,
            self._heartbeat_sequence,
            reason,
            observed,
            observed + self._ttl_seconds,
            self._ttl_seconds,
            owner,
            doctor.ownership,
            doctor,
        )

    def _write_metadata(self) -> None:
        heartbeat = self._heartbeat_receipt
        assert heartbeat is not None
        document = {**self.receipt.to_dict(), "heartbeat": heartbeat.to_dict()}
        payload = canonical_json_bytes(document) + b"\n"
        for descriptor in self._descriptors:
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, payload)
            os.fsync(descriptor)


class _LocalMeasurementGuard:
    def __init__(self, lease: LocalGpuLease, action_id: str) -> None:
        self._lease = lease
        self._action_id = action_id
        self._pre: GpuLeaseHeartbeatReceipt | None = None
        self._started = 0.0
        self.receipt: GpuMeasurementBracketReceipt

    def __enter__(self) -> "_LocalMeasurementGuard":
        self._pre = self._lease.heartbeat("measurement_pre")
        self._started = self._lease._clock()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        finished = self._lease._clock()
        post = self._lease.heartbeat("measurement_post")
        assert self._pre is not None
        self.receipt = GpuMeasurementBracketReceipt(
            1,
            self._lease.receipt.run_id,
            self._action_id,
            self._lease.receipt.digest,
            self._started,
            finished,
            self._pre,
            post,
        )


class LocalGpuLeaseManager:
    def __init__(
        self,
        *,
        lock_root: Path | None = None,
        doctor_inspector: GpuDoctorInspector | None = None,
        ttl_seconds: float = 10_800.0,
        clock: Callable[[], float] = time.time,
        owner_identity_provider: Callable[[], GpuLeaseOwnerIdentity] | None = None,
    ) -> None:
        self._lock_root = lock_root
        self._doctor_inspector = doctor_inspector
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._owner_identity_provider = owner_identity_provider

    def acquire(
        self, run_id: str, *, requested_devices: str | None = None
    ) -> LocalGpuLease:
        return LocalGpuLease(
            run_id,
            lock_root=self._lock_root,
            requested_devices=requested_devices,
            doctor_inspector=self._doctor_inspector,
            ttl_seconds=self._ttl_seconds,
            clock=self._clock,
            owner_identity_provider=self._owner_identity_provider,
        )


def resolve_gpu_device_scope(requested_devices: str | None = None) -> str:
    """Validate a requested selector; the inspector resolves ambient visibility."""

    return selector_scope(requested_devices)


def _physical_lock_paths(
    root: Path, ownership: GpuOwnershipReceipt
) -> tuple[tuple[str, Path], ...]:
    unique_ids = tuple(
        sorted(device.unique_id for device in ownership.selected_devices)
    )
    if not unique_ids or len(unique_ids) != len(set(unique_ids)):
        raise ContractError(
            "Physical GPU lock identities are empty or duplicated",
            "gpu_physical_mapping_unresolved",
        )
    return tuple(
        (
            unique_id,
            (
                root
                / f"gpu-{sha256_json({'physical_unique_id': unique_id})[:24]}.lock"
            ).resolve(),
        )
        for unique_id in unique_ids
    )


def _acquire_physical_lock(unique_id: str, path: Path, physical_scope: str) -> int:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        owner = _read_owner(descriptor)
        os.close(descriptor)
        raise ContractError(
            "Another Apex run holds a selected physical GPU lease",
            "gpu_lease_busy",
            {
                "physical_scope": physical_scope,
                "physical_unique_id": unique_id,
                "lock_path": str(path),
                "owner": owner,
            },
        ) from error
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _release_descriptors(
    descriptors: list[int] | tuple[int, ...],
) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for descriptor in reversed(descriptors):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as error:
            failures.append(
                {"descriptor": descriptor, "operation": "unlock", "errno": error.errno}
            )
        try:
            os.close(descriptor)
        except OSError as error:
            failures.append(
                {"descriptor": descriptor, "operation": "close", "errno": error.errno}
            )
    return failures


def _read_owner(descriptor: int) -> object:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        content = os.read(descriptor, 16 * 1024)
        return json.loads(content.decode("utf-8")) if content else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _reject_foreign_owners(receipt: GpuOwnershipReceipt) -> None:
    if not receipt.foreign_owners:
        return
    raise ContractError(
        "A selected physical GPU has a foreign KFD process owner",
        "gpu_foreign_owner",
        {
            "ownership_receipt": receipt.to_dict(),
            "ownership_receipt_sha256": receipt.digest,
        },
    )


def _reject_doctor(receipt: GpuDoctorReceipt) -> None:
    if receipt.formal_measurement_ready:
        return
    raise ContractError(
        "GPU doctor preflight did not establish formal measurement readiness",
        "gpu_doctor_not_ready",
        {
            "status": receipt.status,
            "doctor_receipt": receipt.to_dict(),
            "doctor_receipt_sha256": receipt.digest,
        },
    )


def _current_owner() -> GpuLeaseOwnerIdentity:
    pid = os.getpid()
    root = Path("/proc") / str(pid)
    try:
        metadata = root.stat()
        raw_stat = (root / "stat").read_text(encoding="utf-8")
        cmdline = (root / "cmdline").read_bytes()
        tail = raw_stat[raw_stat.rindex(")") + 2 :].split()
        start_time_ticks = int(tail[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ContractError(
            "GPU lease owner process identity is unavailable",
            "gpu_lease_owner_identity_unavailable",
            {"pid": pid},
        ) from error
    return GpuLeaseOwnerIdentity(
        pid, metadata.st_uid, start_time_ticks, sha256_bytes(cmdline)
    )


__all__ = [
    "GpuLease",
    "GpuLeaseManager",
    "GpuLeaseReceipt",
    "LocalGpuLease",
    "LocalGpuLeaseManager",
    "resolve_gpu_device_scope",
]
