"""Run-scoped physical GPU lease with fail-closed ownership evidence."""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, canonical_json_bytes, sha256_json

from .gpu_ownership import (
    GpuOwnershipInspector,
    GpuOwnershipReceipt,
    RocmSmiGpuOwnershipInspector,
)
from .gpu_topology import selector_scope


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
    lock_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        paths = self.lock_paths or (self.lock_path,)
        object.__setattr__(self, "lock_paths", paths)
        unique_ids = tuple(
            sorted(device.unique_id for device in self.ownership.selected_devices)
        )
        if (
            self.schema_version != 2
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
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class GpuLease(Protocol):
    receipt: GpuLeaseReceipt

    def __enter__(self) -> "GpuLease": ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


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
        ownership_inspector: GpuOwnershipInspector | None = None,
    ) -> None:
        self._run_id = run_id
        self._selector_scope = resolve_gpu_device_scope(requested_devices)
        self._ownership_inspector = (
            ownership_inspector or RocmSmiGpuOwnershipInspector()
        )
        preflight = self._ownership_inspector.inspect(
            self._selector_scope, allowed_pids=(os.getpid(),)
        )
        _reject_foreign_owners(preflight)
        self._execution_scope = preflight.execution_scope
        self._physical_scope = preflight.physical_scope
        self._preflight = preflight
        root = lock_root or Path("/tmp/apex-gpu-leases")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._locks = _physical_lock_paths(root, preflight)
        self._descriptors: tuple[int, ...] = ()
        self.receipt = GpuLeaseReceipt(
            2,
            run_id,
            self._execution_scope,
            self._physical_scope,
            os.getpid(),
            0.0,
            str(self._locks[0][1]),
            preflight,
            tuple(str(path) for _, path in self._locks),
        )

    def __enter__(self) -> "LocalGpuLease":
        acquired_descriptors: list[int] = []
        try:
            for unique_id, path in self._locks:
                acquired_descriptors.append(
                    _acquire_physical_lock(unique_id, path, self._physical_scope)
                )
            ownership = self._ownership_inspector.inspect(
                self._selector_scope, allowed_pids=(os.getpid(),)
            )
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
            acquired_at = time.time()
            self.receipt = GpuLeaseReceipt(
                2,
                self._run_id,
                self._execution_scope,
                self._physical_scope,
                os.getpid(),
                acquired_at,
                str(self._locks[0][1]),
                ownership,
                tuple(str(path) for _, path in self._locks),
            )
            payload = canonical_json_bytes(self.receipt.to_dict()) + b"\n"
            for descriptor in acquired_descriptors:
                os.ftruncate(descriptor, 0)
                os.lseek(descriptor, 0, os.SEEK_SET)
                os.write(descriptor, payload)
                os.fsync(descriptor)
            self._descriptors = tuple(acquired_descriptors)
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


class LocalGpuLeaseManager:
    def __init__(
        self,
        *,
        lock_root: Path | None = None,
        ownership_inspector: GpuOwnershipInspector | None = None,
    ) -> None:
        self._lock_root = lock_root
        self._ownership_inspector = ownership_inspector

    def acquire(
        self, run_id: str, *, requested_devices: str | None = None
    ) -> LocalGpuLease:
        return LocalGpuLease(
            run_id,
            lock_root=self._lock_root,
            requested_devices=requested_devices,
            ownership_inspector=self._ownership_inspector,
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


__all__ = [
    "GpuLease",
    "GpuLeaseManager",
    "GpuLeaseReceipt",
    "LocalGpuLease",
    "LocalGpuLeaseManager",
    "resolve_gpu_device_scope",
]
