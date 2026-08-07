"""Run-scoped cooperative GPU lease without process-killing side effects."""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, canonical_json_bytes, sha256_json


@dataclass(frozen=True, slots=True)
class GpuLeaseReceipt:
    schema_version: int
    run_id: str
    device_scope: str
    owner_pid: int
    acquired_unix_seconds: float
    lock_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

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
    ) -> None:
        self._run_id = run_id
        self._device_scope = resolve_gpu_device_scope(requested_devices)
        root = lock_root or Path("/tmp/apex-gpu-leases")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = sha256_json({"device_scope": self._device_scope})[:24]
        self._path = (root / f"gpu-{key}.lock").resolve()
        self._descriptor: int | None = None
        self.receipt = GpuLeaseReceipt(
            1,
            run_id,
            self._device_scope,
            os.getpid(),
            0.0,
            str(self._path),
        )

    def __enter__(self) -> "LocalGpuLease":
        descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            owner = _read_owner(descriptor)
            os.close(descriptor)
            raise ContractError(
                "Another Apex run holds the requested GPU lease",
                "gpu_lease_busy",
                {"device_scope": self._device_scope, "owner": owner},
            ) from error
        acquired = time.time()
        self.receipt = GpuLeaseReceipt(
            1,
            self._run_id,
            self._device_scope,
            os.getpid(),
            acquired,
            str(self._path),
        )
        os.ftruncate(descriptor, 0)
        os.write(descriptor, canonical_json_bytes(self.receipt.to_dict()) + b"\n")
        os.fsync(descriptor)
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            self._descriptor = None


class LocalGpuLeaseManager:
    def __init__(self, *, lock_root: Path | None = None) -> None:
        self._lock_root = lock_root

    def acquire(
        self, run_id: str, *, requested_devices: str | None = None
    ) -> LocalGpuLease:
        return LocalGpuLease(
            run_id,
            lock_root=self._lock_root,
            requested_devices=requested_devices,
        )


def resolve_gpu_device_scope(requested_devices: str | None = None) -> str:
    """Resolve one physical AMD GPU set and reject split-brain visibility.

    E2E specs may select devices explicitly while standalone tasks inherit the
    evaluator-owned process visibility.  The lease identity deliberately omits
    the environment-variable name so ROCR and HIP users contend on the same
    physical set.
    """

    requested = (
        _parse_device_set(requested_devices, source="requested GPU devices")
        if requested_devices is not None
        else None
    )
    ambient: list[tuple[str, tuple[str, ...]]] = []
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            ambient.append((name, _parse_device_set(raw, source=name)))

    if len({devices for _, devices in ambient}) > 1:
        raise ContractError(
            "ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES select different GPU sets",
            "gpu_visibility_mismatch",
            {name: ",".join(devices) for name, devices in ambient},
        )
    observed = ambient[0][1] if ambient else None
    if requested is not None and observed is not None and requested != observed:
        raise ContractError(
            "Requested GPU devices disagree with ambient process visibility",
            "gpu_visibility_mismatch",
            {
                "requested": ",".join(requested),
                "ambient": ",".join(observed),
            },
        )
    selected = requested or observed
    return (
        "amd-gpu-set=" + ",".join(selected)
        if selected is not None
        else "all-visible-amd-gpus"
    )


def _parse_device_set(raw: str, *, source: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in raw.split(","))
    if (
        not parts
        or any(not part or len(part) > 128 or "\x00" in part for part in parts)
        or len(parts) != len(set(parts))
    ):
        raise ContractError(
            f"{source} must be a non-empty comma-separated set without duplicates",
            "invalid_gpu_device_scope",
        )
    return tuple(sorted(parts))


def _read_owner(descriptor: int) -> object:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        content = os.read(descriptor, 16 * 1024)
        return json.loads(content.decode("utf-8")) if content else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


__all__ = [
    "GpuLease",
    "GpuLeaseManager",
    "GpuLeaseReceipt",
    "LocalGpuLease",
    "LocalGpuLeaseManager",
    "resolve_gpu_device_scope",
]
