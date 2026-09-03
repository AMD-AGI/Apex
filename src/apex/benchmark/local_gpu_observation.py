"""Bind local KFD/RSMI owners to an active Apex lease and process tree."""

from __future__ import annotations

import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from apex.core import ContractError, sha256_json
from apex.runtime import GpuOwnershipInspector, RocmSmiGpuOwnershipInspector

from .local_process_observation import (
    LocalProcessIdentity,
    LocalProcessObservationClient,
    descendant_closure,
)


_DIGEST = re.compile(r"[0-9a-f]{64}")
_UNIQUE_ID = re.compile(r"GPU-[0-9a-f]{16}")
_MAX_LEASE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LocalGpuLeaseAuthority:
    """Proof that this exact Apex process actively holds all lease lock files."""

    run_id: str
    lease_digest: str
    selector_scope: str
    devices: tuple[tuple[int, str], ...]
    owner: LocalProcessIdentity
    heartbeat_sha256: str
    valid_until_unix_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "lease_digest": self.lease_digest,
            "selector_scope": self.selector_scope,
            "devices": [
                {"rsmi_index": index, "unique_id": unique_id}
                for index, unique_id in self.devices
            ],
            "owner": self.owner.to_dict(),
            "heartbeat_sha256": self.heartbeat_sha256,
            "valid_until_unix_seconds": self.valid_until_unix_seconds,
        }


class LocalGpuEngagementObserver(Protocol):
    def observe(
        self,
        authorized_roots: tuple[LocalProcessIdentity, ...],
        authority: LocalGpuLeaseAuthority,
    ) -> Mapping[str, object] | None: ...

    def require_quiescent(
        self, authority: LocalGpuLeaseAuthority
    ) -> Mapping[str, object]: ...


class RocmLocalGpuEngagementObserver:
    """Use the race-checked ownership inspector; never match process names."""

    def __init__(
        self,
        processes: LocalProcessObservationClient,
        inspector: GpuOwnershipInspector | None = None,
    ) -> None:
        self._processes = processes
        self._inspector = inspector or RocmSmiGpuOwnershipInspector()

    def observe(
        self,
        authorized_roots: tuple[LocalProcessIdentity, ...],
        authority: LocalGpuLeaseAuthority,
    ) -> Mapping[str, object] | None:
        if not authorized_roots:
            return None
        receipt = self._inspect(authority)
        owners = (*receipt.allowed_owners, *receipt.foreign_owners)
        if not owners:
            return None
        processes = self._processes.snapshot()
        allowed = {
            (item.pid, item.start_time_ticks): item
            for item in descendant_closure(processes, authorized_roots)
        }
        observed = []
        for owner in owners:
            process = allowed.get((owner.pid, owner.start_time_ticks))
            if process is None or not _owner_matches(owner, process, authority.owner):
                raise ContractError(
                    "A leased GPU owner escapes the observed Magpie process tree",
                    "magpie_local_gpu_process_escape",
                )
            observed.append(
                {
                    "pid": owner.pid,
                    "uid": owner.uid,
                    "start_time_ticks": owner.start_time_ticks,
                    "cmdline_sha256": owner.cmdline_sha256,
                    "rsmi_device_indices": list(owner.rsmi_device_indices),
                    "root_pid": _bound_root(process, processes, authorized_roots),
                    "cgroup_sha256": process.cgroup_sha256,
                }
            )
        engaged = {index for owner in owners for index in owner.rsmi_device_indices}
        expected = {index for index, _ in authority.devices}
        if engaged != expected:
            return None
        return {
            "devices": _device_receipts(authority),
            "processes": observed,
            "ownership_receipt_sha256": receipt.digest,
        }

    def require_quiescent(
        self, authority: LocalGpuLeaseAuthority
    ) -> Mapping[str, object]:
        receipt = self._inspect(authority)
        owners = (*receipt.allowed_owners, *receipt.foreign_owners)
        if owners:
            raise ContractError(
                "A local Magpie GPU process remains after cleanup",
                "magpie_local_gpu_residual_process",
            )
        return {
            "devices": _device_receipts(authority),
            "ownership_receipt_sha256": receipt.digest,
            "verified": True,
        }

    def _inspect(self, authority: LocalGpuLeaseAuthority):
        receipt = self._inspector.inspect(authority.selector_scope)
        observed = tuple(
            (item.rsmi_index, item.unique_id) for item in receipt.selected_devices
        )
        if observed != authority.devices:
            raise ContractError(
                "GPU inventory differs from the active local lease",
                "magpie_gpu_lease_drift",
            )
        return receipt


def validate_active_local_gpu_lease(
    value: Mapping[str, object],
    *,
    run_id: str,
    processes: LocalProcessObservationClient,
    locks_path: Path = Path("/proc/locks"),
    clock: Callable[[], float] = time.time,
) -> LocalGpuLeaseAuthority:
    """Validate receipt, live heartbeat, FLOCK ownership, and owner identity."""

    owner = processes.process(os.getpid())
    lock_paths = value.get("lock_paths")
    if (
        value.get("schema_version") != 3
        or value.get("run_id") != run_id
        or value.get("owner_pid") != os.getpid()
        or owner is None
        or not isinstance(lock_paths, list)
        or not lock_paths
        or any(not isinstance(item, str) for item in lock_paths)
    ):
        raise ContractError(
            "Local GPU lease owner is unavailable",
            "magpie_gpu_lease_mismatch",
        )
    selector, devices = _lease_devices(value)
    if len(lock_paths) != len(devices):
        raise ContractError(
            "Local GPU lease lock set is incomplete", "magpie_gpu_lease_mismatch"
        )
    lease_digest = sha256_json(value)
    heartbeat = _locked_heartbeat(
        tuple(Path(item) for item in lock_paths), value, locks_path
    )
    valid_until = heartbeat.get("valid_until_unix_seconds")
    heartbeat_owner = heartbeat.get("owner")
    if (
        heartbeat.get("schema_version") != 1
        or heartbeat.get("run_id") != run_id
        or heartbeat.get("lease_digest") != lease_digest
        or not isinstance(valid_until, (int, float))
        or isinstance(valid_until, bool)
        or float(valid_until) < clock()
        or heartbeat_owner != _owner_receipt(owner)
    ):
        raise ContractError(
            "Local GPU lease heartbeat is stale or mismatched",
            "magpie_gpu_lease_inactive",
        )
    return LocalGpuLeaseAuthority(
        run_id,
        lease_digest,
        selector,
        devices,
        owner,
        sha256_json(heartbeat),
        float(valid_until),
    )


def _lease_devices(
    value: Mapping[str, object],
) -> tuple[str, tuple[tuple[int, str], ...]]:
    ownership = value.get("ownership")
    if not isinstance(ownership, Mapping) or ownership.get("foreign_owners") != []:
        raise ContractError("GPU lease ownership is invalid", "magpie_gpu_lease_mismatch")
    selector = ownership.get("selector_scope")
    selected = ownership.get("selected_devices")
    if not isinstance(selector, str) or not selector or not isinstance(selected, list):
        raise ContractError("GPU lease selection is invalid", "magpie_gpu_lease_mismatch")
    devices = []
    for item in selected:
        if not isinstance(item, Mapping):
            raise ContractError("GPU lease device is invalid", "magpie_gpu_lease_mismatch")
        index, unique_id = item.get("rsmi_index"), item.get("unique_id")
        if (
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            or not isinstance(unique_id, str) or not _UNIQUE_ID.fullmatch(unique_id)
        ):
            raise ContractError("GPU lease device is invalid", "magpie_gpu_lease_mismatch")
        devices.append((index, unique_id))
    result = tuple(devices)
    if not result or len(result) != len(set(result)):
        raise ContractError("GPU lease selection is invalid", "magpie_gpu_lease_mismatch")
    return selector, result


def _locked_heartbeat(
    paths: tuple[Path, ...],
    receipt: Mapping[str, object],
    locks_path: Path,
) -> Mapping[str, object]:
    held = _held_flocks(locks_path)
    heartbeat: Mapping[str, object] | None = None
    for path in paths:
        if not path.is_absolute():
            raise ContractError("GPU lease path is unsafe", "magpie_gpu_lease_mismatch")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.is_symlink():
            raise ContractError("GPU lease path is unsafe", "magpie_gpu_lease_mismatch")
        if (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino) not in held:
            raise ContractError("GPU lease lock is not held", "magpie_gpu_lease_inactive")
        document = _load_lock(path)
        if any(document.get(key) != item for key, item in receipt.items()):
            raise ContractError("GPU lease metadata drifted", "magpie_gpu_lease_mismatch")
        current = document.get("heartbeat")
        if not isinstance(current, Mapping) or (heartbeat is not None and current != heartbeat):
            raise ContractError("GPU lease heartbeat is invalid", "magpie_gpu_lease_inactive")
        heartbeat = current
    assert heartbeat is not None
    return heartbeat


def _held_flocks(path: Path) -> set[tuple[int, int, int]]:
    raw = _read_bounded(path, _MAX_LEASE_BYTES).decode("utf-8")
    held: set[tuple[int, int, int]] = set()
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        if fields[4] != str(os.getpid()):
            continue
        try:
            major, minor, inode = fields[5].split(":", 2)
            held.add((int(major, 16), int(minor, 16), int(inode)))
        except ValueError:
            continue
    return held


def _load_lock(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(_read_bounded(path, _MAX_LEASE_BYTES))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("GPU lease metadata is invalid", "magpie_gpu_lease_mismatch") from error
    if not isinstance(value, Mapping):
        raise ContractError("GPU lease metadata is invalid", "magpie_gpu_lease_mismatch")
    return value


def _read_bounded(path: Path, limit: int) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        content = os.read(descriptor, limit + 1)
    finally:
        os.close(descriptor)
    if len(content) > limit:
        raise ContractError("GPU lease evidence is oversized", "magpie_gpu_lease_mismatch")
    return content


def _owner_matches(owner, process, lease_owner) -> bool:
    return bool(
        owner.uid == process.uid
        and owner.cmdline_sha256 == process.cmdline_sha256
        and process.cgroup_sha256 == lease_owner.cgroup_sha256
        and process.cgroup_lines == lease_owner.cgroup_lines
    )


def _owner_receipt(value: LocalProcessIdentity) -> Mapping[str, object]:
    return {
        "pid": value.pid,
        "uid": value.uid,
        "start_time_ticks": value.start_time_ticks,
        "cmdline_sha256": value.cmdline_sha256,
    }


def _device_receipts(authority: LocalGpuLeaseAuthority) -> list[dict[str, object]]:
    return [
        {"rsmi_index": index, "unique_id": unique_id}
        for index, unique_id in authority.devices
    ]


def _bound_root(
    process: LocalProcessIdentity,
    processes: tuple[LocalProcessIdentity, ...],
    roots: tuple[LocalProcessIdentity, ...],
) -> int:
    by_pid = {item.pid: item for item in processes}
    root_ids = {(item.pid, item.start_time_ticks): item.pid for item in roots}
    current = process
    visited: set[int] = set()
    while current.pid not in visited:
        root = root_ids.get((current.pid, current.start_time_ticks))
        if root is not None:
            return root
        visited.add(current.pid)
        parent = by_pid.get(current.ppid)
        if parent is None:
            break
        current = parent
    raise ContractError("GPU process root is unresolved", "magpie_local_gpu_process_escape")


__all__ = [
    "LocalGpuEngagementObserver",
    "LocalGpuLeaseAuthority",
    "RocmLocalGpuEngagementObserver",
    "validate_active_local_gpu_lease",
]
