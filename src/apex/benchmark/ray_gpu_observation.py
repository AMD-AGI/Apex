"""Bind KFD/RSMI GPU processes to one frozen local Ray task worker."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from apex.core import ContractError, sha256_bytes
from apex.runtime import GpuOwnershipInspector, RocmSmiGpuOwnershipInspector

from .docker_gpu_observation import _lease_devices
from .ray_observation import RayTaskObservation, RayWorkerProcessObservation


_MAX_ANCESTORS = 4096
_MAX_PROC_BYTES = 64 * 1024


class RayGpuObserver(Protocol):
    def observe(
        self,
        task: RayTaskObservation,
        worker: RayWorkerProcessObservation,
        gpu_lease: Mapping[str, object],
    ) -> Mapping[str, object] | None: ...

    def cleanup_verified(
        self,
        task: RayTaskObservation,
        worker: RayWorkerProcessObservation,
        gpu_lease: Mapping[str, object],
    ) -> bool: ...


class RocmRayGpuObserver:
    """Require every leased-GPU process to descend from one frozen Ray worker."""

    def __init__(
        self,
        inspector: GpuOwnershipInspector | None = None,
        *,
        proc_root: Path = Path("/proc"),
    ) -> None:
        self._inspector = inspector or RocmSmiGpuOwnershipInspector()
        self._proc_root = proc_root

    def observe(
        self,
        task: RayTaskObservation,
        worker: RayWorkerProcessObservation,
        gpu_lease: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        selector, expected = _lease_devices(gpu_lease)
        receipt = self._inspector.inspect(selector)
        _require_exact_devices(receipt.selected_devices, expected)
        owners = (*receipt.allowed_owners, *receipt.foreign_owners)
        if not owners:
            return None
        processes = tuple(
            _bind_owner(owner, task, worker, proc_root=self._proc_root)
            for owner in owners
        )
        engaged = {
            index for owner in owners for index in owner.rsmi_device_indices
        }
        if engaged != set(expected):
            return None
        return {
            "devices": [
                {"rsmi_index": index, "unique_id": expected[index]}
                for index in sorted(expected)
            ],
            "processes": list(processes),
            "ownership_receipt_sha256": receipt.digest,
            "ray_task_id": task.task_id,
            "ray_job_id": task.job_id,
            "ray_worker_id": task.worker_id,
            "ray_node_id": task.node_id,
        }

    def cleanup_verified(
        self,
        task: RayTaskObservation,
        worker: RayWorkerProcessObservation,
        gpu_lease: Mapping[str, object],
    ) -> bool:
        del task, worker
        selector, expected = _lease_devices(gpu_lease)
        receipt = self._inspector.inspect(selector)
        _require_exact_devices(receipt.selected_devices, expected)
        return not (*receipt.allowed_owners, *receipt.foreign_owners)


def _require_exact_devices(devices: tuple[object, ...], expected: Mapping[int, str]) -> None:
    actual = {
        getattr(device, "rsmi_index", None): getattr(device, "unique_id", None)
        for device in devices
    }
    if actual != dict(expected):
        raise ContractError(
            "GPU inventory differs from the active lease", "magpie_gpu_lease_drift"
        )


def _bind_owner(owner, task, worker, *, proc_root: Path) -> Mapping[str, object]:
    pid = owner.pid
    first = _process_stat(proc_root, pid)
    cgroup = _process_cgroup(proc_root, pid)
    second = _process_stat(proc_root, pid)
    if (
        first != second
        or first[0] != owner.start_time_ticks
        or sha256_bytes(cgroup) != worker.cgroup_sha256
        or not _descends_from(
            pid,
            worker.pid,
            worker.start_time_ticks,
            proc_root=proc_root,
        )
    ):
        raise ContractError(
            "A leased GPU process is outside the frozen Ray worker",
            "magpie_ray_gpu_process_escape",
        )
    return {
        "pid": pid,
        "uid": owner.uid,
        "start_time_ticks": owner.start_time_ticks,
        "cmdline_sha256": owner.cmdline_sha256,
        "cgroup_sha256": sha256_bytes(cgroup),
        "rsmi_device_indices": list(owner.rsmi_device_indices),
        "ray_task_id": task.task_id,
        "ray_worker_id": task.worker_id,
        "ray_worker_pid": worker.pid,
    }


def _descends_from(
    pid: int,
    ancestor_pid: int,
    ancestor_start_time: int,
    *,
    proc_root: Path,
) -> bool:
    current = pid
    visited: set[int] = set()
    for _ in range(_MAX_ANCESTORS):
        if current in visited or current <= 0:
            return False
        if current == 1 and ancestor_pid != 1:
            return False
        visited.add(current)
        start, parent = _process_stat(proc_root, current)
        if current == ancestor_pid:
            return start == ancestor_start_time
        if parent == current:
            return False
        current = parent
    raise ContractError("Process ancestry is too deep", "ray_process_ancestry_unbounded")


def _process_stat(proc_root: Path, pid: int) -> tuple[int, int]:
    try:
        raw = _bounded_read(proc_root / str(pid) / "stat").decode("utf-8")
        fields = raw[raw.rindex(")") + 2 :].split()
        return int(fields[19]), int(fields[1])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ContractError(
            "Ray GPU process identity is unavailable",
            "magpie_ray_gpu_process_unavailable",
        ) from error


def _process_cgroup(proc_root: Path, pid: int) -> bytes:
    try:
        first = _bounded_read(proc_root / str(pid) / "cgroup")
        second = _bounded_read(proc_root / str(pid) / "cgroup")
    except (OSError, ValueError) as error:
        raise ContractError(
            "Ray GPU process cgroup is unavailable",
            "magpie_ray_gpu_process_unavailable",
        ) from error
    if first != second:
        raise ContractError("Ray GPU process cgroup raced", "magpie_ray_gpu_process_race")
    return first


def _bounded_read(path: Path) -> bytes:
    value = path.read_bytes()
    if not value or len(value) > _MAX_PROC_BYTES:
        raise ValueError("procfs field size is invalid")
    return value


__all__ = ["RayGpuObserver", "RocmRayGpuObserver"]
