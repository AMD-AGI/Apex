"""Read-only GPU ownership, Linux process-context, and health-activity preflight."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import ContractError, sha256_bytes, sha256_json

from .gpu_ownership import (
    GpuOwnershipInspector,
    GpuOwnershipReceipt,
    GpuProcessIdentity,
    RocmSmiGpuOwnershipInspector,
)
from .gpu_health import (
    CtypesRocmHealthInspector,
    RocmHealthInspector,
    RocmHealthReceipt,
)


_CONTAINER_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_SLURM_JOB = re.compile(r"(?:^|/)job[_-]([0-9]+)(?:/|$)")
_SLURM_STEP = re.compile(r"(?:^|/)step[_-]([A-Za-z0-9_.-]+)(?:/|$)")
_HEALTH_NAMES = frozenset({"nhc", "rocminfo", "rocm-smi", "amd-smi"})
_MAX_PROC_FILE = 64 * 1024


@dataclass(frozen=True, slots=True)
class GpuProcessContext:
    pid: int
    uid: int
    start_time_ticks: int
    comm: str
    cgroup_sha256: str
    cgroup_paths: tuple[str, ...]
    pid_namespace_inode: int
    mount_namespace_inode: int
    user_namespace_inode: int
    container_id: str | None
    slurm_job_id: str | None
    slurm_step_id: str | None

    def __post_init__(self) -> None:
        if (
            self.pid <= 0
            or self.uid < 0
            or self.start_time_ticks <= 0
            or not self.comm
            or len(self.cgroup_sha256) != 64
            or min(
                self.pid_namespace_inode,
                self.mount_namespace_inode,
                self.user_namespace_inode,
            )
            <= 0
        ):
            raise ContractError(
                "GPU process context is incomplete",
                "gpu_process_context_unavailable",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "uid": self.uid,
            "start_time_ticks": self.start_time_ticks,
            "comm": self.comm,
            "cgroup_sha256": self.cgroup_sha256,
            "cgroup_paths": list(self.cgroup_paths),
            "pid_namespace_inode": self.pid_namespace_inode,
            "mount_namespace_inode": self.mount_namespace_inode,
            "user_namespace_inode": self.user_namespace_inode,
            "container_id": self.container_id,
            "slurm_job_id": self.slurm_job_id,
            "slurm_step_id": self.slurm_step_id,
        }


@dataclass(frozen=True, slots=True)
class GpuDoctorReceipt:
    ownership: GpuOwnershipReceipt
    process_contexts: tuple[GpuProcessContext, ...]
    supervisor_context: GpuProcessContext
    scheduler_environment: tuple[tuple[str, str], ...]
    scheduler_identity_consistent: bool
    health_check_processes: tuple[GpuProcessContext, ...]
    process_scan_complete: bool
    rocm_health: RocmHealthReceipt | None
    rocm_health_error: str | None
    schema: str = "apex.gpu-doctor-receipt/v1"

    def __post_init__(self) -> None:
        owners = (*self.ownership.allowed_owners, *self.ownership.foreign_owners)
        if (
            self.schema != "apex.gpu-doctor-receipt/v1"
            or not self.process_scan_complete
            or (self.rocm_health is None) == (self.rocm_health_error is None)
            or tuple(item.pid for item in self.process_contexts)
            != tuple(item.pid for item in owners)
        ):
            raise ContractError(
                "GPU doctor receipt is incomplete",
                "invalid_gpu_doctor_receipt",
            )
        for owner, context in zip(owners, self.process_contexts, strict=True):
            if (
                owner.pid != context.pid
                or owner.uid != context.uid
                or owner.start_time_ticks != context.start_time_ticks
            ):
                raise ContractError(
                    "GPU owner and Linux process context disagree",
                    "invalid_gpu_doctor_receipt",
                )

    @property
    def formal_measurement_ready(self) -> bool:
        return bool(
            self.rocm_health is not None
            and not self.ownership.foreign_owners
            and self.scheduler_identity_consistent
            and not self.health_check_processes
        )

    @property
    def status(self) -> str:
        if self.ownership.foreign_owners or not self.scheduler_identity_consistent:
            return "blocked"
        if self.health_check_processes:
            return "blocked"
        return "ready" if self.formal_measurement_ready else "incomplete"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "status": self.status,
            "formal_measurement_ready": self.formal_measurement_ready,
            "ownership": self.ownership.to_dict(),
            "ownership_receipt_sha256": self.ownership.digest,
            "process_contexts": [item.to_dict() for item in self.process_contexts],
            "supervisor_context": self.supervisor_context.to_dict(),
            "scheduler_environment": dict(self.scheduler_environment),
            "scheduler_identity_consistent": self.scheduler_identity_consistent,
            "health_check_processes": [
                item.to_dict() for item in self.health_check_processes
            ],
            "process_scan_complete": self.process_scan_complete,
            "rocm_health_status": (
                "healthy" if self.rocm_health is not None else "unavailable"
            ),
            "rocm_health": (
                self.rocm_health.to_dict() if self.rocm_health is not None else None
            ),
            "rocm_health_error": self.rocm_health_error,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class GpuDoctorInspector(Protocol):
    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuDoctorReceipt: ...


class LinuxGpuDoctorInspector:
    """Add race-checked procfs context to the authoritative RSMI owner map."""

    def __init__(
        self,
        ownership: GpuOwnershipInspector | None = None,
        *,
        proc_root: Path = Path("/proc"),
        environment: Mapping[str, str] | None = None,
        health: RocmHealthInspector | None = None,
    ) -> None:
        self._ownership = ownership or RocmSmiGpuOwnershipInspector()
        self._proc_root = proc_root
        self._environment = environment
        self._health = health or CtypesRocmHealthInspector()

    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuDoctorReceipt:
        ownership = self._ownership.inspect(
            selector_scope, allowed_pids=allowed_pids
        )
        owners = (*ownership.allowed_owners, *ownership.foreign_owners)
        contexts = tuple(
            _process_context(self._proc_root, owner.pid, expected=owner)
            for owner in owners
        )
        supervisor = _process_context(self._proc_root, os.getpid())
        environment = self._environment if self._environment is not None else os.environ
        scheduler = _scheduler_environment(environment)
        health = _health_activity(self._proc_root)
        rocm_health, health_error = self._rocm_health(ownership)
        return GpuDoctorReceipt(
            ownership=ownership,
            process_contexts=contexts,
            supervisor_context=supervisor,
            scheduler_environment=tuple(sorted(scheduler.items())),
            scheduler_identity_consistent=_scheduler_consistent(supervisor, scheduler),
            health_check_processes=health,
            process_scan_complete=True,
            rocm_health=rocm_health,
            rocm_health_error=health_error,
        )

    def _rocm_health(
        self, ownership: GpuOwnershipReceipt
    ) -> tuple[RocmHealthReceipt | None, str | None]:
        try:
            return self._health.inspect(ownership), None
        except ContractError as error:
            return None, error.reason_code


def _process_context(
    proc_root: Path,
    pid: int,
    *,
    expected: GpuProcessIdentity | None = None,
) -> GpuProcessContext:
    root = proc_root / str(pid)
    try:
        metadata = root.stat()
        first = _start_time(root)
        comm = _bounded_read(root / "comm").decode("utf-8").strip()
        cgroup = _bounded_read(root / "cgroup")
        paths = _cgroup_paths(cgroup)
        namespaces = tuple((root / "ns" / name).stat().st_ino for name in ("pid", "mnt", "user"))
        second = _start_time(root)
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ContractError(
            "Linux context for a GPU process could not be frozen",
            "gpu_process_context_unavailable",
            {"pid": pid},
        ) from error
    if first != second or (
        expected is not None
        and (metadata.st_uid, first) != (expected.uid, expected.start_time_ticks)
    ):
        raise ContractError(
            "GPU process identity changed during context inspection",
            "gpu_process_context_race",
            {"pid": pid},
        )
    container = _single_match(_CONTAINER_ID, paths, "container")
    job = _single_match(_SLURM_JOB, paths, "Slurm job")
    step = _single_match(_SLURM_STEP, paths, "Slurm step")
    return GpuProcessContext(
        pid=pid,
        uid=metadata.st_uid,
        start_time_ticks=first,
        comm=comm,
        cgroup_sha256=sha256_bytes(cgroup),
        cgroup_paths=paths,
        pid_namespace_inode=namespaces[0],
        mount_namespace_inode=namespaces[1],
        user_namespace_inode=namespaces[2],
        container_id=container,
        slurm_job_id=job,
        slurm_step_id=step,
    )


def _health_activity(proc_root: Path) -> tuple[GpuProcessContext, ...]:
    result: list[GpuProcessContext] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise ContractError(
            "The process table cannot be scanned for GPU health activity",
            "gpu_health_activity_unavailable",
        ) from error
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            comm = _bounded_read(entry / "comm").decode("utf-8").strip()
        except (OSError, UnicodeError, ValueError):
            continue
        if comm.casefold() not in _HEALTH_NAMES:
            continue
        try:
            result.append(_process_context(proc_root, int(entry.name)))
        except ContractError as error:
            if error.reason_code != "gpu_process_context_unavailable":
                raise
    return tuple(sorted(result, key=lambda item: item.pid))


def _start_time(root: Path) -> int:
    raw = _bounded_read(root / "stat").decode("utf-8")
    return int(raw[raw.rindex(")") + 2 :].split()[19])


def _bounded_read(path: Path) -> bytes:
    content = path.read_bytes()
    if not content or len(content) > _MAX_PROC_FILE:
        raise ValueError("procfs field size is invalid")
    return content


def _cgroup_paths(content: bytes) -> tuple[str, ...]:
    lines = content.decode("utf-8").splitlines()
    paths: list[str] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3 or not parts[2].startswith("/"):
            raise ValueError("malformed cgroup membership")
        paths.append(parts[2])
    if not paths:
        raise ValueError("empty cgroup membership")
    return tuple(paths)


def _single_match(pattern: re.Pattern[str], values: tuple[str, ...], label: str) -> str | None:
    matches = {match.group(1) if match.lastindex else match.group(0) for value in values for match in pattern.finditer(value)}
    if len(matches) > 1:
        raise ContractError(
            f"GPU process has ambiguous {label} identity",
            "gpu_process_context_ambiguous",
        )
    return next(iter(matches), None)


def _scheduler_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID"):
        value = environment.get(name)
        if value is None:
            continue
        if not value or len(value) > 256 or any(character.isspace() for character in value):
            raise ContractError(
                "Slurm environment identity is invalid",
                "gpu_scheduler_identity_invalid",
            )
        result[name] = value
    return result


def _scheduler_consistent(
    context: GpuProcessContext, environment: Mapping[str, str]
) -> bool:
    expected_job = environment.get("SLURM_JOB_ID")
    expected_step = environment.get("SLURM_STEP_ID")
    return (expected_job is None or expected_job == context.slurm_job_id) and (
        expected_step is None or expected_step == context.slurm_step_id
    )


__all__ = [
    "GpuDoctorInspector",
    "GpuDoctorReceipt",
    "GpuProcessContext",
    "LinuxGpuDoctorInspector",
]
