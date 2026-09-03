"""Bounded Ray task and local worker observations for Magpie attestation."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import ContractError, sha256_bytes, sha256_file, sha256_json
from apex.execution import ProcessResult, SubprocessSupervisor, build_subprocess_environment


_RAY_ID = re.compile(r"[0-9a-f]{8,64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RUN_TASK_NAMES = frozenset({"run_task", "Magpie.remote.tasks.run_task"})
_RUNNING_STATES = frozenset(
    {
        "SUBMITTED_TO_WORKER",
        "RUNNING",
        "RUNNING_IN_RAY_GET",
        "RUNNING_IN_RAY_WAIT",
    }
)
_TERMINAL_STATES = frozenset({"FINISHED", "FAILED"})
_MAX_PROC_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class RayJobObservation:
    """One Ray driver job tied to its host PID."""

    job_id: str
    driver_pid: int
    is_dead: bool

    def __post_init__(self) -> None:
        if (
            not _RAY_ID.fullmatch(self.job_id)
            or self.driver_pid <= 0
            or not isinstance(self.is_dead, bool)
        ):
            raise ContractError("Ray job identity is invalid", "ray_job_observation_invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "driver_pid": self.driver_pid,
            "is_dead": self.is_dead,
        }


@dataclass(frozen=True, slots=True)
class RayTaskObservation:
    """One Ray state record with a concrete worker assignment."""

    task_id: str
    attempt_number: int
    job_id: str
    worker_id: str
    worker_pid: int
    node_id: str
    name: str
    func_or_class_name: str
    state: str

    def __post_init__(self) -> None:
        identifiers = (self.task_id, self.job_id, self.worker_id, self.node_id)
        if (
            any(not _RAY_ID.fullmatch(value) for value in identifiers)
            or self.attempt_number < 0
            or self.worker_pid <= 0
            or self.name not in _RUN_TASK_NAMES
            or self.func_or_class_name not in _RUN_TASK_NAMES
            or self.state not in _RUNNING_STATES | _TERMINAL_STATES
        ):
            raise ContractError("Ray task identity is invalid", "ray_task_observation_invalid")

    @property
    def key(self) -> tuple[str, int]:
        return self.task_id, self.attempt_number

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.task_id,
            self.attempt_number,
            self.job_id,
            self.worker_id,
            self.worker_pid,
            self.node_id,
            self.name,
            self.func_or_class_name,
        )

    @property
    def running(self) -> bool:
        return self.state in _RUNNING_STATES

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "attempt_number": self.attempt_number,
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "worker_pid": self.worker_pid,
            "node_id": self.node_id,
            "name": self.name,
            "func_or_class_name": self.func_or_class_name,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class RayTaskSnapshot:
    """One bounded Ray CLI query tied to exact client bytes and address."""

    address_sha256: str
    executable_path: str
    executable_sha256: str
    observed_unix_ns: int
    tasks: tuple[RayTaskObservation, ...]

    def __post_init__(self) -> None:
        executable = Path(self.executable_path)
        if (
            not _DIGEST.fullmatch(self.address_sha256)
            or not _DIGEST.fullmatch(self.executable_sha256)
            or not executable.is_absolute()
            or self.observed_unix_ns <= 0
            or len({task.key for task in self.tasks}) != len(self.tasks)
        ):
            raise ContractError(
                "Ray task snapshot identity is invalid", "ray_task_observation_invalid"
            )

    @property
    def identity_digest(self) -> str:
        return sha256_json(
            {
                "address_sha256": self.address_sha256,
                "executable_path": self.executable_path,
                "executable_sha256": self.executable_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class RayJobSnapshot:
    """One bounded Ray jobs query under the same cluster identity."""

    address_sha256: str
    executable_path: str
    executable_sha256: str
    observed_unix_ns: int
    jobs: tuple[RayJobObservation, ...]

    def __post_init__(self) -> None:
        executable = Path(self.executable_path)
        if (
            not _DIGEST.fullmatch(self.address_sha256)
            or not _DIGEST.fullmatch(self.executable_sha256)
            or not executable.is_absolute()
            or self.observed_unix_ns <= 0
            or len({job.job_id for job in self.jobs}) != len(self.jobs)
        ):
            raise ContractError(
                "Ray job snapshot identity is invalid", "ray_job_observation_invalid"
            )

    @property
    def identity_digest(self) -> str:
        return sha256_json(
            {
                "address_sha256": self.address_sha256,
                "executable_path": self.executable_path,
                "executable_sha256": self.executable_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class RayWorkerProcessObservation:
    """PID-reuse-resistant local procfs identity of a Ray worker."""

    pid: int
    uid: int
    start_time_ticks: int
    cmdline_sha256: str
    cgroup_sha256: str
    cgroup_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.pid <= 0
            or self.uid < 0
            or self.start_time_ticks <= 0
            or not _DIGEST.fullmatch(self.cmdline_sha256)
            or not _DIGEST.fullmatch(self.cgroup_sha256)
            or not self.cgroup_paths
            or any(not path.startswith("/") for path in self.cgroup_paths)
        ):
            raise ContractError(
                "Ray worker process identity is invalid", "ray_worker_process_unavailable"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "uid": self.uid,
            "start_time_ticks": self.start_time_ticks,
            "cmdline_sha256": self.cmdline_sha256,
            "cgroup_sha256": self.cgroup_sha256,
            "cgroup_paths": list(self.cgroup_paths),
        }


@dataclass(frozen=True, slots=True)
class RayDriverProcessObservation:
    """Exact local process identity of the Magpie command driving Ray."""

    process: RayWorkerProcessObservation
    argv_sha256: str

    def __post_init__(self) -> None:
        if not _DIGEST.fullmatch(self.argv_sha256):
            raise ContractError(
                "Ray driver argv identity is invalid", "ray_driver_process_unavailable"
            )

    def to_dict(self) -> dict[str, object]:
        return {**self.process.to_dict(), "argv_sha256": self.argv_sha256}


class RayObservationClient(Protocol):
    def jobs(self) -> RayJobSnapshot: ...

    def tasks(self) -> RayTaskSnapshot: ...


class RayObservationClientFactory(Protocol):
    @property
    def is_available(self) -> bool: ...

    def create(self, address: str) -> RayObservationClient: ...


class RayWorkerProcessObserver(Protocol):
    def freeze(self, task: RayTaskObservation) -> RayWorkerProcessObservation: ...


class RayDriverProcessObserver(Protocol):
    def freeze(
        self, job: RayJobObservation, benchmark_argv: tuple[str, ...]
    ) -> RayDriverProcessObservation: ...


class RayCliObservationClient:
    """Read Ray's state API through one fixed, bounded CLI invocation."""

    def __init__(
        self,
        address: str,
        *,
        executable: str | None = None,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        if not address or len(address) > 2048 or any(character.isspace() for character in address):
            raise ValueError("Ray address must be a bounded non-whitespace value")
        discovered = executable or shutil.which("ray")
        if discovered is None:
            raise ContractError("Ray CLI is unavailable", "ray_observer_unavailable")
        resolved = Path(discovered).resolve(strict=True)
        if not resolved.is_file():
            raise ContractError("Ray CLI is unavailable", "ray_observer_unavailable")
        self._address = address
        self._executable = resolved
        self._executable_sha256 = sha256_file(resolved)
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=4 << 20)
        self._environment = build_subprocess_environment({})

    def jobs(self) -> RayJobSnapshot:
        result = self._query("jobs")
        return RayJobSnapshot(
            sha256_bytes(self._address.encode("utf-8")),
            str(self._executable),
            self._executable_sha256,
            time.time_ns(),
            _parse_jobs(result.stdout),
        )

    def tasks(self) -> RayTaskSnapshot:
        result = self._query("tasks", "--filter", "name=run_task")
        tasks = _parse_tasks(result.stdout)
        return RayTaskSnapshot(
            sha256_bytes(self._address.encode("utf-8")),
            str(self._executable),
            self._executable_sha256,
            time.time_ns(),
            tasks,
        )

    def _query(self, resource: str, *filters: str) -> ProcessResult:
        if sha256_file(self._executable) != self._executable_sha256:
            raise ContractError("Ray CLI changed during observation", "ray_cli_identity_changed")
        argv = (
            str(self._executable),
            "list",
            resource,
            "--detail",
            "--format=json",
            *filters,
            "--limit",
            "10000",
            "--address",
            self._address,
        )
        result = self._supervisor.run(
            argv,
            cwd=Path("/"),
            environment=self._environment,
            timeout_seconds=15,
        )
        _require_success(result)
        return result


class RayCliObservationClientFactory:
    """Create an address-bound observer for each resolved benchmark config."""

    def __init__(
        self,
        *,
        executable: str | None = None,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self._executable = executable or shutil.which("ray")
        self._supervisor = supervisor

    @property
    def is_available(self) -> bool:
        if self._executable is None:
            return False
        path = Path(self._executable)
        return path.is_file() and not path.is_symlink()

    def create(self, address: str) -> RayObservationClient:
        if not self.is_available or self._executable is None:
            raise ContractError("Ray CLI is unavailable", "ray_observer_unavailable")
        return RayCliObservationClient(
            address,
            executable=self._executable,
            supervisor=self._supervisor,
        )


class LocalRayDriverProcessObserver:
    """Tie a Ray job driver to the exact supervised Magpie argv and Apex PID."""

    def __init__(self, *, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root
        self._owner_pid = os.getpid()
        self._owner_start = _process_fields(proc_root, self._owner_pid)[0]

    def freeze(
        self, job: RayJobObservation, benchmark_argv: tuple[str, ...]
    ) -> RayDriverProcessObservation:
        process, cmdline = _freeze_process(self._proc_root, job.driver_pid)
        argv = tuple(
            item.decode("utf-8", errors="strict")
            for item in cmdline.rstrip(b"\0").split(b"\0")
        )
        if argv != benchmark_argv:
            raise ContractError(
                "Ray driver argv differs from Magpie", "ray_driver_process_mismatch"
            )
        if not _descends_from(
            self._proc_root,
            job.driver_pid,
            self._owner_pid,
            self._owner_start,
        ):
            raise ContractError(
                "Ray driver is outside the Apex process tree",
                "ray_driver_process_mismatch",
            )
        return RayDriverProcessObservation(process, sha256_json(list(argv)))


class LocalRayWorkerProcessObserver:
    """Freeze a Ray state worker PID against the local procfs view."""

    def __init__(self, *, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root

    def freeze(self, task: RayTaskObservation) -> RayWorkerProcessObservation:
        process, cmdline = _freeze_process(self._proc_root, task.worker_pid)
        text = cmdline.replace(b"\0", b" ").decode("utf-8", errors="strict")
        if "ray" not in text.casefold() or task.node_id not in text:
            raise ContractError(
                "Ray state worker does not match local procfs", "ray_worker_process_mismatch"
            )
        return process


def _require_success(result: ProcessResult) -> None:
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
        or not result.cleanup_succeeded
    ):
        raise ContractError("Ray state query failed", "ray_observer_failed")


def _parse_tasks(raw: str) -> tuple[RayTaskObservation, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError("Ray task output is invalid", "ray_observer_failed") from error
    if not isinstance(value, list) or len(value) > 10_000:
        raise ContractError("Ray task output is invalid", "ray_observer_failed")
    tasks = tuple(_parse_task(item) for item in value)
    if len({task.key for task in tasks}) != len(tasks):
        raise ContractError("Ray task output is ambiguous", "ray_observer_failed")
    return tuple(sorted(tasks, key=lambda task: task.key))


def _parse_jobs(raw: str) -> tuple[RayJobObservation, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError("Ray job output is invalid", "ray_observer_failed") from error
    if not isinstance(value, list) or len(value) > 10_000:
        raise ContractError("Ray job output is invalid", "ray_observer_failed")
    jobs = tuple(_parse_job(item) for item in value)
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ContractError("Ray job output is ambiguous", "ray_observer_failed")
    return tuple(sorted(jobs, key=lambda job: job.job_id))


def _parse_job(value: object) -> RayJobObservation:
    if not isinstance(value, Mapping):
        raise ContractError("Ray job output is invalid", "ray_observer_failed")
    pid, dead = value.get("driver_pid"), value.get("is_dead")
    if isinstance(pid, bool) or not isinstance(pid, int) or not isinstance(dead, bool):
        raise ContractError("Ray job output is invalid", "ray_observer_failed")
    try:
        return RayJobObservation(str(value["job_id"]), pid, dead)
    except (KeyError, ContractError) as error:
        raise ContractError("Ray job output is invalid", "ray_observer_failed") from error


def _parse_task(value: object) -> RayTaskObservation:
    if not isinstance(value, Mapping):
        raise ContractError("Ray task output is invalid", "ray_observer_failed")
    attempt = value.get("attempt_number", 0)
    pid = value.get("worker_pid")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise ContractError("Ray task output is invalid", "ray_observer_failed")
    if isinstance(pid, bool) or not isinstance(pid, int):
        raise ContractError("Ray task output is invalid", "ray_observer_failed")
    try:
        return RayTaskObservation(
            task_id=str(value["task_id"]),
            attempt_number=attempt,
            job_id=str(value["job_id"]),
            worker_id=str(value["worker_id"]),
            worker_pid=pid,
            node_id=str(value["node_id"]),
            name=str(value["name"]),
            func_or_class_name=str(value["func_or_class_name"]),
            state=str(value["state"]),
        )
    except (KeyError, ContractError) as error:
        raise ContractError("Ray task output is invalid", "ray_observer_failed") from error


def _read(path: Path) -> bytes:
    value = path.read_bytes()
    if not value or len(value) > _MAX_PROC_BYTES:
        raise ValueError("procfs field size is invalid")
    return value


def _freeze_process(
    proc_root: Path, pid: int
) -> tuple[RayWorkerProcessObservation, bytes]:
    root = proc_root / str(pid)
    try:
        metadata = root.stat()
        first_stat = _read(root / "stat")
        first_cgroup = _read(root / "cgroup")
        cmdline = _read(root / "cmdline")
        second_cgroup = _read(root / "cgroup")
        second_stat = _read(root / "stat")
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ContractError(
            "Ray process is not locally observable", "ray_worker_process_unavailable"
        ) from error
    if first_stat != second_stat or first_cgroup != second_cgroup:
        raise ContractError("Ray process identity raced", "ray_worker_process_race")
    return (
        RayWorkerProcessObservation(
            pid,
            metadata.st_uid,
            _start_time(first_stat),
            sha256_bytes(cmdline),
            sha256_bytes(first_cgroup),
            _cgroup_paths(first_cgroup),
        ),
        cmdline,
    )


def _process_fields(proc_root: Path, pid: int) -> tuple[int, int]:
    raw = _read(proc_root / str(pid) / "stat").decode("utf-8")
    fields = raw[raw.rindex(")") + 2 :].split()
    return int(fields[19]), int(fields[1])


def _descends_from(
    proc_root: Path, pid: int, ancestor_pid: int, ancestor_start: int
) -> bool:
    current = pid
    seen: set[int] = set()
    for _ in range(4096):
        if current <= 0 or current in seen:
            return False
        seen.add(current)
        start, parent = _process_fields(proc_root, current)
        if current == ancestor_pid:
            return start == ancestor_start
        if current == 1 or current == parent:
            return False
        current = parent
    raise ContractError("Ray driver ancestry is too deep", "ray_driver_process_mismatch")


def _start_time(raw: bytes) -> int:
    text = raw.decode("utf-8")
    return int(text[text.rindex(")") + 2 :].split()[19])


def _cgroup_paths(raw: bytes) -> tuple[str, ...]:
    paths: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3 or not parts[2].startswith("/"):
            raise ValueError("malformed cgroup membership")
        paths.append(parts[2])
    if not paths:
        raise ValueError("empty cgroup membership")
    return tuple(paths)


__all__ = [
    "LocalRayDriverProcessObserver",
    "LocalRayWorkerProcessObserver",
    "RayCliObservationClient",
    "RayCliObservationClientFactory",
    "RayDriverProcessObservation",
    "RayDriverProcessObserver",
    "RayJobObservation",
    "RayJobSnapshot",
    "RayObservationClient",
    "RayObservationClientFactory",
    "RayTaskObservation",
    "RayTaskSnapshot",
    "RayWorkerProcessObservation",
    "RayWorkerProcessObserver",
]
