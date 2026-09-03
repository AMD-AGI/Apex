"""Authoritative Linux PID-namespace containment for untrusted agent CLIs."""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from apex.core import ContractError, DependencyError, sha256_file
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentProcessContainmentReceipt,
)

from .procfs import (
    ProcfsMountIdentity,
    live_namespace_members,
    open_verified_private_procfs,
    private_procfs_verified,
    procfs_mount_identity,
)


_STATUS_LIMIT = 4096
_READINESS_POLL_SECONDS = 0.005
_MASKED_DIRECTORIES = (
    "/proc/acpi",
    "/proc/asound",
    "/proc/scsi",
    "/sys/devices/virtual/powercap",
    "/sys/firmware",
)
_MASKED_FILES = (
    "/proc/interrupts",
    "/proc/kcore",
    "/proc/keys",
    "/proc/latency_stats",
    "/proc/sched_debug",
    "/proc/timer_list",
    "/proc/timer_stats",
)
_READONLY_PATHS = (
    "/proc/bus",
    "/proc/fs",
    "/proc/irq",
    "/proc/sys",
    "/proc/sysrq-trigger",
)


@dataclass(slots=True)
class PreparedPidNamespace:
    """A gated bubblewrap command and the descriptors owned by its parent."""

    argv: tuple[str, ...]
    launcher_path: str
    launcher_sha256: str
    status_read_fd: int
    status_write_fd: int
    gate_read_fd: int
    gate_write_fd: int

    @property
    def pass_fds(self) -> tuple[int, int]:
        return self.status_write_fd, self.gate_read_fd

    def release_child_fds(self) -> None:
        self.status_write_fd = _close_fd(self.status_write_fd)
        self.gate_read_fd = _close_fd(self.gate_read_fd)

    def close(self) -> None:
        self.status_read_fd = _close_fd(self.status_read_fd)
        self.status_write_fd = _close_fd(self.status_write_fd)
        self.gate_read_fd = _close_fd(self.gate_read_fd)
        self.gate_write_fd = _close_fd(self.gate_write_fd)


@dataclass(slots=True)
class ActivePidNamespace:
    """A pidfd-pinned namespace init held until teardown evidence is complete."""

    init_pidfd: int
    private_procfs_fd: int
    status_fd: int
    init_host_pid: int
    init_starttime: int
    pid_namespace_inode: int
    mount_namespace_inode: int
    ipc_namespace_inode: int
    user_namespace_inode: int
    private_procfs_verified: bool
    launcher_path: str
    launcher_sha256: str
    sigkill_sent: bool = False

    def terminate_now(self) -> None:
        """Kill the exact namespace init; the kernel then kills every member."""

        if self.init_pidfd < 0 or _pidfd_ready(self.init_pidfd, 0):
            return
        try:
            signal.pidfd_send_signal(self.init_pidfd, signal.SIGKILL)
        except ProcessLookupError:
            if _pidfd_ready(self.init_pidfd, 0):
                return
            raise
        self.sigkill_sent = True

    def close(self) -> None:
        self.init_pidfd = _close_fd(self.init_pidfd)
        self.private_procfs_fd = _close_fd(self.private_procfs_fd)
        self.status_fd = _close_fd(self.status_fd)


@dataclass(frozen=True, slots=True)
class _LaunchIdentity:
    starttime: int
    parent_pid: int
    inner_pid: int
    pid_namespace_inode: int
    mount_namespace_inode: int
    ipc_namespace_inode: int
    user_namespace_inode: int


def prepare_pid_namespace(command: Sequence[str]) -> PreparedPidNamespace:
    """Build a gated command whose descendants cannot escape its PID namespace."""

    launcher = _bubblewrap_identity()
    descriptors = [-1, -1, -1, -1]
    try:
        descriptors[0], descriptors[1] = os.pipe2(os.O_CLOEXEC)
        descriptors[2], descriptors[3] = os.pipe2(os.O_CLOEXEC)
        status_read, status_write, gate_read, gate_write = descriptors
        argv = [
            launcher[0],
            "--die-with-parent",
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-pid",
            "--bind",
            "/",
            "/",
            "--dev-bind",
            "/dev",
            "/dev",
            "--tmpfs",
            "/dev/shm",
            "--proc",
            "/proc",
        ]
        _append_system_path_masks(argv)
        argv.extend(
            [
                "--json-status-fd",
                str(status_write),
                "--block-fd",
                str(gate_read),
                "--",
                *command,
            ]
        )
        prepared = PreparedPidNamespace(
            argv=tuple(argv),
            launcher_path=launcher[0],
            launcher_sha256=launcher[1],
            status_read_fd=status_read,
            status_write_fd=status_write,
            gate_read_fd=gate_read,
            gate_write_fd=gate_write,
        )
        descriptors[:] = [-1, -1, -1, -1]
        return prepared
    finally:
        for descriptor in descriptors:
            _close_fd(descriptor)


def establish_pid_namespace(
    prepared: PreparedPidNamespace,
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float = 5.0,
) -> ActivePidNamespace:
    """Bind namespace PID 1 through bwrap status and a non-reusable pidfd."""

    boundary: ActivePidNamespace | None = None
    established = False
    pidfd = -1
    private_procfs_fd = -1
    deadline = time.monotonic() + timeout_seconds
    try:
        status = _read_launch_status(
            prepared.status_read_fd,
            max(0.0, deadline - time.monotonic()),
        )
        init_pid = status["child-pid"]
        pidfd = os.pidfd_open(init_pid, 0)
        supervisor_procfs = procfs_mount_identity(Path("/proc"))
        identity = _await_verified_launch_identity(
            init_pid,
            wrapper_pid=process.pid,
            pidfd=pidfd,
            status=status,
            supervisor_procfs=supervisor_procfs,
            deadline=deadline,
        )
        private_procfs_fd = open_verified_private_procfs(
            init_pid, supervisor_procfs
        )
        final_identity = _launch_identity(init_pid)
        _validate_launch_identity(final_identity, identity, process.pid, status)
        if _pidfd_ready(pidfd, 0):
            raise ContractError(
                "Agent PID namespace exited before release",
                "agent_process_containment_failed",
            )
        boundary = ActivePidNamespace(
            init_pidfd=pidfd,
            private_procfs_fd=private_procfs_fd,
            status_fd=prepared.status_read_fd,
            init_host_pid=init_pid,
            init_starttime=identity.starttime,
            pid_namespace_inode=identity.pid_namespace_inode,
            mount_namespace_inode=identity.mount_namespace_inode,
            ipc_namespace_inode=identity.ipc_namespace_inode,
            user_namespace_inode=identity.user_namespace_inode,
            private_procfs_verified=True,
            launcher_path=prepared.launcher_path,
            launcher_sha256=prepared.launcher_sha256,
        )
        pidfd = -1
        private_procfs_fd = -1
        prepared.status_read_fd = -1
        os.write(prepared.gate_write_fd, b"1")
        established = True
        return boundary
    except (OSError, KeyError, ValueError) as error:
        raise ContractError(
            "Agent PID namespace could not be established",
            "agent_process_containment_failed",
        ) from error
    finally:
        _close_fd(pidfd)
        _close_fd(private_procfs_fd)
        prepared.gate_write_fd = _close_fd(prepared.gate_write_fd)
        prepared.close()
        if boundary is not None and not established:
            boundary.close()


def finalize_pid_namespace(
    process: subprocess.Popen[str],
    boundary: ActivePidNamespace,
    *,
    termination_reason: str,
    terminate: bool,
    timeout_seconds: float,
) -> AgentProcessContainmentReceipt:
    """Destroy or observe the namespace, then prove that no member remains."""

    force_killed = False
    try:
        if terminate:
            boundary.terminate_now()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            force_killed = True
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                pass
        init_exited = _pidfd_ready(boundary.init_pidfd, timeout_seconds)
        payload, eof = _read_terminal_status(boundary.status_fd, timeout_seconds)
        terminal_verified = _terminal_status_matches(payload, process.poll())
        members = live_namespace_members(boundary.private_procfs_fd)
        return AgentProcessContainmentReceipt(
            policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
            launcher_path=boundary.launcher_path,
            launcher_sha256=boundary.launcher_sha256,
            namespace_init_host_pid=boundary.init_host_pid,
            namespace_init_starttime=boundary.init_starttime,
            namespace_init_inner_pid=1,
            pid_namespace_inode=boundary.pid_namespace_inode,
            mount_namespace_inode=boundary.mount_namespace_inode,
            ipc_namespace_inode=boundary.ipc_namespace_inode,
            user_namespace_inode=boundary.user_namespace_inode,
            private_procfs_verified=boundary.private_procfs_verified,
            pidfd_opened=boundary.init_pidfd >= 0,
            termination_reason=termination_reason,
            teardown_mode=("pidfd_sigkill" if boundary.sigkill_sent else "natural_exit"),
            pidfd_sigkill_sent=boundary.sigkill_sent,
            namespace_init_exit_verified=init_exited,
            wrapper_exit_verified=process.poll() is not None,
            wrapper_force_killed=force_killed,
            terminal_status_verified=terminal_verified,
            terminal_status_absent_after_sigkill=(not payload and boundary.sigkill_sent),
            status_eof_verified=eof,
            namespace_membership_scan_complete=members is not None,
            live_namespace_members_after=tuple(() if members is None else members),
        )
    finally:
        boundary.close()


def abort_prepared_namespace(
    prepared: PreparedPidNamespace, process: subprocess.Popen[str] | None
) -> None:
    """Close gates and stop a launch that failed before authority was established."""

    prepared.close()
    if process is None:
        return
    if process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe is not None:
            pipe.close()


def _bubblewrap_identity() -> tuple[str, str]:
    discovered = shutil.which("bwrap")
    if discovered is None:
        raise DependencyError(
            "Agent process containment requires bubblewrap",
            "agent_process_containment_unavailable",
        )
    try:
        resolved = Path(discovered).resolve(strict=True)
    except OSError as error:
        raise DependencyError(
            "Agent process containment launcher is unavailable",
            "agent_process_containment_unavailable",
        ) from error
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise DependencyError(
            "Agent process containment launcher is not executable",
            "agent_process_containment_unavailable",
        )
    return str(resolved), sha256_file(resolved)


def _append_system_path_masks(command: list[str]) -> None:
    """Restore Docker's masks after replacing its procfs inside the attempt."""

    for path in _MASKED_DIRECTORIES:
        if Path(path).is_dir():
            command.extend(["--tmpfs", path, "--remount-ro", path])
    for path in _MASKED_FILES:
        if Path(path).exists():
            command.extend(["--ro-bind", "/dev/null", path])
    for path in _READONLY_PATHS:
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])


def _read_launch_status(descriptor: int, timeout_seconds: float) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ContractError(
                "Agent PID namespace status timed out",
                "agent_process_containment_failed",
            )
        ready, _, _ = select.select([descriptor], [], [], remaining)
        if not ready:
            raise ContractError(
                "Agent PID namespace status timed out",
                "agent_process_containment_failed",
            )
        chunk = os.read(descriptor, _STATUS_LIMIT + 1)
        if not chunk:
            break
        chunks.append(chunk)
        payload = b"".join(chunks)
        if len(payload) > _STATUS_LIMIT or b"\n" in payload:
            break
    payload = b"".join(chunks)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(
            "Agent PID namespace status is malformed",
            "agent_process_containment_failed",
        ) from error
    expected = {"child-pid", "ipc-namespace", "mnt-namespace", "pid-namespace"}
    if (
        len(payload) > _STATUS_LIMIT
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
        or not isinstance(value, dict)
        or set(value) != expected
        or any(type(value.get(key)) is not int or value[key] <= 0 for key in expected)
    ):
        raise ContractError(
            "Agent PID namespace status is invalid",
            "agent_process_containment_failed",
        )
    return value


def _read_terminal_status(descriptor: int, timeout_seconds: float) -> tuple[bytes, bool]:
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    while sum(map(len, chunks)) <= _STATUS_LIMIT:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return b"".join(chunks), False
        ready, _, _ = select.select([descriptor], [], [], remaining)
        if not ready:
            return b"".join(chunks), False
        chunk = os.read(descriptor, _STATUS_LIMIT + 1)
        if not chunk:
            return b"".join(chunks), True
        chunks.append(chunk)
    return b"".join(chunks), False


def _terminal_status_matches(payload: bytes, exit_code: int | None) -> bool:
    if len(payload) > _STATUS_LIMIT or not payload.endswith(b"\n"):
        return False
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"exit-code"}
        and type(value.get("exit-code")) is int
        and value["exit-code"] == exit_code
    )


def _process_starttime(pid: int) -> int:
    stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = stat_line[stat_line.rfind(")") + 2 :].split()
    value = int(fields[19])
    if value <= 0:
        raise ValueError("invalid process starttime")
    return value


def _process_namespace_identity(pid: int) -> tuple[int, int]:
    lines = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
    values = {
        key: value.strip()
        for line in lines
        for key, separator, value in [line.partition(":")]
        if separator and key in {"PPid", "NSpid"}
    }
    return int(values["PPid"]), int(values["NSpid"].split()[-1])


def _namespace_inode(pid: int, kind: str) -> int:
    return (Path(f"/proc/{pid}/ns") / kind).stat().st_ino


def _launch_identity(pid: int) -> _LaunchIdentity:
    parent_pid, inner_pid = _process_namespace_identity(pid)
    return _LaunchIdentity(
        starttime=_process_starttime(pid),
        parent_pid=parent_pid,
        inner_pid=inner_pid,
        pid_namespace_inode=_namespace_inode(pid, "pid"),
        mount_namespace_inode=_namespace_inode(pid, "mnt"),
        ipc_namespace_inode=_namespace_inode(pid, "ipc"),
        user_namespace_inode=_namespace_inode(pid, "user"),
    )


def _await_verified_launch_identity(
    pid: int,
    *,
    wrapper_pid: int,
    pidfd: int,
    status: dict[str, int],
    supervisor_procfs: ProcfsMountIdentity,
    deadline: float,
) -> _LaunchIdentity:
    expected: _LaunchIdentity | None = None
    supervisor_user_namespace = _namespace_inode(os.getpid(), "user")
    while time.monotonic() < deadline:
        if _pidfd_ready(pidfd, 0):
            break
        try:
            before = _launch_identity(pid)
            expected = expected or before
            _validate_launch_identity(before, expected, wrapper_pid, status)
            private_procfs = private_procfs_verified(pid, supervisor_procfs)
            after = _launch_identity(pid)
            _validate_launch_identity(after, expected, wrapper_pid, status)
            if (
                private_procfs
                and before == after
                and after.user_namespace_inode != supervisor_user_namespace
                and not _pidfd_ready(pidfd, 0)
            ):
                return after
        except (OSError, ValueError):
            pass
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(_READINESS_POLL_SECONDS, remaining))
    raise ContractError(
        "Agent PID namespace did not reach verified private-proc readiness",
        "agent_process_containment_failed",
    )


def _validate_launch_identity(
    current: _LaunchIdentity,
    expected: _LaunchIdentity,
    wrapper_pid: int,
    status: dict[str, int],
) -> None:
    if (
        current != expected
        or current.parent_pid != wrapper_pid
        or current.inner_pid != 1
        or current.pid_namespace_inode != status["pid-namespace"]
        or current.mount_namespace_inode != status["mnt-namespace"]
        or current.ipc_namespace_inode != status["ipc-namespace"]
    ):
        raise ContractError(
            "Agent PID namespace identity changed before release",
            "agent_process_containment_failed",
        )


def _pidfd_ready(descriptor: int, timeout_seconds: float) -> bool:
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(max(0, int(timeout_seconds * 1000))))


def _close_fd(descriptor: int) -> int:
    if descriptor >= 0:
        try:
            os.close(descriptor)
        except OSError:
            pass
    return -1


__all__ = [
    "ActivePidNamespace",
    "PreparedPidNamespace",
    "abort_prepared_namespace",
    "establish_pid_namespace",
    "finalize_pid_namespace",
    "prepare_pid_namespace",
]
