"""Private-procfs identity and retained namespace-membership evidence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from apex.core import ContractError


_STAT_LIMIT = 4096


@dataclass(frozen=True, slots=True)
class ProcfsMountIdentity:
    """The visible mount ID and superblock device for one procfs path."""

    mount_id: int
    device_major: int
    device_minor: int


def procfs_mount_identity(path: Path) -> ProcfsMountIdentity:
    """Read the identity of the mount actually visible at ``path``."""

    descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        return _procfs_identity_for_fd(descriptor)
    finally:
        os.close(descriptor)


def private_procfs_verified(
    pid: int, supervisor_procfs: ProcfsMountIdentity
) -> bool:
    """Verify that the target's visible procfs is private and fully mounted."""

    try:
        visible = procfs_mount_identity(Path(f"/proc/{pid}/root/proc"))
        return _private_procfs_identity_verified(pid, visible, supervisor_procfs)
    except (OSError, ValueError):
        return False


def open_verified_private_procfs(
    pid: int, supervisor_procfs: ProcfsMountIdentity
) -> int:
    """Retain the verified procfs so it remains enumerable after teardown."""

    descriptor = os.open(
        Path(f"/proc/{pid}/root/proc"),
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        visible = _procfs_identity_for_fd(descriptor)
        if not _private_procfs_identity_verified(pid, visible, supervisor_procfs):
            raise ContractError(
                "Agent private procfs changed before retention",
                "agent_process_containment_failed",
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def live_namespace_members(procfs_fd: int) -> list[int] | None:
    """List live members through retained procfs, or fail the scan closed."""

    try:
        entries = os.listdir(procfs_fd)
    except OSError:
        return None
    members: list[int] = []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            state = _process_state(procfs_fd, entry)
            if state != "Z":
                members.append(int(entry))
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError, IndexError, ValueError):
            return None
    return sorted(members)


def _private_procfs_identity_verified(
    pid: int,
    visible: ProcfsMountIdentity,
    supervisor_procfs: ProcfsMountIdentity,
) -> bool:
    try:
        lines = Path(f"/proc/{pid}/mountinfo").read_text(encoding="utf-8").splitlines()
        pid_one_private = _private_proc_pid_one_verified(pid)
    except (OSError, ValueError):
        return False
    if _same_mount_or_superblock(visible, supervisor_procfs):
        return False
    for line in lines:
        if _is_visible_private_procfs(line, visible):
            return pid_one_private
    return False


def _same_mount_or_superblock(
    target: ProcfsMountIdentity, host: ProcfsMountIdentity
) -> bool:
    return target.mount_id == host.mount_id or (
        target.device_major,
        target.device_minor,
    ) == (host.device_major, host.device_minor)


def _is_visible_private_procfs(line: str, visible: ProcfsMountIdentity) -> bool:
    fields = line.split()
    if len(fields) < 10 or "-" not in fields:
        return False
    separator = fields.index("-")
    try:
        mount_id = int(fields[0])
        device = tuple(map(int, fields[2].split(":")))
    except (ValueError, IndexError):
        return False
    return (
        mount_id == visible.mount_id
        and fields[4] == "/proc"
        and not fields[6:separator]
        and fields[3] == "/"
        and device == (visible.device_major, visible.device_minor)
        and fields[separator + 1 : separator + 3] == ["proc", "proc"]
    )


def _private_proc_pid_one_verified(pid: int) -> bool:
    lines = Path(f"/proc/{pid}/root/proc/1/status").read_text(
        encoding="utf-8"
    ).splitlines()
    nspid = [
        line.partition(":")[2].split()
        for line in lines
        if line.startswith("NSpid:")
    ]
    return len(nspid) == 1 and [int(value) for value in nspid[0]] == [1]


def _procfs_identity_for_fd(descriptor: int) -> ProcfsMountIdentity:
    device = os.fstat(descriptor).st_dev
    lines = Path(f"/proc/self/fdinfo/{descriptor}").read_text(
        encoding="utf-8"
    ).splitlines()
    values = [
        line.partition(":")[2].strip()
        for line in lines
        if line.startswith("mnt_id:")
    ]
    if len(values) != 1 or int(values[0]) <= 0:
        raise ValueError("procfs mount identity is invalid")
    return ProcfsMountIdentity(
        mount_id=int(values[0]),
        device_major=os.major(device),
        device_minor=os.minor(device),
    )


def _process_state(procfs_fd: int, pid: str) -> str:
    descriptor = os.open(
        f"{pid}/stat",
        os.O_RDONLY | os.O_CLOEXEC,
        dir_fd=procfs_fd,
    )
    try:
        payload = os.read(descriptor, _STAT_LIMIT + 1)
    finally:
        os.close(descriptor)
    return payload.decode("utf-8").rsplit(")", 1)[1].split()[0]


__all__ = [
    "ProcfsMountIdentity",
    "live_namespace_members",
    "open_verified_private_procfs",
    "private_procfs_verified",
    "procfs_mount_identity",
]
