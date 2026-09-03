"""Race-checked procfs observations for unchanged local Magpie execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError, sha256_bytes


_MAX_PROC_FIELD_BYTES = 64 * 1024
_MAX_PROCESSES = 65_536


@dataclass(frozen=True, slots=True)
class LocalProcessIdentity:
    """PID-reuse-resistant identity and containment facts for one process."""

    pid: int
    uid: int
    ppid: int
    process_group: int
    session_id: int
    start_time_ticks: int
    cmdline_sha256: str
    argv: tuple[str, ...]
    cwd: Path | None
    cgroup_sha256: str
    cgroup_lines: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "uid": self.uid,
            "ppid": self.ppid,
            "process_group": self.process_group,
            "session_id": self.session_id,
            "start_time_ticks": self.start_time_ticks,
            "cmdline_sha256": self.cmdline_sha256,
            "argv": list(self.argv),
            "cwd": str(self.cwd) if self.cwd is not None else None,
            "cgroup_sha256": self.cgroup_sha256,
            "cgroup_lines": list(self.cgroup_lines),
        }


class LocalProcessObservationClient(Protocol):
    """Read process identities without starting or signaling a process."""

    def snapshot(self) -> tuple[LocalProcessIdentity, ...]: ...

    def process(self, pid: int) -> LocalProcessIdentity | None: ...


class ProcfsLocalProcessObservationClient:
    """Double-read procfs identities with bounded inputs and no shell tools."""

    def __init__(self, *, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root

    def snapshot(self) -> tuple[LocalProcessIdentity, ...]:
        try:
            pids = sorted(
                int(item.name)
                for item in self._proc_root.iterdir()
                if item.name.isdigit()
            )
        except OSError as error:
            raise ContractError(
                "Local process table is unavailable",
                "magpie_local_process_table_unavailable",
            ) from error
        if len(pids) > _MAX_PROCESSES:
            raise ContractError(
                "Local process table exceeds the observation bound",
                "magpie_local_process_table_unavailable",
            )
        observed = []
        for pid in pids:
            try:
                identity = self.process(pid)
            except ContractError:
                # Unrelated hidepid/kernel/zombie entries are not authority.
                # A required Magpie or GPU PID is checked again by exact PID.
                continue
            if identity is not None:
                observed.append(identity)
        return tuple(observed)

    def process(self, pid: int) -> LocalProcessIdentity | None:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return None
        root = self._proc_root / str(pid)
        try:
            first_stat = _read_bounded(root / "stat")
            fields = _stat_fields(first_stat)
            cmdline = _read_bounded(root / "cmdline")
            if not cmdline:
                return None
            cgroup = _read_bounded(root / "cgroup")
            cwd = _read_cwd(root / "cwd")
            uid = root.stat().st_uid
            second_stat = _read_bounded(root / "stat")
        except (FileNotFoundError, ProcessLookupError):
            return None
        except (OSError, UnicodeError, ValueError, IndexError) as error:
            raise ContractError(
                "A local process identity could not be frozen",
                "magpie_local_process_identity_unavailable",
                {"pid": pid},
            ) from error
        if first_stat != second_stat:
            raise ContractError(
                "A local process identity raced during observation",
                "magpie_local_process_identity_race",
                {"pid": pid},
            )
        try:
            argv = _argv(cmdline)
            lines = _cgroup_lines(cgroup)
        except (UnicodeError, ValueError) as error:
            raise ContractError(
                "A local process identity could not be frozen",
                "magpie_local_process_identity_unavailable",
                {"pid": pid},
            ) from error
        return LocalProcessIdentity(
            pid=pid,
            uid=uid,
            ppid=int(fields[1]),
            process_group=int(fields[2]),
            session_id=int(fields[3]),
            start_time_ticks=int(fields[19]),
            cmdline_sha256=sha256_bytes(cmdline),
            argv=argv,
            cwd=cwd,
            cgroup_sha256=sha256_bytes(cgroup),
            cgroup_lines=lines,
        )


def matching_processes(
    processes: tuple[LocalProcessIdentity, ...],
    *,
    argv: tuple[str, ...],
    cwd: Path,
) -> tuple[LocalProcessIdentity, ...]:
    """Select only exact argv/cwd matches; substring process matching is forbidden."""

    expected_cwd = cwd.resolve()
    return tuple(
        item for item in processes
        if item.argv == argv and item.cwd is not None and item.cwd == expected_cwd
    )


def descendant_closure(
    processes: tuple[LocalProcessIdentity, ...],
    roots: tuple[LocalProcessIdentity, ...],
) -> tuple[LocalProcessIdentity, ...]:
    """Return exact roots and descendants in one frozen process-table snapshot."""

    by_pid = {item.pid: item for item in processes}
    root_ids = {(item.pid, item.start_time_ticks) for item in roots}
    selected = []
    for item in processes:
        current: LocalProcessIdentity | None = item
        visited: set[int] = set()
        while current is not None and current.pid not in visited:
            if (current.pid, current.start_time_ticks) in root_ids:
                selected.append(item)
                break
            visited.add(current.pid)
            current = by_pid.get(current.ppid)
    return tuple(selected)


def same_process(
    observed: LocalProcessIdentity,
    current: LocalProcessIdentity | None,
) -> bool:
    """Compare every immutable identity field to reject PID reuse and drift."""

    return current == observed


def belongs_to_root(
    value: LocalProcessIdentity,
    observed: dict[tuple[int, int], LocalProcessIdentity],
    root: LocalProcessIdentity,
) -> bool:
    """Whether a captured identity descends from one exact captured root."""

    by_pid = {item.pid: item for item in observed.values()}
    current = value
    visited: set[int] = set()
    while current.pid not in visited:
        if current == root:
            return True
        visited.add(current.pid)
        parent = by_pid.get(current.ppid)
        if parent is None:
            return False
        current = parent
    return False


def _read_bounded(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        content = os.read(descriptor, _MAX_PROC_FIELD_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(content) > _MAX_PROC_FIELD_BYTES:
        raise ValueError(f"procfs field exceeds {_MAX_PROC_FIELD_BYTES} bytes")
    return content


def _read_cwd(path: Path) -> Path | None:
    """Return cwd when procfs exposes it; Docker containment does not require it."""

    try:
        return Path(os.readlink(path)).resolve(strict=True)
    except PermissionError:
        return None


def _stat_fields(raw: bytes) -> tuple[str, ...]:
    value = raw.decode("utf-8")
    marker = value.rfind(")")
    if marker < 1:
        raise ValueError("invalid procfs stat")
    fields = tuple(value[marker + 2 :].split())
    if len(fields) <= 19:
        raise ValueError("incomplete procfs stat")
    return fields


def _argv(raw: bytes) -> tuple[str, ...]:
    parts = raw[:-1].split(b"\0") if raw.endswith(b"\0") else raw.split(b"\0")
    if not parts or any(not part for part in parts):
        raise ValueError("empty procfs command line")
    return tuple(part.decode("utf-8", errors="surrogateescape") for part in parts)


def _cgroup_lines(raw: bytes) -> tuple[str, ...]:
    lines = tuple(sorted(line for line in raw.decode("utf-8").splitlines() if line))
    if not lines:
        raise ValueError("empty procfs cgroup")
    return lines


__all__ = [
    "LocalProcessIdentity",
    "LocalProcessObservationClient",
    "ProcfsLocalProcessObservationClient",
    "belongs_to_root",
    "descendant_closure",
    "matching_processes",
    "same_process",
]
