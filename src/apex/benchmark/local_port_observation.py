"""Bound one local TCP listener to exact observed Magpie process identities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from apex.core import ContractError

from .local_process_observation import LocalProcessIdentity


_MAX_TCP_BYTES = 4 * 1024 * 1024
_MAX_FDS_PER_PROCESS = 16_384


class LocalPortObservationClient(Protocol):
    def listener_owners(
        self,
        port: int,
        processes: tuple[LocalProcessIdentity, ...],
    ) -> tuple[LocalProcessIdentity, ...]: ...


class ProcfsLocalPortObservationClient:
    """Join procfs LISTEN socket inodes to a bounded exact process set."""

    def __init__(self, *, proc_root: Path = Path("/proc")) -> None:
        self._proc_root = proc_root

    def listener_owners(
        self,
        port: int,
        processes: tuple[LocalProcessIdentity, ...],
    ) -> tuple[LocalProcessIdentity, ...]:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise ContractError("Local server port is invalid", "magpie_local_port_invalid")
        inodes = self._listening_inodes(port)
        if not inodes:
            return ()
        owners = []
        for process in processes:
            if self._owns_any(process.pid, inodes):
                owners.append(process)
        return tuple(owners)

    def _listening_inodes(self, port: int) -> set[int]:
        result: set[int] = set()
        for name in ("tcp", "tcp6"):
            path = self._proc_root / "net" / name
            try:
                content = _read_bounded(path, _MAX_TCP_BYTES).decode("ascii")
            except FileNotFoundError:
                continue
            except (OSError, UnicodeError) as error:
                raise ContractError(
                    "Local TCP table is unavailable",
                    "magpie_local_port_observation_unavailable",
                ) from error
            for line in content.splitlines()[1:]:
                fields = line.split()
                if len(fields) < 10 or fields[3] != "0A":
                    continue
                try:
                    observed_port = int(fields[1].rsplit(":", 1)[1], 16)
                    inode = int(fields[9])
                except (ValueError, IndexError):
                    continue
                if observed_port == port and inode > 0:
                    result.add(inode)
        return result

    def _owns_any(self, pid: int, inodes: set[int]) -> bool:
        root = self._proc_root / str(pid) / "fd"
        try:
            entries = tuple(root.iterdir())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return False
        except OSError as error:
            raise ContractError(
                "Local process sockets are unavailable",
                "magpie_local_port_observation_unavailable",
                {"pid": pid},
            ) from error
        if len(entries) > _MAX_FDS_PER_PROCESS:
            raise ContractError(
                "Local process descriptor set exceeds the observation bound",
                "magpie_local_port_observation_unavailable",
                {"pid": pid},
            )
        for entry in entries:
            try:
                target = os.readlink(entry)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            except OSError as error:
                raise ContractError(
                    "Local process socket identity is unavailable",
                    "magpie_local_port_observation_unavailable",
                    {"pid": pid},
                ) from error
            inode = _socket_inode(target)
            if inode in inodes:
                return True
        return False


def _read_bounded(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        content = os.read(descriptor, limit + 1)
    finally:
        os.close(descriptor)
    if len(content) > limit:
        raise ContractError(
            "Local TCP evidence exceeds the observation bound",
            "magpie_local_port_observation_unavailable",
        )
    return content


def _socket_inode(value: str) -> int | None:
    if not value.startswith("socket:[") or not value.endswith("]"):
        return None
    try:
        return int(value[8:-1])
    except ValueError:
        return None


__all__ = ["LocalPortObservationClient", "ProcfsLocalPortObservationClient"]
