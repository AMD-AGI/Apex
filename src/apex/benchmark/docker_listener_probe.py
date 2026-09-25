"""Docker-daemon-assisted listener ownership probe for root-owned containers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    SubprocessSupervisor,
    build_subprocess_environment,
)

from .local_process_observation import LocalProcessIdentity


PROBE_SCHEMA = "apex.docker-listener-probe/v1"
_CONTAINER = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PROBE = r'''
import hashlib, json, os, sys

port = int(sys.argv[1])
inodes = set()
for name in ("tcp", "tcp6"):
    try:
        with open("/proc/net/" + name, "r", encoding="ascii") as stream:
            lines = stream.read(4 * 1024 * 1024 + 1).splitlines()
    except FileNotFoundError:
        continue
    if sum(len(line) for line in lines) > 4 * 1024 * 1024:
        raise SystemExit(2)
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        try:
            if int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(int(fields[9]))
        except (IndexError, ValueError):
            pass
owners = []
for raw_pid in sorted((item for item in os.listdir("/proc") if item.isdigit()), key=int):
    pid = int(raw_pid)
    try:
        entries = os.listdir(f"/proc/{pid}/fd")
        if len(entries) > 16384:
            raise SystemExit(3)
        owns = any(
            os.readlink(f"/proc/{pid}/fd/{entry}") in {f"socket:[{inode}]" for inode in inodes}
            for entry in entries
        )
        if not owns:
            continue
        first = open(f"/proc/{pid}/stat", "rb").read(65537)
        cmdline = open(f"/proc/{pid}/cmdline", "rb").read(65537)
        second = open(f"/proc/{pid}/stat", "rb").read(65537)
        if first != second or len(first) > 65536 or not cmdline or len(cmdline) > 65536:
            raise SystemExit(4)
        marker = first.rfind(b")")
        fields = first[marker + 2:].split()
        owners.append({
            "start_time_ticks": int(fields[19]),
            "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        })
    except (FileNotFoundError, ProcessLookupError):
        continue
print(json.dumps({
    "schema": "apex.docker-listener-probe/v1",
    "port": port,
    "owners": owners,
}, sort_keys=True, separators=(",", ":")))
'''.strip()


class DockerExecListenerProbe:
    """Map in-container socket owners back to a frozen host process closure."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(
            max_output_bytes=1024 * 1024
        )
        self._environment = build_subprocess_environment(
            {}, inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS
        )

    def listener_owners(
        self,
        container_id: str,
        port: int,
        processes: tuple[LocalProcessIdentity, ...],
    ) -> tuple[LocalProcessIdentity, ...]:
        if (
            not _CONTAINER.fullmatch(container_id)
            or isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 < port < 65536
        ):
            raise _invalid("Docker listener probe request is invalid")
        result = self._supervisor.run(
            (
                "docker", "container", "exec", "--user", "0:0",
                container_id, "python3", "-c", _PROBE, str(port),
            ),
            cwd=Path("/"),
            environment=self._environment,
            timeout_seconds=30,
        )
        if (
            result.exit_code != 0
            or result.timed_out
            or result.stdout_truncated
            or result.stderr_truncated
            or not result.cleanup_succeeded
        ):
            raise _invalid("Docker listener probe command failed")
        value = _document(result.stdout)
        if value.get("schema") != PROBE_SCHEMA or value.get("port") != port:
            raise _invalid("Docker listener probe response differs")
        owners = value.get("owners")
        if not isinstance(owners, list) or not owners or len(owners) > 64:
            raise _invalid("Docker listener probe found no bounded owner")
        selected = tuple(_match_owner(item, processes) for item in owners)
        if len(set(selected)) != len(selected):
            raise _invalid("Docker listener probe owners are ambiguous")
        return selected


def _document(output: str) -> Mapping[str, Any]:
    try:
        value = json.loads(output)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _invalid("Docker listener probe output is invalid JSON") from error
    if not isinstance(value, Mapping) or set(value) != {"schema", "port", "owners"}:
        raise _invalid("Docker listener probe output is invalid")
    return value


def _match_owner(
    value: object, processes: tuple[LocalProcessIdentity, ...]
) -> LocalProcessIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "start_time_ticks", "cmdline_sha256"
    }:
        raise _invalid("Docker listener owner identity is invalid")
    start = value.get("start_time_ticks")
    digest = value.get("cmdline_sha256")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or start <= 0
        or not isinstance(digest, str)
        or _DIGEST.fullmatch(digest) is None
    ):
        raise _invalid("Docker listener owner identity is invalid")
    matches = tuple(
        item for item in processes
        if item.start_time_ticks == start and item.cmdline_sha256 == digest
    )
    if len(matches) != 1:
        raise _invalid("Docker listener owner is outside the frozen closure")
    return matches[0]


def _invalid(message: str) -> ContractError:
    return ContractError(message, "docker_listener_probe_invalid")


__all__ = ["DockerExecListenerProbe", "PROBE_SCHEMA"]
