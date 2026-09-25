from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.benchmark.docker_listener_probe import DockerExecListenerProbe
from apex.benchmark.local_process_observation import LocalProcessIdentity
from apex.core import ContractError
from apex.execution import ProcessResult


_CONTAINER = "a" * 64


class FakeSupervisor:
    def __init__(self, result: ProcessResult) -> None:
        self.result = result
        self.calls = []

    def run(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return self.result


def _identity(pid: int, start: int, digest: str) -> LocalProcessIdentity:
    return LocalProcessIdentity(
        pid, 0, 1, pid, pid, start, digest, ("python3", "server.py"),
        None, "b" * 64, (f"0::/docker/{_CONTAINER}",),
    )


def _result(value: object, *, exit_code: int = 0) -> ProcessResult:
    return ProcessResult(
        (), exit_code, False, json.dumps(value), "", False, False, 0.1
    )


def test_maps_in_container_socket_identity_to_frozen_host_process() -> None:
    owner = _identity(101, 500, "c" * 64)
    other = _identity(102, 501, "d" * 64)
    supervisor = FakeSupervisor(_result({
        "schema": "apex.docker-listener-probe/v1",
        "port": 8888,
        "owners": [{
            "start_time_ticks": owner.start_time_ticks,
            "cmdline_sha256": owner.cmdline_sha256,
        }],
    }))

    observed = DockerExecListenerProbe(supervisor).listener_owners(
        _CONTAINER, 8888, (owner, other)
    )

    assert observed == (owner,)
    argv, options = supervisor.calls[0]
    assert argv[:7] == (
        "docker", "container", "exec", "--user", "0:0", _CONTAINER,
        "python3",
    )
    assert options["cwd"] == Path("/")
    assert options["timeout_seconds"] == 30


def test_rejects_owner_outside_frozen_container_closure() -> None:
    supervisor = FakeSupervisor(_result({
        "schema": "apex.docker-listener-probe/v1",
        "port": 8888,
        "owners": [{
            "start_time_ticks": 999,
            "cmdline_sha256": "e" * 64,
        }],
    }))

    with pytest.raises(ContractError, match="outside the frozen closure"):
        DockerExecListenerProbe(supervisor).listener_owners(
            _CONTAINER, 8888, (_identity(101, 500, "c" * 64),)
        )


def test_rejects_failed_or_empty_probe() -> None:
    failed = FakeSupervisor(_result({}, exit_code=1))
    empty = FakeSupervisor(_result({
        "schema": "apex.docker-listener-probe/v1",
        "port": 8888,
        "owners": [],
    }))

    with pytest.raises(ContractError, match="command failed"):
        DockerExecListenerProbe(failed).listener_owners(
            _CONTAINER, 8888, (_identity(101, 500, "c" * 64),)
        )
    with pytest.raises(ContractError, match="no bounded owner"):
        DockerExecListenerProbe(empty).listener_owners(
            _CONTAINER, 8888, (_identity(101, 500, "c" * 64),)
        )
