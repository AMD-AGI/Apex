from __future__ import annotations

import os
from pathlib import Path

import pytest

from apex.benchmark.docker_observation import DockerContainerObservation
from apex.benchmark.docker_serving_listener import DockerServingListenerAuthority
from apex.benchmark.local_process_observation import LocalProcessIdentity
from apex.core import ContractError, sha256_json


_CONTAINER = "a" * 64
_IMAGE = "sha256:" + "b" * 64


class FakeProcesses:
    def __init__(self, identities: tuple[LocalProcessIdentity, ...]) -> None:
        self.identities = identities

    def snapshot(self) -> tuple[LocalProcessIdentity, ...]:
        return self.identities

    def process(self, pid: int) -> LocalProcessIdentity | None:
        return next((item for item in self.identities if item.pid == pid), None)


class FakePorts:
    def __init__(self, owner_pids: tuple[int, ...]) -> None:
        self.owner_pids = owner_pids
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def listener_owners(
        self,
        port: int,
        processes: tuple[LocalProcessIdentity, ...],
    ) -> tuple[LocalProcessIdentity, ...]:
        self.calls.append((port, tuple(item.pid for item in processes)))
        return tuple(item for item in processes if item.pid in self.owner_pids)


def _identity(
    pid: int,
    *,
    ppid: int,
    cgroup: tuple[str, ...] | None = None,
    start: int | None = None,
) -> LocalProcessIdentity:
    lines = cgroup or (f"0::/system.slice/docker-{_CONTAINER}.scope",)
    return LocalProcessIdentity(
        pid=pid,
        uid=os.getuid(),
        ppid=ppid,
        process_group=pid,
        session_id=pid,
        start_time_ticks=start or pid * 10,
        cmdline_sha256=f"{pid % 10}" * 64,
        argv=("process", str(pid)),
        cwd=Path("/").resolve(),
        cgroup_sha256="c" * 64,
        cgroup_lines=lines,
    )


def _container(*, running: bool = True, pid: int = 100) -> DockerContainerObservation:
    return DockerContainerObservation(
        container_id=_CONTAINER,
        name="magpie-benchmark-example",
        image_id=_IMAGE,
        configured_image="example/image@sha256:" + "d" * 64,
        pid=pid,
        running=running,
        workspace_mount=Path("/tmp/workspace"),
        inferencex_mount=Path("/tmp/inferencex"),
        container_spec_sha256="e" * 64,
        kfd_exposed=True,
        dri_exposed=True,
    )


def _observe(
    identities: tuple[LocalProcessIdentity, ...],
    owners: tuple[int, ...],
    *,
    container: DockerContainerObservation | None = None,
):
    ports = FakePorts(owners)
    receipt = DockerServingListenerAuthority(
        processes=FakeProcesses(identities),
        ports=ports,
    ).observe(container or _container(), 8888)
    return receipt, ports


def test_binds_listener_to_exact_container_process_closure() -> None:
    root = _identity(100, ppid=1)
    child = _identity(101, ppid=100)
    grandchild = _identity(102, ppid=101)

    receipt, ports = _observe((grandchild, root, child), (102, 101))

    assert receipt.container_id == _CONTAINER
    assert receipt.root_process == root
    assert receipt.listener_processes == (child, grandchild)
    assert receipt.closure_sha256 == sha256_json(
        [item.to_dict() for item in (root, child, grandchild)]
    )
    assert receipt.to_dict()["receipt_sha256"] == receipt.sha256
    assert ports.calls == [(8888, (102, 100, 101))]


@pytest.mark.parametrize(
    ("identities", "owners", "container"),
    [
        ((_identity(100, ppid=1),), (), _container()),
        ((_identity(101, ppid=1),), (101,), _container()),
        ((_identity(100, ppid=1),), (100,), _container(running=False)),
    ],
)
def test_rejects_missing_listener_root_or_running_container(
    identities: tuple[LocalProcessIdentity, ...],
    owners: tuple[int, ...],
    container: DockerContainerObservation,
) -> None:
    with pytest.raises(ContractError) as caught:
        _observe(identities, owners, container=container)

    assert caught.value.reason_code == "docker_serving_listener_invalid"


def test_rejects_foreign_listener_owner() -> None:
    root = _identity(100, ppid=1)
    child = _identity(101, ppid=100)
    foreign = _identity(200, ppid=1, cgroup=("0::/docker/" + "f" * 64,))

    with pytest.raises(ContractError) as caught:
        _observe((root, child, foreign), (101, 200))

    assert caught.value.reason_code == "docker_serving_listener_invalid"


def test_rejects_descendant_that_escaped_container_cgroup() -> None:
    root = _identity(100, ppid=1)
    escaped = _identity(101, ppid=100, cgroup=("0::/docker/" + "f" * 64,))

    with pytest.raises(ContractError) as caught:
        _observe((root, escaped), (100,))

    assert caught.value.reason_code == "docker_serving_listener_invalid"


def test_rejects_container_id_as_hex_substring() -> None:
    misleading = _identity(
        100,
        ppid=1,
        cgroup=(f"0::/docker/0{_CONTAINER}0",),
    )

    with pytest.raises(ContractError) as caught:
        _observe((misleading,), (100,))

    assert caught.value.reason_code == "docker_serving_listener_invalid"


def test_rejects_port_observer_identity_not_from_frozen_snapshot() -> None:
    root = _identity(100, ppid=1)
    drifted = _identity(100, ppid=1, start=root.start_time_ticks + 1)

    class DriftedPorts:
        def listener_owners(self, port, processes):
            assert port == 8888
            assert processes == (root,)
            return (drifted,)

    authority = DockerServingListenerAuthority(
        processes=FakeProcesses((root,)),
        ports=DriftedPorts(),
    )
    with pytest.raises(ContractError) as caught:
        authority.observe(_container(), 8888)

    assert caught.value.reason_code == "docker_serving_listener_invalid"
