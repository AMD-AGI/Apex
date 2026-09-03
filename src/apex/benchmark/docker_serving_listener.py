"""Bind the serving TCP listener to one exact Magpie Docker process tree."""

from __future__ import annotations

import re
from dataclasses import dataclass

from apex.core import ContractError, sha256_json

from .docker_observation import DockerContainerObservation
from .docker_listener_probe import DockerExecListenerProbe
from .local_port_observation import (
    LocalPortObservationClient,
)
from .local_process_observation import (
    LocalProcessIdentity,
    LocalProcessObservationClient,
    ProcfsLocalProcessObservationClient,
    descendant_closure,
)


SCHEMA = "apex.docker-serving-listener-receipt/v1"
_CONTAINER = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class DockerServingListenerReceipt:
    """Exact container root, closure, and listener-owner identities."""

    container_id: str
    port: int
    root_process: LocalProcessIdentity
    listener_processes: tuple[LocalProcessIdentity, ...]
    closure_sha256: str

    def __post_init__(self) -> None:
        if (
            not _CONTAINER.fullmatch(self.container_id)
            or not 1 <= self.port <= 65535
            or not self.listener_processes
            or len({item.pid for item in self.listener_processes})
            != len(self.listener_processes)
            or len(self.closure_sha256) != 64
        ):
            raise ValueError("Docker serving listener receipt is invalid")

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "container_id": self.container_id,
            "port": self.port,
            "root_process": self.root_process.to_dict(),
            "listener_processes": [
                item.to_dict() for item in self.listener_processes
            ],
            "closure_sha256": self.closure_sha256,
            "verified": True,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.sha256}


class DockerServingListenerAuthority:
    """Observe procfs once at handoff and reject any foreign listener owner."""

    def __init__(
        self,
        *,
        processes: LocalProcessObservationClient | None = None,
        ports: LocalPortObservationClient | None = None,
        docker_ports: DockerExecListenerProbe | None = None,
    ) -> None:
        self._processes = processes or ProcfsLocalProcessObservationClient()
        self._ports = ports
        self._docker_ports = docker_ports or DockerExecListenerProbe()

    def observe(
        self, container: DockerContainerObservation, port: int
    ) -> DockerServingListenerReceipt:
        if not container.running:
            raise _invalid("Magpie container is not running at evaluator handoff")
        snapshot = self._processes.snapshot()
        roots = tuple(item for item in snapshot if item.pid == container.pid)
        if len(roots) != 1 or not _in_container(roots[0], container.container_id):
            raise _invalid("Magpie container root process is unavailable")
        closure = tuple(
            sorted(
                descendant_closure(snapshot, roots),
                key=lambda item: (item.pid, item.start_time_ticks),
            )
        )
        owners = (
            self._ports.listener_owners(port, snapshot)
            if self._ports is not None
            else self._docker_ports.listener_owners(
                container.container_id, port, closure
            )
        )
        if not owners or any(item not in closure for item in owners):
            raise _invalid("Serving listener is not contained by Magpie")
        if any(not _in_container(item, container.container_id) for item in closure):
            raise _invalid("Magpie process escaped its container cgroup")
        ordered = tuple(sorted(owners, key=lambda item: item.pid))
        return DockerServingListenerReceipt(
            container.container_id,
            port,
            roots[0],
            ordered,
            sha256_json([item.to_dict() for item in closure]),
        )


def _in_container(process: LocalProcessIdentity, container_id: str) -> bool:
    identifiers = (container_id, container_id[:12])
    return any(
        re.search(rf"(?<![0-9a-f]){identifier}(?![0-9a-f])", line) is not None
        for identifier in identifiers
        for line in process.cgroup_lines
    )


def _invalid(message: str) -> ContractError:
    return ContractError(message, "docker_serving_listener_invalid")


__all__ = [
    "DockerServingListenerAuthority",
    "DockerServingListenerReceipt",
]
