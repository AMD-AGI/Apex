"""Bounded Docker CLI observations used by the Magpie execution attestor."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.core import ContractError, sha256_json
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)


_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class DockerImageObservation:
    reference: str
    image_id: str
    repo_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DockerContainerObservation:
    container_id: str
    name: str
    image_id: str
    configured_image: str
    pid: int
    running: bool
    workspace_mount: Path | None
    inferencex_mount: Path | None
    container_spec_sha256: str
    kfd_exposed: bool
    dri_exposed: bool


class DockerObservationClient(Protocol):
    def resolve_image(self, reference: str) -> DockerImageObservation: ...

    def running_containers(self) -> tuple[DockerContainerObservation, ...]: ...

    def container_state(self, container_id: str) -> str | None: ...


class DockerCliObservationClient:
    """Read the public Docker daemon view with fixed, non-shell argv."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=4 << 20)
        self._environment = build_subprocess_environment(
            {}, inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS
        )

    def resolve_image(self, reference: str) -> DockerImageObservation:
        if not reference or any(character.isspace() for character in reference):
            raise ContractError("Docker image reference is invalid", "invalid_docker_image")
        result = self._run(("docker", "image", "inspect", reference))
        value = _single_inspection(result, "docker_image_observation_failed")
        image_id = value.get("Id")
        digests = value.get("RepoDigests")
        if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id):
            raise ContractError("Docker image ID is invalid", "docker_image_observation_failed")
        if not isinstance(digests, list) or any(not isinstance(item, str) for item in digests):
            raise ContractError("Docker repo digests are invalid", "docker_image_observation_failed")
        return DockerImageObservation(reference, image_id, tuple(sorted(digests)))

    def running_containers(self) -> tuple[DockerContainerObservation, ...]:
        result = self._run(
            (
                "docker", "container", "list", "--no-trunc",
                "--filter", "name=magpie-benchmark-", "--format", "{{.ID}}",
            )
        )
        identifiers = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if len(identifiers) != len(set(identifiers)) or any(
            not _CONTAINER_ID.fullmatch(item) for item in identifiers
        ):
            raise ContractError("Docker container list is invalid", "docker_observer_failed")
        return tuple(self._inspect_container(identifier) for identifier in identifiers)

    def container_state(self, container_id: str) -> str | None:
        if not _CONTAINER_ID.fullmatch(container_id):
            raise ContractError("Docker container ID is invalid", "docker_observer_failed")
        result = self._run(
            (
                "docker", "container", "list", "--all", "--no-trunc",
                "--filter", f"id={container_id}", "--format", "{{.ID}} {{.State}}",
            )
        )
        lines = tuple(line.split() for line in result.stdout.splitlines() if line.strip())
        if not lines:
            return None
        valid_states = {"created", "running", "paused", "restarting", "removing", "exited", "dead"}
        if (
            len(lines) != 1
            or len(lines[0]) != 2
            or lines[0][0] != container_id
            or lines[0][1] not in valid_states
        ):
            raise ContractError("Docker container state is ambiguous", "docker_observer_failed")
        return lines[0][1]

    def _inspect_container(self, identifier: str) -> DockerContainerObservation:
        result = self._run(("docker", "container", "inspect", identifier))
        value = _single_inspection(result, "docker_observer_failed")
        return _container_observation(value, identifier)

    def _run(self, argv: tuple[str, ...]) -> ProcessResult:
        result = self._supervisor.run(
            argv,
            cwd=Path("/"),
            environment=self._environment,
            timeout_seconds=15,
        )
        if (
            result.exit_code != 0
            or result.timed_out
            or result.stdout_truncated
            or result.stderr_truncated
            or not result.cleanup_succeeded
        ):
            raise ContractError("Docker observation command failed", "docker_observer_failed")
        return result


def _single_inspection(result: ProcessResult, reason: str) -> Mapping[str, Any]:
    try:
        value = json.loads(result.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("Docker inspection is invalid JSON", reason) from error
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise ContractError("Docker inspection is ambiguous", reason)
    return value[0]


def _container_observation(
    value: Mapping[str, Any], expected_id: str
) -> DockerContainerObservation:
    identifier = value.get("Id")
    image_id = value.get("Image")
    name = value.get("Name")
    state = value.get("State")
    config = value.get("Config")
    host = value.get("HostConfig")
    if not _container_identity_valid(identifier, image_id, name, state, config, host):
        raise ContractError("Docker container inspection is invalid", "docker_observer_failed")
    assert isinstance(state, Mapping) and isinstance(config, Mapping)
    assert isinstance(host, Mapping) and isinstance(identifier, str)
    if identifier != expected_id:
        raise ContractError("Docker container ID drifted", "docker_observer_failed")
    mounts = value.get("Mounts")
    workspace = _mount_source(mounts, "/workspace")
    inferencex = _mount_source(mounts, "/opt/InferenceX")
    paths = _device_paths(host.get("Devices"))
    return DockerContainerObservation(
        identifier,
        str(name).removeprefix("/"),
        str(image_id),
        str(config["Image"]),
        int(state["Pid"]),
        state["Running"] is True,
        workspace,
        inferencex,
        sha256_json({"path": value.get("Path"), "args": value.get("Args")}),
        "/dev/kfd" in paths,
        any(path == "/dev/dri" or path.startswith("/dev/dri/") for path in paths),
    )


def _container_identity_valid(identifier, image_id, name, state, config, host) -> bool:
    return bool(
        isinstance(identifier, str)
        and _CONTAINER_ID.fullmatch(identifier)
        and isinstance(image_id, str)
        and _IMAGE_ID.fullmatch(image_id)
        and isinstance(name, str)
        and isinstance(state, Mapping)
        and state.get("Running") is True
        and isinstance(state.get("Pid"), int)
        and not isinstance(state.get("Pid"), bool)
        and state["Pid"] > 0
        and isinstance(config, Mapping)
        and isinstance(config.get("Image"), str)
        and isinstance(host, Mapping)
    )


def _mount_source(value: object, destination: str) -> Path | None:
    if not isinstance(value, list):
        raise ContractError("Docker mounts are invalid", "docker_observer_failed")
    matches = [item for item in value if isinstance(item, Mapping) and item.get("Destination") == destination]
    if not matches:
        return None
    if len(matches) != 1:
        raise ContractError("Docker mount is ambiguous", "docker_observer_failed")
    item = matches[0]
    source = item.get("Source")
    if item.get("Type") != "bind" or not isinstance(source, str) or not Path(source).is_absolute():
        raise ContractError("Docker bind mount is invalid", "docker_observer_failed")
    return Path(source).resolve()


def _device_paths(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise ContractError("Docker device mappings are invalid", "docker_observer_failed")
    paths: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("PathOnHost"), str):
            raise ContractError("Docker device mapping is invalid", "docker_observer_failed")
        paths.add(item["PathOnHost"])
    return frozenset(paths)


__all__ = [
    "DockerCliObservationClient",
    "DockerContainerObservation",
    "DockerImageObservation",
    "DockerObservationClient",
]
