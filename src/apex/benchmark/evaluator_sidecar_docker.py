"""Bounded Docker CLI lifecycle and observations for evaluator sidecars."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, sha256_json
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)

from .evaluator_sidecar_spec import EvaluatorSidecarDockerSpec


_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_ENVIRONMENT_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STATES = frozenset({"created", "running", "exited", "dead"})
_TMPFS_OPTIONS = frozenset(
    {"rw", "noexec", "nosuid", "nodev", "size=1073741824", "mode=1777"}
)


@dataclass(frozen=True, slots=True)
class EvaluatorSidecarBindObservation:
    """One daemon-observed bind mount."""

    source: Path
    destination: str
    read_write: bool
    mode: str


@dataclass(frozen=True, slots=True)
class EvaluatorSidecarDockerObservation:
    """Security-relevant fields from one exact Docker inspection."""

    container_id: str
    name: str
    image_id: str
    configured_image: str
    path: str
    args: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    workdir: str
    user: str
    contract_label: str | None
    network_mode: str
    read_only_root: bool
    cap_drop: tuple[str, ...]
    security_opt: tuple[str, ...]
    privileged: bool
    pids_limit: int
    tmpfs: tuple[tuple[str, tuple[str, ...]], ...]
    device_mappings: tuple[tuple[str, str, str], ...]
    device_requests: tuple[str, ...]
    mounts: tuple[EvaluatorSidecarBindObservation, ...]
    state: str
    running: bool
    exit_code: int

    @property
    def environment_map(self) -> Mapping[str, str]:
        return dict(self.environment)

    @property
    def sha256(self) -> str:
        return sha256_json(_observation_payload(self))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.evaluator-sidecar-docker-observation/v1",
            **_observation_payload(self),
            "observation_sha256": self.sha256,
        }


class EvaluatorSidecarDockerCliClient:
    """Create and supervise a sidecar through fixed, non-shell Docker argv."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(
            max_output_bytes=4 * 1024 * 1024
        )
        self._environment = build_subprocess_environment(
            {}, inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS
        )

    def create(self, spec: EvaluatorSidecarDockerSpec) -> str:
        result = self._run(spec.create_argv, timeout_seconds=30)
        _require_success(result, "evaluator_sidecar_create_failed")
        lines = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
        if len(lines) != 1 or not _CONTAINER_ID.fullmatch(lines[0]):
            raise _invalid("Docker create returned an invalid container ID")
        return lines[0]

    def inspect(self, container: str) -> EvaluatorSidecarDockerObservation:
        token = _container_token(container)
        result = self._run(
            ("docker", "container", "inspect", token), timeout_seconds=15
        )
        _require_success(result, "evaluator_sidecar_inspect_failed")
        value = _single_inspection(result.stdout)
        observation = _parse_observation(value)
        _validate_inspected_token(token, observation)
        return observation

    def start_attach(self, container: str, *, timeout_seconds: int) -> ProcessResult:
        token = _container_token(container)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 0 < timeout_seconds <= 24 * 60 * 60
        ):
            raise ValueError("Evaluator sidecar timeout is invalid")
        return self._run(
            ("docker", "container", "start", "--attach", token),
            timeout_seconds=timeout_seconds,
        )

    def stop(self, container: str) -> None:
        token = _container_token(container)
        result = self._run(
            ("docker", "container", "stop", "--time", "10", token),
            timeout_seconds=30,
        )
        _require_success(result, "evaluator_sidecar_stop_failed")

    def remove(self, container: str) -> None:
        token = _container_token(container)
        result = self._run(
            ("docker", "container", "rm", token), timeout_seconds=30
        )
        _require_success(result, "evaluator_sidecar_remove_failed")

    def _run(self, argv: tuple[str, ...], *, timeout_seconds: int) -> ProcessResult:
        return self._supervisor.run(
            argv,
            cwd=Path("/"),
            environment=self._environment,
            timeout_seconds=timeout_seconds,
        )


def validate_observation_against_spec(
    spec: EvaluatorSidecarDockerSpec,
    observation: EvaluatorSidecarDockerObservation,
) -> None:
    """Fail closed unless the daemon view matches the frozen create contract."""

    expected = {
        "name": spec.container_name,
        "image_id": spec.image_id,
        "configured_image": spec.image_repo_digest,
        "path": spec.sidecar_argv[0],
        "args": spec.sidecar_argv[1:],
        "workdir": spec.cwd,
        "user": f"{spec.uid}:{spec.gid}",
        "contract_label": spec.contract_sha256,
        "network_mode": "none",
        "read_only_root": True,
        "cap_drop": ("ALL",),
        "security_opt": ("no-new-privileges:true",),
        "privileged": False,
        "pids_limit": 512,
    }
    for field, value in expected.items():
        if getattr(observation, field) != value:
            raise _drift(f"Evaluator sidecar {field} drifted")
    _validate_environment(spec, observation)
    _validate_tmpfs(observation)
    _validate_devices(observation)
    _validate_mounts(spec, observation)


def _parse_observation(value: Mapping[str, Any]) -> EvaluatorSidecarDockerObservation:
    identifier = _string(value, "Id")
    name = _string(value, "Name").removeprefix("/")
    image_id = _string(value, "Image")
    path = _string(value, "Path")
    args = _string_sequence(value.get("Args"), "container arguments")
    config = _mapping(value.get("Config"), "container config")
    host = _mapping(value.get("HostConfig"), "host config")
    state = _mapping(value.get("State"), "container state")
    if not _CONTAINER_ID.fullmatch(identifier) or not _IMAGE_ID.fullmatch(image_id):
        raise _invalid("Docker inspection identity is invalid")
    if not _CONTAINER_NAME.fullmatch(name):
        raise _invalid("Docker inspection name is invalid")
    status, running, exit_code = _parse_state(state)
    return EvaluatorSidecarDockerObservation(
        container_id=identifier,
        name=name,
        image_id=image_id,
        configured_image=_string(config, "Image"),
        path=path,
        args=args,
        environment=_environment(config.get("Env")),
        workdir=_string(config, "WorkingDir"),
        user=_string(config, "User"),
        contract_label=_contract_label(config.get("Labels")),
        network_mode=_string(host, "NetworkMode"),
        read_only_root=_boolean(host, "ReadonlyRootfs"),
        cap_drop=_string_sequence(host.get("CapDrop"), "capability drops"),
        security_opt=_string_sequence(host.get("SecurityOpt"), "security options"),
        privileged=_boolean(host, "Privileged"),
        pids_limit=_integer(host, "PidsLimit"),
        tmpfs=_tmpfs(host.get("Tmpfs")),
        device_mappings=_devices(host.get("Devices")),
        device_requests=_device_requests(host.get("DeviceRequests")),
        mounts=_mounts(value.get("Mounts")),
        state=status,
        running=running,
        exit_code=exit_code,
    )


def _parse_state(state: Mapping[str, Any]) -> tuple[str, bool, int]:
    status = _string(state, "Status")
    running = _boolean(state, "Running")
    exit_code = _integer(state, "ExitCode")
    if (
        status not in _STATES
        or running != (status == "running")
        or not 0 <= exit_code <= 255
    ):
        raise _invalid("Docker sidecar state is invalid")
    return status, running, exit_code


def _environment(value: object) -> tuple[tuple[str, str], ...]:
    entries = _string_sequence(value, "environment")
    parsed: dict[str, str] = {}
    for entry in entries:
        key, separator, content = entry.partition("=")
        if (
            not separator
            or not _ENVIRONMENT_KEY.fullmatch(key)
            or key in parsed
            or any(character in content for character in "\0\r\n")
        ):
            raise _invalid("Docker sidecar environment is invalid")
        parsed[key] = content
    return tuple(sorted(parsed.items()))


def _tmpfs(value: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    mapping = _mapping(value, "tmpfs")
    parsed: list[tuple[str, tuple[str, ...]]] = []
    for destination, raw_options in mapping.items():
        if not isinstance(destination, str) or not destination.startswith("/"):
            raise _invalid("Docker tmpfs destination is invalid")
        if not isinstance(raw_options, str):
            raise _invalid("Docker tmpfs options are invalid")
        options = raw_options.split(",")
        if not options or any(not item for item in options) or len(options) != len(set(options)):
            raise _invalid("Docker tmpfs options are invalid")
        parsed.append((destination, tuple(sorted(options))))
    return tuple(sorted(parsed))


def _devices(value: object) -> tuple[tuple[str, str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _invalid("Docker device mappings are invalid")
    parsed: list[tuple[str, str, str]] = []
    for item in value:
        mapping = _mapping(item, "device mapping")
        parsed.append(
            (
                _string(mapping, "PathOnHost"),
                _string(mapping, "PathInContainer"),
                _string(mapping, "CgroupPermissions"),
            )
        )
    return tuple(parsed)


def _device_requests(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _invalid("Docker device requests are invalid")
    requests = []
    for item in value:
        mapping = _mapping(item, "device request")
        requests.append(sha256_json(dict(mapping)))
    return tuple(sorted(requests))


def _mounts(value: object) -> tuple[EvaluatorSidecarBindObservation, ...]:
    if not isinstance(value, list):
        raise _invalid("Docker mounts are invalid")
    parsed = [_mount(item) for item in value]
    destinations = [item.destination for item in parsed]
    if len(destinations) != len(set(destinations)):
        raise _invalid("Docker mount destinations are ambiguous")
    return tuple(sorted(parsed, key=lambda item: item.destination))


def _mount(value: object) -> EvaluatorSidecarBindObservation:
    item = _mapping(value, "mount")
    source = _string(item, "Source")
    destination = _string(item, "Destination")
    mode = _string(item, "Mode")
    read_write = _boolean(item, "RW")
    if item.get("Type") != "bind" or not Path(source).is_absolute():
        raise _invalid("Docker sidecar mount is not an absolute bind")
    if not destination.startswith("/"):
        raise _invalid("Docker sidecar mount destination is invalid")
    tokens = frozenset(token for token in mode.split(",") if token)
    if (
        not tokens.issubset({"ro", "rw"})
        or len(tokens) > 1
        or ("ro" in tokens and read_write)
        or ("rw" in tokens and not read_write)
    ):
        raise _invalid("Docker sidecar mount mode conflicts with RW")
    return EvaluatorSidecarBindObservation(Path(source), destination, read_write, mode)


def _validate_environment(
    spec: EvaluatorSidecarDockerSpec,
    observation: EvaluatorSidecarDockerObservation,
) -> None:
    observed = observation.environment_map
    for key, value in spec.environment.items():
        if observed.get(key) != value:
            raise _drift(f"Evaluator sidecar environment {key} drifted")


def _validate_tmpfs(observation: EvaluatorSidecarDockerObservation) -> None:
    expected = (("/tmp", tuple(sorted(_TMPFS_OPTIONS))),)
    if observation.tmpfs != expected:
        raise _drift("Evaluator sidecar tmpfs drifted")


def _validate_devices(observation: EvaluatorSidecarDockerObservation) -> None:
    if observation.device_mappings or observation.device_requests:
        raise _drift("Evaluator sidecar unexpectedly exposes devices")


def _validate_mounts(
    spec: EvaluatorSidecarDockerSpec,
    observation: EvaluatorSidecarDockerObservation,
) -> None:
    try:
        expected = {
            item.destination: (item.source.resolve(strict=True), not item.read_only)
            for item in spec.mounts
        }
        observed = {
            item.destination: (item.source.resolve(strict=True), item.read_write)
            for item in observation.mounts
        }
    except OSError as error:
        raise _drift("Evaluator sidecar bind mount cannot be resolved") from error
    if observed != expected:
        raise _drift("Evaluator sidecar bind mounts drifted")


def _single_inspection(output: str) -> Mapping[str, Any]:
    try:
        value = json.loads(output)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _invalid("Docker sidecar inspection is invalid JSON") from error
    if not isinstance(value, list) or len(value) != 1:
        raise _invalid("Docker sidecar inspection is ambiguous")
    return _mapping(value[0], "inspection")


def _validate_inspected_token(
    token: str, observation: EvaluatorSidecarDockerObservation
) -> None:
    expected = observation.container_id if _CONTAINER_ID.fullmatch(token) else observation.name
    if token != expected:
        raise _drift("Docker inspected container identity drifted")


def _container_token(value: str) -> str:
    if not isinstance(value, str) or not (
        _CONTAINER_ID.fullmatch(value) or _CONTAINER_NAME.fullmatch(value)
    ):
        raise ValueError("Evaluator sidecar container identifier is invalid")
    return value


def _contract_label(value: object) -> str | None:
    if value is None:
        return None
    labels = _mapping(value, "container labels")
    label = labels.get("apex.evaluator.contract")
    if label is not None and not isinstance(label, str):
        raise _invalid("Evaluator sidecar contract label is invalid")
    return label


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid(f"Docker {field} is invalid")
    return value


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or "\0" in item:
        raise _invalid(f"Docker {key} is invalid")
    return item


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise _invalid(f"Docker {key} is invalid")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise _invalid(f"Docker {key} is invalid")
    return item


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _invalid(f"Docker {field} is invalid")
    return tuple(value)


def _require_success(result: ProcessResult, reason: str) -> None:
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
        or not result.cleanup_succeeded
    ):
        raise ContractError("Docker sidecar command failed", reason)


def _observation_payload(value: EvaluatorSidecarDockerObservation) -> dict[str, object]:
    payload = {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
        if field != "mounts"
    }
    payload["mounts"] = [
        {
            "source": str(item.source),
            "destination": item.destination,
            "read_write": item.read_write,
            "mode": item.mode,
        }
        for item in value.mounts
    ]
    return payload


def _invalid(message: str) -> ContractError:
    return ContractError(message, "evaluator_sidecar_observation_invalid")


def _drift(message: str) -> ContractError:
    return ContractError(message, "evaluator_sidecar_observation_drift")


__all__ = [
    "EvaluatorSidecarBindObservation",
    "EvaluatorSidecarDockerCliClient",
    "EvaluatorSidecarDockerObservation",
    "validate_observation_against_spec",
]
