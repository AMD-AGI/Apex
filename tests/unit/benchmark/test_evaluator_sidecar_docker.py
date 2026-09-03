from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from apex.benchmark.evaluator_sidecar_docker import (
    EvaluatorSidecarBindObservation,
    EvaluatorSidecarDockerCliClient,
    validate_observation_against_spec,
)
from apex.benchmark.evaluator_sidecar_spec import (
    EvaluatorSidecarDockerSpec,
    EvaluatorSidecarMount,
)
from apex.core import ContractError
from apex.execution import ProcessResult


_CONTAINER = "2" * 64
_IMAGE = "sha256:" + "3" * 64
_REPO = "example/evaluator@sha256:" + "4" * 64
_CONTRACT = "5" * 64


class FakeSupervisor:
    def __init__(self, results: list[ProcessResult]) -> None:
        self.results = results
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(self, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs))
        result = self.results.pop(0)
        return replace(result, argv=tuple(argv))


def _result(
    stdout: str = "",
    *,
    exit_code: int | None = 0,
    timed_out: bool = False,
    stdout_truncated: bool = False,
    cleanup_succeeded: bool = True,
) -> ProcessResult:
    return ProcessResult(
        (),
        exit_code,
        timed_out,
        stdout,
        "",
        stdout_truncated,
        False,
        0.01,
        cleanup_succeeded=cleanup_succeeded,
    )


def _spec(tmp_path: Path) -> EvaluatorSidecarDockerSpec:
    writable = tmp_path / "authority"
    readonly = tmp_path / "task"
    writable.mkdir()
    readonly.mkdir()
    return EvaluatorSidecarDockerSpec(
        container_name="apex-lm-eval-run-123",
        image_repo_digest=_REPO,
        image_id=_IMAGE,
        uid=1000,
        gid=1001,
        cwd="/authority",
        environment={
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": "/evaluator/runtime/site-packages",
        },
        mounts=(
            EvaluatorSidecarMount("authority", writable, "/authority", False),
            EvaluatorSidecarMount("task", readonly, "/evaluator/task", True),
        ),
        sidecar_argv=("python3", "/evaluator/launcher.py", "--", "python3", "-m", "lm_eval"),
        contract_sha256=_CONTRACT,
        input_projection_sha256="f" * 64,
    )


def _inspection(spec: EvaluatorSidecarDockerSpec) -> dict[str, object]:
    mounts = []
    for mount in spec.mounts:
        mounts.append(
            {
                "Type": "bind",
                "Source": str(mount.source),
                "Destination": mount.destination,
                "Mode": "ro" if mount.read_only else "rw",
                "RW": not mount.read_only,
                "Propagation": "rprivate",
            }
        )
    return {
        "Id": _CONTAINER,
        "Name": f"/{spec.container_name}",
        "Image": _IMAGE,
        "Path": "python3",
        "Args": list(spec.sidecar_argv[1:]),
        "Config": {
            "Image": _REPO,
            "Env": [
                "PATH=/usr/local/bin:/usr/bin",
                "HF_HUB_OFFLINE=1",
                "PYTHONPATH=/evaluator/runtime/site-packages",
            ],
            "WorkingDir": "/authority",
            "User": "1000:1001",
            "Labels": {"apex.evaluator.contract": _CONTRACT},
        },
        "HostConfig": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"],
            "Privileged": False,
            "PidsLimit": 512,
            "Tmpfs": {
                "/tmp": "rw,noexec,nosuid,nodev,size=1073741824,mode=1777"
            },
            "Devices": [],
            "DeviceRequests": None,
        },
        "Mounts": mounts,
        "State": {"Status": "exited", "Running": False, "ExitCode": 0},
    }


def _inspect(spec, payload):
    supervisor = FakeSupervisor([_result(json.dumps([payload]))])
    observation = EvaluatorSidecarDockerCliClient(supervisor).inspect(
        spec.container_name
    )
    return observation, supervisor


def test_create_uses_the_frozen_spec_argv_and_bounded_environment(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    supervisor = FakeSupervisor([_result(_CONTAINER + "\n")])
    client = EvaluatorSidecarDockerCliClient(supervisor)

    assert client.create(spec) == _CONTAINER
    argv, keywords = supervisor.calls[0]
    assert argv == spec.create_argv
    assert keywords["cwd"] == Path("/")
    assert keywords["timeout_seconds"] == 30
    environment = keywords["environment"]
    assert "PATH" in environment
    assert "LD_PRELOAD" not in environment
    assert "DOCKER_AUTH_CONFIG" not in environment


@pytest.mark.parametrize(
    "result",
    [
        _result("not-an-id\n"),
        _result(_CONTAINER + "\nextra\n"),
        _result(_CONTAINER, exit_code=1),
        _result(_CONTAINER, timed_out=True),
        _result(_CONTAINER, stdout_truncated=True),
        _result(_CONTAINER, cleanup_succeeded=False),
    ],
)
def test_create_fails_closed_on_invalid_or_incomplete_commands(
    tmp_path: Path, result: ProcessResult
) -> None:
    with pytest.raises(ContractError):
        EvaluatorSidecarDockerCliClient(FakeSupervisor([result])).create(
            _spec(tmp_path)
        )


def test_inspect_parses_all_security_and_state_fields(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    observation, supervisor = _inspect(spec, _inspection(spec))

    validate_observation_against_spec(spec, observation)
    assert observation.container_id == _CONTAINER
    assert observation.environment_map["PATH"].startswith("/usr/local")
    assert observation.mounts[0].destination == "/authority"
    assert observation.mounts[0].read_write is True
    assert observation.state == "exited"
    assert observation.running is False
    assert observation.exit_code == 0
    assert len(observation.sha256) == 64
    assert supervisor.calls[0][0] == (
        "docker",
        "container",
        "inspect",
        spec.container_name,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("name", "different"),
        ("image_id", "sha256:" + "9" * 64),
        ("configured_image", "other/image@sha256:" + "8" * 64),
        ("path", "bash"),
        ("args", ("wrong",)),
        ("workdir", "/tmp"),
        ("user", "0:0"),
        ("contract_label", "7" * 64),
        ("network_mode", "bridge"),
        ("read_only_root", False),
        ("cap_drop", ()),
        ("security_opt", ()),
        ("privileged", True),
        ("pids_limit", 0),
    ],
)
def test_validation_rejects_identity_and_security_drift(
    tmp_path: Path, field: str, replacement: object
) -> None:
    spec = _spec(tmp_path)
    observation, _ = _inspect(spec, _inspection(spec))
    with pytest.raises(ContractError, match="drifted"):
        validate_observation_against_spec(
            spec, replace(observation, **{field: replacement})
        )


def test_validation_rejects_environment_tmpfs_device_and_mount_drift(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    observation, _ = _inspect(spec, _inspection(spec))
    cases = (
        replace(
            observation,
            environment=(("HF_HUB_OFFLINE", "0"),),
        ),
        replace(observation, tmpfs=()),
        replace(observation, device_requests=("8" * 64,)),
        replace(
            observation,
            mounts=(
                EvaluatorSidecarBindObservation(
                    spec.mounts[0].source, "/authority", False, "ro"
                ),
            ),
        ),
    )
    for changed in cases:
        with pytest.raises(ContractError):
            validate_observation_against_spec(spec, changed)


def test_inspect_rejects_malformed_or_ambiguous_daemon_json(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    payloads = (
        "not-json",
        "[]",
        json.dumps([_inspection(spec), _inspection(spec)]),
        json.dumps([{**_inspection(spec), "Id": "short"}]),
    )
    for output in payloads:
        with pytest.raises(ContractError):
            EvaluatorSidecarDockerCliClient(
                FakeSupervisor([_result(output)])
            ).inspect(spec.container_name)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["Config"].update(Env=["A=1", "A=2"]),
        lambda value: value["Config"].update(Env=["BAD-KEY=value"]),
        lambda value: value["Config"].update(Env=["A=line\nbreak"]),
        lambda value: value["HostConfig"].update(Devices={}),
        lambda value: value["HostConfig"].update(DeviceRequests={}),
        lambda value: value["HostConfig"].update(Tmpfs={"relative": "rw"}),
        lambda value: value["Mounts"][0].update(Type="volume"),
        lambda value: value["Mounts"][0].update(Mode="rw,z"),
        lambda value: value["State"].update(Status="running", Running=False),
        lambda value: value["State"].update(ExitCode=256),
    ],
)
def test_inspect_rejects_malformed_typed_fields(tmp_path: Path, mutator) -> None:
    spec = _spec(tmp_path)
    payload = copy.deepcopy(_inspection(spec))
    mutator(payload)
    with pytest.raises(ContractError):
        _inspect(spec, payload)


def test_start_stop_and_remove_use_fixed_argv_and_preserve_evaluator_exit(
    tmp_path: Path,
) -> None:
    del tmp_path
    attached = _result("evaluator failed", exit_code=7)
    supervisor = FakeSupervisor([attached, _result(), _result()])
    client = EvaluatorSidecarDockerCliClient(supervisor)

    result = client.start_attach(_CONTAINER, timeout_seconds=321)
    client.stop(_CONTAINER)
    client.remove(_CONTAINER)

    assert result.exit_code == 7
    assert [call[0] for call in supervisor.calls] == [
        ("docker", "container", "start", "--attach", _CONTAINER),
        ("docker", "container", "stop", "--time", "10", _CONTAINER),
        ("docker", "container", "rm", _CONTAINER),
    ]
    assert supervisor.calls[0][1]["timeout_seconds"] == 321


@pytest.mark.parametrize("container", ["", "--help", "contains space", "x" * 129])
def test_lifecycle_rejects_untrusted_container_tokens(container: str) -> None:
    client = EvaluatorSidecarDockerCliClient(FakeSupervisor([]))
    with pytest.raises(ValueError):
        client.inspect(container)


@pytest.mark.parametrize("timeout", [0, -1, True, 24 * 60 * 60 + 1])
def test_start_rejects_unbounded_timeout(timeout: int) -> None:
    client = EvaluatorSidecarDockerCliClient(FakeSupervisor([]))
    with pytest.raises(ValueError):
        client.start_attach(_CONTAINER, timeout_seconds=timeout)
