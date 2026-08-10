from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from apex.benchmark.docker_magpie_attestor import (
    DockerOneShotMagpieExecutionAttestor,
)
from apex.benchmark.docker_observation import (
    DockerCliObservationClient,
    DockerContainerObservation,
    DockerImageObservation,
)
from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_execution import (
    LmEvalExecutionContract,
    LmEvalExecutionReceipt,
)
from apex.benchmark.evaluator_handoff import CompletedEvaluatorHandoff
from apex.benchmark.evaluator_inferencex_projection import (
    EvaluatorInferenceXProjectionReceipt,
)
from apex.benchmark.magpie_launch_projection import MagpieLaunchConfigReceipt
from apex.benchmark.magpie_attestation import load_magpie_execution_attestation
from apex.core import ContractError, sha256_file, sha256_json
from apex.execution import ProcessResult
from apex.ports import BenchmarkPass, MagpieAttestationRequest
from apex.runtime import DependencyReceipt


_IMAGE = "sha256:" + "1" * 64
_CONTAINER = "2" * 64


class StateSupervisor:
    def __init__(self, output: str) -> None:
        self.output = output
        self.argv: tuple[str, ...] | None = None

    def run(self, argv, **kwargs):
        del kwargs
        self.argv = tuple(argv)
        return ProcessResult(
            tuple(argv), 0, False, self.output, "", False, False, 0.01
        )


class FakeDocker:
    def __init__(self, container: DockerContainerObservation) -> None:
        self.container = container
        self.running = False
        self.list_calls = 0
        self.image = DockerImageObservation("example/image:fixed", _IMAGE, ())
        self.extra: tuple[DockerContainerObservation, ...] = ()
        self.state_after: str | None = None

    def resolve_image(self, reference: str) -> DockerImageObservation:
        assert reference == self.image.reference
        return self.image

    def running_containers(self) -> tuple[DockerContainerObservation, ...]:
        self.list_calls += 1
        return (self.container, *self.extra) if self.running else ()

    def container_state(self, container_id: str) -> str | None:
        assert container_id == self.container.container_id
        return self.state_after


class FakeGpu:
    def __init__(self, *, engaged: bool = True) -> None:
        self.engaged = engaged

    def observe(self, container, gpu_lease):
        assert container.container_id == _CONTAINER
        assert gpu_lease["owner_pid"] == os.getpid()
        if not self.engaged:
            return None
        return {
            "devices": [{"rsmi_index": 0, "unique_id": "GPU-0000000000000001"}],
            "processes": [{
                "pid": 321,
                "uid": 1000,
                "start_time_ticks": 654,
                "cmdline_sha256": "6" * 64,
                "rsmi_device_indices": [0],
                "container_id": _CONTAINER,
            }],
        }


def _receipt(tmp_path: Path) -> DependencyReceipt:
    roots = {name: tmp_path / name for name in ("magpie", "tracelens", "inferencex")}
    for root in roots.values():
        root.mkdir()
    return DependencyReceipt(
        "apex.dependencies.receipt/v1",
        "3" * 64,
        Path("/usr/bin/python3"),
        roots,
        {name: "4" * 40 for name in roots},
        {},
    )


def _dependency_snapshot(receipt: DependencyReceipt):
    return {
        "lock_sha256": receipt.lock_sha256,
        "dependencies": {
            name: {"root": str(root.resolve()), "commit": "4" * 40, "tree": "5" * 40}
            for name, root in receipt.roots.items()
        },
    }


def _lease() -> dict[str, object]:
    return {
        "owner_pid": os.getpid(),
        "ownership": {
            "selector_scope": "0",
            "selected_devices": [
                {"rsmi_index": 0, "unique_id": "GPU-0000000000000001"}
            ],
        },
    }


def _request(root: Path, config: Path, **changes) -> MagpieAttestationRequest:
    values = {
        "run_id": "baseline-measurement",
        "pass_type": BenchmarkPass.MEASUREMENT,
        "config_path": config,
        "run_root": root,
        "benchmark_argv": ("python", "-m", "Magpie"),
        "config_sha256": sha256_file(config),
        "execution_mode": "docker",
        "lifecycle": "one_shot",
        "requested_image": "example/image:fixed",
        "gpu_lease": _lease(),
    }
    values.update(changes)
    return MagpieAttestationRequest(**values)


def _container(workspace: Path, inferencex: Path) -> DockerContainerObservation:
    return DockerContainerObservation(
        _CONTAINER,
        "magpie-benchmark-bench_123",
        _IMAGE,
        "example/image:fixed",
        222,
        True,
        workspace.resolve(),
        inferencex.resolve(),
        "6" * 64,
        True,
        True,
    )


def _report(workspace: Path) -> Path:
    workspace.mkdir(exist_ok=True)
    path = workspace / "benchmark_report.json"
    path.write_text(
        json.dumps(
            {
                "success": True,
                "framework": "vllm",
                "model": "Qwen/example",
                "workspace_dir": str(workspace.resolve()),
                "profiling_enabled": False,
                "throughput": {"total_token_throughput": 10.0},
                "latency": {"ttft": {"p99_ms": 3.0}, "tpot": {"p99_ms": 1.0}},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _wait_for_observation(docker: FakeDocker) -> None:
    deadline = time.monotonic() + 2
    while docker.list_calls < 2 and time.monotonic() < deadline:
        time.sleep(0.002)
    assert docker.list_calls >= 2


def test_observes_container_gpu_and_writes_bound_sidecar(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    run_root = tmp_path / "results" / "baseline-measurement" / "measurement"
    run_root.mkdir(parents=True)
    workspace = run_root / "benchmark_vllm_1"
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    docker = FakeDocker(_container(workspace, receipt.root("inferencex")))
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt,
        docker=docker,
        gpu=FakeGpu(),
        dependency_observer=_dependency_snapshot,
        poll_seconds=0.001,
    )

    session = attestor.prepare(_request(run_root, config))
    report = _report(workspace)
    docker.running = True
    _wait_for_observation(docker)
    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )

    assert path == run_root / "evaluator" / "execution_attestation.json"
    evidence = load_magpie_execution_attestation(
        path,
        report_path=report,
        report=json.loads(report.read_text(encoding="utf-8")),
        expected_config_sha256=sha256_file(config),
        expected_run_id="baseline-measurement",
        expected_pass_type=BenchmarkPass.MEASUREMENT,
        command_exit_code=0,
        timed_out=False,
    )
    assert evidence.gpu_engagement["verified"] is True
    assert evidence.runtime["serving_runtime_receipt"]["resolved_image_id"] == _IMAGE
    assert (
        evidence.runtime["serving_runtime_receipt"]["schema"]
        == "apex.magpie-serving-runtime-observation/v3"
    )
    assert "docker_argv_sha256" not in evidence.runtime["serving_runtime_receipt"]
    assert "execution_attestation_quality_unverified" in evidence.verdict_errors()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"execution_mode": "local"}, "magpie_observer_mode_unavailable"),
        ({"lifecycle": "reuse"}, "magpie_observer_lifecycle_unavailable"),
        ({"lifecycle": "cleanup"}, "magpie_observer_lifecycle_unavailable"),
    ],
)
def test_rejects_unsupported_modes_before_docker(
    tmp_path: Path, change: dict[str, object], reason: str
) -> None:
    receipt = _receipt(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    docker = FakeDocker(_container(root / "workspace", receipt.root("inferencex")))
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt, docker=docker, gpu=FakeGpu(), dependency_observer=_dependency_snapshot
    )

    with pytest.raises(ContractError) as caught:
        attestor.prepare(_request(root, config, **change))

    assert caught.value.reason_code == reason
    assert docker.list_calls == 0


def test_support_matrix_is_docker_one_shot_only(tmp_path: Path) -> None:
    supports = DockerOneShotMagpieExecutionAttestor.supports

    assert supports("docker", "one_shot") is True
    assert supports("local", "one_shot") is False
    assert supports("ray", "one_shot") is False
    assert supports("docker", "reuse") is False
    assert supports("docker", "cleanup") is False
    formal = DockerOneShotMagpieExecutionAttestor(
        _receipt(tmp_path)
    ).formal_measurement_support("docker", "one_shot")
    assert not formal.available
    assert "magpie_inferencex_eval_argument_mismatch" in formal.blockers


@pytest.mark.parametrize(
    ("output", "expected"),
    [("", None), (f"{_CONTAINER} running\n", "running"), (f"{_CONTAINER} exited\n", "exited")],
)
def test_container_cleanup_query_uses_exact_id_and_all_states(
    output: str, expected: str | None
) -> None:
    supervisor = StateSupervisor(output)
    client = DockerCliObservationClient(supervisor)

    assert client.container_state(_CONTAINER) == expected
    assert supervisor.argv == (
        "docker", "container", "list", "--all", "--no-trunc",
        "--filter", f"id={_CONTAINER}", "--format", "{{.ID}} {{.State}}",
    )


def test_gpu_non_engagement_is_recorded_fail_closed(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    workspace = root / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    docker = FakeDocker(_container(workspace, receipt.root("inferencex")))
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt, docker=docker, gpu=FakeGpu(engaged=False),
        dependency_observer=_dependency_snapshot, poll_seconds=0.001,
    )
    session = attestor.prepare(_request(root, config))
    report = _report(workspace)
    docker.running = True
    _wait_for_observation(docker)

    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )
    value = json.loads(path.read_text(encoding="utf-8"))

    assert value["gpu_engagement"]["verified"] is False
    assert "magpie_gpu_not_engaged" in value["errors"]
    assert value["reward_eligible"] is False


def test_image_drift_after_observation_is_recorded_fail_closed(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    workspace = root / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    docker = FakeDocker(_container(workspace, receipt.root("inferencex")))
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt, docker=docker, gpu=FakeGpu(),
        dependency_observer=_dependency_snapshot, poll_seconds=0.001,
    )
    session = attestor.prepare(_request(root, config))
    report = _report(workspace)
    docker.running = True
    _wait_for_observation(docker)
    docker.image = DockerImageObservation("example/image:fixed", "sha256:" + "9" * 64, ())

    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )
    value = json.loads(path.read_text(encoding="utf-8"))

    assert "docker_image_changed_during_execution" in value["errors"]
    assert value["reward_eligible"] is False


def test_multiple_run_bound_containers_are_recorded_fail_closed(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    workspace = root / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    docker = FakeDocker(_container(workspace, receipt.root("inferencex")))
    second = DockerContainerObservation(
        "7" * 64, "magpie-benchmark-second", _IMAGE, "example/image:fixed",
        333, True, workspace.resolve(), receipt.root("inferencex").resolve(),
        "8" * 64, True, True,
    )
    docker.extra = (second,)
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt, docker=docker, gpu=FakeGpu(),
        dependency_observer=_dependency_snapshot, poll_seconds=0.001,
    )
    session = attestor.prepare(_request(root, config))
    report = _report(workspace)
    docker.running = True
    _wait_for_observation(docker)
    time.sleep(0.01)

    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )
    value = json.loads(path.read_text(encoding="utf-8"))

    assert "docker_observer_failed:ambiguous_magpie_container" in value["errors"]
    assert value["reward_eligible"] is False


@pytest.mark.parametrize("remaining", ["running", "exited"])
def test_lingering_or_stopped_container_is_recorded_fail_closed(
    tmp_path: Path, remaining: str
) -> None:
    receipt = _receipt(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    workspace = root / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    docker = FakeDocker(_container(workspace, receipt.root("inferencex")))
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt, docker=docker, gpu=FakeGpu(),
        dependency_observer=_dependency_snapshot, poll_seconds=0.001,
    )
    session = attestor.prepare(_request(root, config))
    report = _report(workspace)
    docker.running = True
    _wait_for_observation(docker)
    docker.running = remaining == "running"
    docker.state_after = remaining

    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )
    value = json.loads(path.read_text(encoding="utf-8"))

    assert f"magpie_container_not_removed:{remaining}" in value["errors"]
    assert value["runtime"]["verified"] is False
    assert value["runtime"]["serving_runtime_receipt"]["verified"] is False
    assert value["reward_eligible"] is False


class _FormalPreparer:
    def __init__(self, prepared) -> None:
        self.prepared = prepared

    def prepare(self, request):
        assert request.pass_type is BenchmarkPass.MEASUREMENT
        return self.prepared


class _FormalBarrier:
    def __init__(self, completed: CompletedEvaluatorHandoff) -> None:
        self.completed = completed
        self.started = False
        self.aborted: str | None = None

    def start(self, prepared, authority):
        self.started = True
        return SimpleNamespace(prepared=prepared, authority=authority)

    def complete(self, session):
        assert session.prepared is not None
        return self.completed

    def abort(self, session, *, reason):
        assert session.prepared is not None
        self.aborted = reason


def _formal_contract(config: Path) -> LmEvalExecutionContract:
    return LmEvalExecutionContract(
        "baseline-measurement", sha256_file(config), "Qwen/example", 8888,
        "3" * 64, "4" * 64, "5" * 64, "6" * 64, "7" * 64,
        "8" * 64, "9" * 40, "a" * 64, "b" * 64, "c" * 64,
        "d" * 64, "example/eval@sha256:" + "e" * 64,
        "sha256:" + "f" * 64, 2248, 480, 64, 30,
    )


def _formal_prepared(tmp_path: Path, config: Path):
    contract = _formal_contract(config)
    authority = tmp_path / "run" / "authority" / "lm_eval"
    projection = authority / "inferencex"
    projection.mkdir(parents=True)
    launch = authority / "magpie-launch.yaml"
    launch.write_text("benchmark: {}\n", encoding="utf-8")
    launch.chmod(0o400)
    projection_receipt = EvaluatorInferenceXProjectionReceipt(
        "1" * 40, "2" * 40, "3" * 40, "4" * 40,
        "5" * 64, "6" * 64, "7" * 64, "8" * 64, "9" * 64,
    )
    prepared_projection = SimpleNamespace(
        root=projection.resolve(), receipt=projection_receipt
    )
    launch_receipt = MagpieLaunchConfigReceipt(
        contract.config_sha256, sha256_file(launch), "/source/inferencex",
        str(projection.resolve()), projection_receipt.sha256,
    )
    return SimpleNamespace(
        contract=contract,
        authority_root=authority,
        launch_config_path=launch,
        launch_config_receipt=launch_receipt,
        inferencex_projection=prepared_projection,
        inferencex_projection_receipt=projection_receipt,
    )


def _formal_completion(workspace: Path, contract: LmEvalExecutionContract):
    output = workspace / "evaluator" / contract.sha256
    output.mkdir(parents=True)
    results = output / "results.json"
    samples = output / "samples_gsm8k.jsonl"
    results.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 1.0}}}),
        encoding="utf-8",
    )
    samples.write_text('{"doc_id": 1}\n', encoding="utf-8")
    execution = LmEvalExecutionReceipt(
        contract.sha256, contract.config_sha256, contract.policy_sha256,
        contract.policy_lock_sha256, contract.task_definition_sha256,
        contract.effective_task_definition_sha256,
        contract.task_materialization_receipt_sha256,
        contract.dataset_receipt_sha256, contract.dataset_revision,
        contract.runtime_sha256, contract.runtime_manifest_sha256,
        contract.runtime_lock_sha256, contract.launcher_sha256,
        contract.image_repo_digest, contract.image_id, "a" * 64,
        "b" * 64, "c" * 64, "d" * 64, "e" * 64, "f" * 64,
        "0" * 64, "1" * 64, "2" * 64,
        (EvaluatorArtifactReceipt(
            results.relative_to(workspace).as_posix(), results.stat().st_size,
            sha256_file(results),
        ),),
        (EvaluatorArtifactReceipt(
            samples.relative_to(workspace).as_posix(), samples.stat().st_size,
            sha256_file(samples),
        ),),
    )
    handoff = workspace.parent / "authority" / "handoff.json"
    handoff.write_text("{}\n", encoding="utf-8")
    return CompletedEvaluatorHandoff(execution, handoff, "3" * 64)


def test_formal_measurement_launches_private_projection_and_binds_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _receipt(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    workspace = root / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    prepared = _formal_prepared(tmp_path, config)
    completion = _formal_completion(workspace, prepared.contract)
    barrier = _FormalBarrier(completion)
    docker = FakeDocker(_container(workspace, prepared.inferencex_projection.root))
    runtime_evidence = {"schema": "magpie.lm-eval-runtime-evidence/v1"}
    inferencex_evidence = {"schema": "magpie.inferencex-runtime-receipt/v2"}
    monkeypatch.setattr(
        "apex.benchmark.docker_magpie_attestor.verify_inferencex_projection",
        lambda projection: None,
    )
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt,
        docker=docker,
        gpu=FakeGpu(),
        dependency_observer=_dependency_snapshot,
        evaluator_preparer=_FormalPreparer(prepared),
        handoff_barrier=barrier,
        sidecar_factory=lambda supplier: SimpleNamespace(supplier=supplier),
        runtime_loader=lambda root, execution: runtime_evidence,
        inferencex_publisher=lambda *args, **kwargs: inferencex_evidence,
        poll_seconds=0.001,
    )
    canonical_argv = (
        "python", "-m", "Magpie", "benchmark", "--benchmark-config",
        str(config.resolve()), "--output-dir", str(root.resolve()),
    )
    request = _request(root, config, benchmark_argv=canonical_argv)

    session = attestor.prepare(request)
    assert attestor.formal_measurement_support("docker", "one_shot").available
    assert attestor.launch_argv(session)[5] == str(prepared.launch_config_path)
    report = _report(workspace)
    docker.running = True
    _wait_for_observation(docker)
    path = attestor.complete(
        session, report_path=report, command_exit_code=0, timed_out=False
    )
    value = json.loads(path.read_text(encoding="utf-8"))

    assert barrier.started
    assert value["reward_eligible"] is True
    assert value["runtime"]["lm_eval_runtime_receipt"] == runtime_evidence
    assert value["runtime"]["inferencex_runtime_receipt"] == inferencex_evidence
    assert value["quality_gate"]["receipt"][
        "evaluator_execution_receipt"
    ]["receipt_sha256"] == completion.execution_receipt.sha256
    assert value["process"]["argv_sha256"] == sha256_json(
        list(attestor.launch_argv(session))
    )


def test_abort_stops_docker_observer_and_handoff_without_success_receipts(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    workspace = root / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    prepared = _formal_prepared(tmp_path, config)
    barrier = _FormalBarrier(_formal_completion(workspace, prepared.contract))
    attestor = DockerOneShotMagpieExecutionAttestor(
        receipt,
        docker=FakeDocker(
            _container(workspace, prepared.inferencex_projection.root)
        ),
        gpu=FakeGpu(),
        dependency_observer=_dependency_snapshot,
        evaluator_preparer=_FormalPreparer(prepared),
        handoff_barrier=barrier,
        sidecar_factory=lambda supplier: SimpleNamespace(supplier=supplier),
        poll_seconds=0.001,
    )
    canonical = (
        "python", "-m", "Magpie", "benchmark", "--benchmark-config",
        str(config.resolve()), "--output-dir", str(root.resolve()),
    )
    session = attestor.prepare(_request(root, config, benchmark_argv=canonical))

    attestor.abort(session, reason="magpie_launch_argv_invalid")

    assert barrier.aborted == "magpie_launch_argv_invalid"
    assert session.thread is not None and not session.thread.is_alive()
    assert session.completed_handoff is None
    assert session.execution_receipt is None
    assert not (prepared.authority_root / "handoff_receipt.json").exists()
    assert not (root / "evaluator" / "execution_attestation.json").exists()
