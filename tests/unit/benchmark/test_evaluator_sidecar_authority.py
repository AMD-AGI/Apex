from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.benchmark.docker_observation import DockerContainerObservation
from apex.benchmark.docker_serving_listener import DockerServingListenerReceipt
from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_execution import LmEvalExecutionContract
from apex.benchmark.evaluator_output_publication import PublishedEvaluatorOutputs
from apex.benchmark.evaluator_runtime_publication import (
    load_lm_eval_runtime_publication,
)
from apex.benchmark.evaluator_serving_broker import EvaluatorServingBrokerReceipt
from apex.benchmark.evaluator_sidecar_authority import (
    DockerEvaluatorSidecarAuthority,
)
from apex.benchmark.evaluator_sidecar_docker import (
    EvaluatorSidecarDockerObservation,
)
from apex.benchmark.evaluator_sidecar_spec import EvaluatorSidecarDockerSpec
from apex.benchmark.local_process_observation import LocalProcessIdentity
from apex.core import ConfigurationError
from apex.execution import ProcessResult


_MAIN = "1" * 64
_SIDECAR = "2" * 64
_IMAGE = "sha256:" + "3" * 64
_REPO = "example/eval@sha256:" + "4" * 64


class FakeListener:
    def __init__(self, receipt: DockerServingListenerReceipt) -> None:
        self.receipts = [receipt, receipt]
        self.calls = 0

    def observe(self, container, port):
        assert container.container_id == _MAIN
        assert port == 8888
        selected = self.receipts[min(self.calls, len(self.receipts) - 1)]
        self.calls += 1
        return selected


class FakeBroker:
    def __init__(self, root: Path, receipt: EvaluatorServingBrokerReceipt) -> None:
        self.root = root
        self.receipt = receipt
        self.started = False
        self.stopped = False

    def start(self, root, **kwargs):
        assert root == self.root
        assert kwargs["serving_port"] == 8888
        self.root.mkdir()
        self.started = True
        return SimpleNamespace(root=self.root)

    def stop(self, session):
        assert session.root == self.root
        self.stopped = True
        return self.receipt

    def abort(self, session):
        assert session.root == self.root
        self.stopped = True


class FakeDocker:
    def __init__(self, created, exited, result=None) -> None:
        self.created = created
        self.exited = exited
        self.result = result or _result()
        self.inspect_calls = 0
        self.remove_calls = 0
        self.remove_failures = 0
        self.removed = False
        self.stopped = False

    def create(self, spec):
        assert spec.image_id == _IMAGE
        return _SIDECAR

    def inspect(self, identifier):
        assert identifier == _SIDECAR
        self.inspect_calls += 1
        return self.created if self.inspect_calls == 1 else self.exited

    def start_attach(self, identifier, *, timeout_seconds):
        assert identifier == _SIDECAR
        assert timeout_seconds == 30
        return self.result

    def stop(self, identifier):
        assert identifier == _SIDECAR
        self.stopped = True

    def remove(self, identifier):
        assert identifier == _SIDECAR
        self.remove_calls += 1
        if self.remove_calls <= self.remove_failures:
            raise RuntimeError("remove failed")
        self.removed = True


def _contract() -> LmEvalExecutionContract:
    return LmEvalExecutionContract(
        "baseline-measurement", "5" * 64, "Qwen/example", 8888,
        "6" * 64, "7" * 64, "8" * 64, "9" * 64, "a" * 64,
        "b" * 64, "c" * 40, "d" * 64, "e" * 64, "f" * 64,
        "0" * 64, _REPO, _IMAGE, 2248, 480, 64, 30,
    )


def _process(pid: int) -> LocalProcessIdentity:
    return LocalProcessIdentity(
        pid, 1000, 1, pid, pid, pid * 10, "5" * 64,
        ("python",), Path("/"), "6" * 64, (f"0::/docker/{_MAIN}",),
    )


def _listener() -> DockerServingListenerReceipt:
    process = _process(101)
    return DockerServingListenerReceipt(
        _MAIN, 8888, process, (process,), "7" * 64
    )


def _main(workspace: Path, projection: Path) -> DockerContainerObservation:
    return DockerContainerObservation(
        _MAIN, "magpie-benchmark-run", _IMAGE, _REPO, 101, True,
        workspace, projection, "8" * 64, True, True,
    )


def _spec(tmp_path: Path, contract: LmEvalExecutionContract):
    return EvaluatorSidecarDockerSpec(
        "apex-lm-eval-test", _REPO, _IMAGE, 1000, 1000, "/authority",
        {"HF_HUB_OFFLINE": "1"}, (),
        ("python3", "/evaluator/launcher.py"), contract.sha256,
    )


def _observation(spec: EvaluatorSidecarDockerSpec, *, exited: bool):
    return EvaluatorSidecarDockerObservation(
        _SIDECAR, spec.container_name, _IMAGE, _REPO,
        "python3", spec.sidecar_argv[1:], tuple(spec.environment.items()),
        "/authority", "1000:1000", spec.contract_sha256, "none", True,
        ("ALL",), ("no-new-privileges:true",), False, 512,
        (("/tmp", ("mode=1777", "nodev", "noexec", "nosuid", "rw", "size=1073741824")),),
        (), (), (), "exited" if exited else "created", False, 0,
    )


def _result(*, exit_code: int = 0) -> ProcessResult:
    return ProcessResult((), exit_code, False, "", "", False, False, 0.1)


def _prepared(tmp_path: Path, contract: LmEvalExecutionContract):
    authority = tmp_path / "authority"
    authority.mkdir()
    output = authority / "sidecar" / "output"
    output.mkdir(parents=True)
    projection = authority / "inferencex"
    projection.mkdir()
    return SimpleNamespace(
        authority_root=authority,
        output_root=output,
        contract=contract,
        inferencex_projection=SimpleNamespace(root=projection),
    )


def _authority(tmp_path: Path, *, result: ProcessResult | None = None):
    contract = _contract()
    prepared = _prepared(tmp_path, contract)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = _spec(tmp_path, contract)
    created = _observation(spec, exited=False)
    exited = _observation(spec, exited=True)
    docker = FakeDocker(created, exited, result)
    listener_receipt = _listener()
    broker_receipt = EvaluatorServingBrokerReceipt(
        8888, listener_receipt.sha256, contract.sidecar_connection_limit, 3
    )
    broker = FakeBroker(prepared.authority_root / "broker", broker_receipt)
    published = PublishedEvaluatorOutputs(
        workspace / "evaluator" / contract.sha256,
        (EvaluatorArtifactReceipt("evaluator/results.json", 1, "1" * 64),),
        (EvaluatorArtifactReceipt("evaluator/samples.jsonl", 1, "2" * 64),),
    )
    calls = []

    def publish(source, target, *, contract_sha256):
        calls.append((source, target, contract_sha256))
        return published

    authority = DockerEvaluatorSidecarAuthority(
        lambda: _main(workspace, prepared.inferencex_projection.root),
        listener=FakeListener(listener_receipt), docker=docker, broker=broker,
        spec_builder=lambda _prepared, root: spec,
        output_publisher=publish,
        runtime_publisher=lambda _prepared, _workspace: SimpleNamespace(
            runtime_probe_sha256="3" * 64,
            evidence={"schema": "magpie.lm-eval-runtime-evidence/v1"},
        ),
    )
    return authority, prepared, docker, broker, calls


def test_executes_observes_cleans_and_receipts_exact_sidecar(tmp_path: Path) -> None:
    authority, prepared, docker, broker, calls = _authority(tmp_path)

    receipt = authority.execute(prepared)

    assert receipt.contract_sha256 == prepared.contract.sha256
    assert receipt.sidecar_spec_sha256 == _spec(tmp_path, prepared.contract).sha256
    assert receipt.created_observation_sha256 == docker.created.sha256
    assert receipt.exited_observation_sha256 == docker.exited.sha256
    assert receipt.broker_receipt_sha256 == broker.receipt.sha256
    assert receipt.runtime_probe_sha256 == "3" * 64
    assert load_lm_eval_runtime_publication(
        prepared.authority_root, receipt
    ) == {"schema": "magpie.lm-eval-runtime-evidence/v1"}
    assert docker.removed and not docker.stopped
    assert broker.started and broker.stopped
    assert calls == [
        (prepared.output_root, tmp_path / "workspace", prepared.contract.sha256)
    ]
    for name in (
        "serving_listener_receipt.json", "sidecar_spec.json",
        "sidecar_created_observation.json", "sidecar_exited_observation.json",
        "serving_broker_receipt.json", "sidecar_cleanup_receipt.json",
        "lm_eval_runtime_publication.json",
    ):
        assert (prepared.authority_root / name).stat().st_mode & 0o777 == 0o400


def test_rejects_unavailable_main_container_before_sidecar_create(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path, _contract())
    authority = DockerEvaluatorSidecarAuthority(lambda: None)

    with pytest.raises(ConfigurationError, match="unavailable"):
        authority.execute(prepared)


def test_nonzero_sidecar_exit_fails_and_cleans_owned_container(
    tmp_path: Path,
) -> None:
    authority, prepared, docker, broker, calls = _authority(
        tmp_path, result=_result(exit_code=7)
    )
    docker.exited = replace(docker.exited, exit_code=7)

    with pytest.raises(ConfigurationError, match="did not exit successfully"):
        authority.execute(prepared)

    assert docker.removed
    assert broker.stopped
    assert calls == []


def test_listener_swap_rejects_execution_after_sidecar_cleanup(
    tmp_path: Path,
) -> None:
    authority, prepared, docker, _broker, calls = _authority(tmp_path)
    authority._listener.receipts[1] = replace(
        authority._listener.receipts[0], closure_sha256="f" * 64
    )

    with pytest.raises(ConfigurationError, match="listener changed"):
        authority.execute(prepared)

    assert docker.removed
    assert calls == []


def test_spec_failure_aborts_broker_before_container_create(tmp_path: Path) -> None:
    authority, prepared, docker, broker, calls = _authority(tmp_path)

    def fail_spec(_prepared, _root):
        raise ConfigurationError("invalid sidecar spec")

    authority._spec_builder = fail_spec

    with pytest.raises(ConfigurationError, match="invalid sidecar spec"):
        authority.execute(prepared)

    assert broker.started and broker.stopped
    assert docker.inspect_calls == 0
    assert calls == []


def test_failed_execution_stops_then_retries_owned_container_removal(
    tmp_path: Path,
) -> None:
    authority, prepared, docker, broker, calls = _authority(
        tmp_path, result=_result(exit_code=7)
    )
    docker.exited = replace(docker.exited, exit_code=7)
    docker.remove_failures = 1

    with pytest.raises(ConfigurationError, match="did not exit successfully"):
        authority.execute(prepared)

    assert docker.stopped and docker.removed
    assert docker.remove_calls == 2
    assert broker.stopped
    assert calls == []
