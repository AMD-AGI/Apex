from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

import apex.benchmark.evaluator_handoff as evaluator_handoff
from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_execution import (
    LmEvalExecutionContract,
    LmEvalExecutionReceipt,
)
from apex.benchmark.evaluator_handoff import EvaluatorHandoffBarrier
from apex.benchmark.evaluator_inferencex_projection import (
    materialize_inferencex_projection,
)
from apex.core import ConfigurationError


def _contract(**changes) -> LmEvalExecutionContract:
    values = {
        "run_id": "baseline-measurement",
        "config_sha256": "1" * 64,
        "model": "Qwen/example",
        "endpoint_port": 8888,
        "policy_sha256": "2" * 64,
        "policy_lock_sha256": "3" * 64,
        "task_definition_sha256": "4" * 64,
        "effective_task_definition_sha256": "5" * 64,
        "task_materialization_receipt_sha256": "6" * 64,
        "dataset_receipt_sha256": "7" * 64,
        "dataset_revision": "8" * 40,
        "runtime_sha256": "9" * 64,
        "runtime_manifest_sha256": "a" * 64,
        "runtime_lock_sha256": "b" * 64,
        "launcher_sha256": "0" * 64,
        "image_repo_digest": "example/eval@sha256:" + "c" * 64,
        "image_id": "sha256:" + "d" * 64,
        "max_length": 2248,
        "max_gen_tokens": 480,
        "concurrent_requests": 64,
        "timeout_seconds": 2,
    }
    values.update(changes)
    return LmEvalExecutionContract(**values)


def _execution(contract: LmEvalExecutionContract) -> LmEvalExecutionReceipt:
    return LmEvalExecutionReceipt(
        contract_sha256=contract.sha256,
        config_sha256=contract.config_sha256,
        policy_sha256=contract.policy_sha256,
        policy_lock_sha256=contract.policy_lock_sha256,
        task_definition_sha256=contract.task_definition_sha256,
        effective_task_definition_sha256=contract.effective_task_definition_sha256,
        task_materialization_receipt_sha256=contract.task_materialization_receipt_sha256,
        dataset_receipt_sha256=contract.dataset_receipt_sha256,
        dataset_revision=contract.dataset_revision,
        runtime_sha256=contract.runtime_sha256,
        runtime_manifest_sha256=contract.runtime_manifest_sha256,
        runtime_lock_sha256=contract.runtime_lock_sha256,
        launcher_sha256=contract.launcher_sha256,
        image_repo_digest=contract.image_repo_digest,
        image_id=contract.image_id,
        container_id="e" * 64,
        listener_receipt_sha256="f" * 64,
        sidecar_spec_sha256="1" * 64,
        created_observation_sha256="2" * 64,
        exited_observation_sha256="3" * 64,
        broker_receipt_sha256="4" * 64,
        container_cleanup_sha256="5" * 64,
        runtime_probe_sha256="0" * 64,
        runtime_publication_sha256="6" * 64,
        result_artifacts=(EvaluatorArtifactReceipt("results.json", 1, "1" * 64),),
        sample_artifacts=(EvaluatorArtifactReceipt("samples.jsonl", 1, "2" * 64),),
    )


def _prepared(tmp_path: Path, contract: LmEvalExecutionContract):
    inferencex = tmp_path / "InferenceX"
    library = inferencex / "benchmarks" / "benchmark_lib.sh"
    library.parent.mkdir(parents=True)
    library.write_text("run_eval() { return 9; }\n")
    magpie = tmp_path / "MagpieRoot"
    scripts = magpie / "Magpie" / "scripts" / "benchmark"
    scripts.mkdir(parents=True)
    (scripts / "vllm_mi300x.sh").write_text("run_eval --framework lm-eval\n")
    authority = tmp_path / "authority"
    projection = materialize_inferencex_projection(
        inferencex,
        magpie,
        authority / "inferencex",
        inferencex_commit="1" * 40,
        inferencex_tree="2" * 40,
        magpie_commit="3" * 40,
        magpie_tree="4" * 40,
        execution_contract=contract,
        nonce="5" * 64,
    )
    return SimpleNamespace(
        authority_root=authority,
        inferencex_projection=projection,
        contract=contract,
    )


class _Authority:
    def __init__(self, receipt: LmEvalExecutionReceipt) -> None:
        self.receipt = receipt
        self.calls = 0

    def execute(self, _prepared) -> LmEvalExecutionReceipt:
        self.calls += 1
        return self.receipt


class _FailingAuthority:
    def execute(self, _prepared) -> LmEvalExecutionReceipt:
        raise ConfigurationError(
            "Docker could not create evaluator sidecar",
            "evaluator_sidecar_create_failed",
        )


def _request(prepared, *, argv=None) -> dict[str, object]:
    handoff = json.loads(
        prepared.inferencex_projection.handoff_contract_path.read_text()
    )
    return {
        "schema": "apex.evaluator-handoff-request/v1",
        "run_id": prepared.contract.run_id,
        "execution_contract_sha256": prepared.contract.sha256,
        "nonce": handoff["nonce"],
        "argv": argv
        or [
            "--framework", "lm-eval",
            "--port", "8888",
            "--concurrent-requests", "64",
        ],
    }


def _exchange(path: Path, value: dict[str, object]) -> dict[str, object]:
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(f"/proc/self/fd/{directory}/{path.name}")
            client.sendall(json.dumps(value).encode() + b"\n")
            payload = bytearray()
            while not payload.endswith(b"\n"):
                payload.extend(client.recv(4096))
    finally:
        os.close(directory)
    return json.loads(payload)


def test_blocks_until_exact_sidecar_receipt_then_releases(tmp_path: Path) -> None:
    contract = _contract()
    prepared = _prepared(tmp_path, contract)
    authority = _Authority(_execution(contract))
    barrier = EvaluatorHandoffBarrier()
    session = barrier.start(prepared, authority)

    response = _exchange(
        prepared.inferencex_projection.handoff_socket, _request(prepared)
    )
    completed = barrier.complete(session)

    assert response == {
        "schema": "apex.evaluator-handoff-response/v1",
        "exit_code": 0,
    }
    assert authority.calls == 1
    assert completed.execution_receipt == authority.receipt
    value = json.loads(completed.handoff_receipt_path.read_text())
    ordered = value["ordering_ns"]
    assert ordered["request_received"] <= ordered["sidecar_started"]
    assert ordered["sidecar_started"] <= ordered["sidecar_finished"]
    assert ordered["sidecar_finished"] <= ordered["handoff_released"]


def test_rejects_argument_drift_without_running_sidecar(tmp_path: Path) -> None:
    contract = _contract()
    prepared = _prepared(tmp_path, contract)
    authority = _Authority(_execution(contract))
    barrier = EvaluatorHandoffBarrier()
    session = barrier.start(prepared, authority)

    response = _exchange(
        prepared.inferencex_projection.handoff_socket,
        _request(prepared, argv=["--framework", "lm-eval", "--port", "9999"]),
    )

    assert response["exit_code"] == 125
    assert authority.calls == 0
    with pytest.raises(ConfigurationError, match="differs from its contract"):
        barrier.complete(session)


def test_rejects_sidecar_receipt_swap(tmp_path: Path) -> None:
    contract = _contract()
    prepared = _prepared(tmp_path, contract)
    authority = _Authority(_execution(_contract(config_sha256="f" * 64)))
    barrier = EvaluatorHandoffBarrier()
    session = barrier.start(prepared, authority)

    response = _exchange(
        prepared.inferencex_projection.handoff_socket, _request(prepared)
    )

    assert response["exit_code"] == 125
    with pytest.raises(ConfigurationError, match="evaluator_contract_mismatch"):
        barrier.complete(session)


def test_preserves_typed_sidecar_failure_across_handoff(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, _contract())
    barrier = EvaluatorHandoffBarrier()
    session = barrier.start(prepared, _FailingAuthority())

    response = _exchange(
        prepared.inferencex_projection.handoff_socket, _request(prepared)
    )

    assert response["exit_code"] == 125
    with pytest.raises(ConfigurationError) as caught:
        barrier.complete(session)
    assert caught.value.reason_code == "evaluator_sidecar_create_failed"


def test_abort_cleans_listener_without_execution(tmp_path: Path) -> None:
    contract = _contract()
    prepared = _prepared(tmp_path, contract)
    authority = _Authority(_execution(contract))
    barrier = EvaluatorHandoffBarrier()
    session = barrier.start(prepared, authority)

    barrier.abort(session, reason="magpie_never_started")

    assert not prepared.inferencex_projection.handoff_socket.exists()
    assert authority.calls == 0


def test_listener_closes_when_socket_directory_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Listener:
        closed = False

        def close(self) -> None:
            self.closed = True

    listener = _Listener()

    def _cannot_open(_path: Path) -> int:
        raise OSError("directory unavailable")

    monkeypatch.setattr(evaluator_handoff.socket, "socket", lambda *_args: listener)
    monkeypatch.setattr(evaluator_handoff, "_socket_directory", _cannot_open)

    with pytest.raises(OSError, match="directory unavailable"):
        evaluator_handoff._bind_listener(tmp_path / "handoff.sock")

    assert listener.closed
