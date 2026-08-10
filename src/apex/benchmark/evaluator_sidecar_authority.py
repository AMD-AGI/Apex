"""Evaluator-owned Docker sidecar execution at the Magpie quality handoff."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from apex.core import ConfigurationError, canonical_json_bytes, sha256_json
from apex.execution import ProcessResult

from .docker_observation import DockerContainerObservation
from .docker_serving_listener import (
    DockerServingListenerAuthority,
    DockerServingListenerReceipt,
)
from .evaluator_execution import LmEvalExecutionReceipt
from .evaluator_output_publication import (
    PublishedEvaluatorOutputs,
    publish_evaluator_outputs,
)
from .evaluator_preparation import PreparedLmEvalExecution
from .evaluator_serving_broker import (
    EvaluatorServingBroker,
    EvaluatorServingBrokerReceipt,
    EvaluatorServingBrokerSession,
)
from .evaluator_sidecar_docker import (
    EvaluatorSidecarDockerCliClient,
    EvaluatorSidecarDockerObservation,
    validate_observation_against_spec,
)
from .evaluator_sidecar_spec import (
    EvaluatorSidecarDockerSpec,
    build_evaluator_sidecar_spec,
)


class _RuntimePublication(Protocol):
    runtime_probe_sha256: str
    evidence: Mapping[str, object]


class _DockerClient(Protocol):
    def create(self, spec: EvaluatorSidecarDockerSpec) -> str: ...
    def inspect(self, container: str) -> EvaluatorSidecarDockerObservation: ...
    def start_attach(self, container: str, *, timeout_seconds: int) -> ProcessResult: ...
    def stop(self, container: str) -> None: ...
    def remove(self, container: str) -> None: ...


@dataclass(frozen=True, slots=True)
class _SidecarRun:
    spec: EvaluatorSidecarDockerSpec
    container_id: str
    created: EvaluatorSidecarDockerObservation
    exited: EvaluatorSidecarDockerObservation
    broker: EvaluatorServingBrokerReceipt
    listener: DockerServingListenerReceipt
    cleanup_sha256: str


class DockerEvaluatorSidecarAuthority:
    """Run, observe, clean, and receipt one exact no-network evaluator."""

    def __init__(
        self,
        container_supplier: Callable[[], DockerContainerObservation | None],
        *,
        listener: DockerServingListenerAuthority | None = None,
        docker: _DockerClient | None = None,
        broker: EvaluatorServingBroker | None = None,
        spec_builder: Callable[
            [PreparedLmEvalExecution, Path], EvaluatorSidecarDockerSpec
        ] = build_evaluator_sidecar_spec,
        output_publisher: Callable[..., PublishedEvaluatorOutputs] = (
            publish_evaluator_outputs
        ),
        runtime_publisher: Callable[
            [PreparedLmEvalExecution, Path], _RuntimePublication
        ] | None = None,
    ) -> None:
        self._container_supplier = container_supplier
        self._listener = listener or DockerServingListenerAuthority()
        self._docker = docker or EvaluatorSidecarDockerCliClient()
        self._broker = broker or EvaluatorServingBroker()
        self._spec_builder = spec_builder
        self._output_publisher = output_publisher
        self._runtime_publisher = runtime_publisher or _publish_runtime

    def execute(
        self, prepared: PreparedLmEvalExecution
    ) -> LmEvalExecutionReceipt:
        container, workspace = _main_container(self._container_supplier(), prepared)
        listener = self._listener.observe(container, prepared.contract.endpoint_port)
        run = self._run_sidecar(prepared, container, listener)
        _persist_run_evidence(prepared.authority_root, run)
        published = self._output_publisher(
            prepared.output_root,
            workspace,
            contract_sha256=prepared.contract.sha256,
        )
        runtime = self._runtime_publisher(prepared, workspace)
        _write_new(
            prepared.authority_root / "lm_eval_runtime_publication.json",
            _runtime_publication(runtime),
        )
        return _execution_receipt(prepared, run, runtime, published)

    def _run_sidecar(
        self,
        prepared: PreparedLmEvalExecution,
        container: DockerContainerObservation,
        listener: DockerServingListenerReceipt,
    ) -> _SidecarRun:
        broker_session = self._broker.start(
            prepared.authority_root / "broker",
            serving_port=prepared.contract.endpoint_port,
            listener_receipt_sha256=listener.sha256,
            max_connections=prepared.contract.sidecar_connection_limit,
        )
        try:
            spec = self._spec_builder(prepared, broker_session.root)
        except Exception as error:
            try:
                self._broker.abort(broker_session)
            except Exception as cleanup_error:
                raise _invalid(
                    _failure("sidecar_broker_cleanup_failed", cleanup_error)
                ) from error
            raise
        return self._supervise(prepared, container, listener, broker_session, spec)

    def _supervise(
        self,
        prepared: PreparedLmEvalExecution,
        main: DockerContainerObservation,
        listener: DockerServingListenerReceipt,
        broker_session: EvaluatorServingBrokerSession,
        spec: EvaluatorSidecarDockerSpec,
    ) -> _SidecarRun:
        container_id: str | None = None
        created: EvaluatorSidecarDockerObservation | None = None
        broker_stopped = False
        removed = False
        try:
            container_id = self._docker.create(spec)
            created = self._docker.inspect(container_id)
            _require_created(spec, container_id, created)
            result = self._docker.start_attach(
                container_id, timeout_seconds=prepared.contract.timeout_seconds
            )
            exited = self._docker.inspect(container_id)
            _require_exited(spec, container_id, exited, result)
            try:
                broker = self._broker.stop(broker_session)
            finally:
                broker_stopped = True
            after = self._listener.observe(main, prepared.contract.endpoint_port)
            if after.sha256 != listener.sha256:
                raise _invalid("Magpie serving listener changed during evaluation")
            self._docker.remove(container_id)
            removed = True
            cleanup = _cleanup_receipt(container_id, exited)
            return _SidecarRun(
                spec, container_id, created, exited, broker, listener,
                str(cleanup["cleanup_sha256"]),
            )
        except Exception as error:
            failures = self._cleanup(
                broker_session, broker_stopped, container_id, removed
            )
            if failures:
                raise _invalid(";".join(failures)) from error
            raise

    def _cleanup(
        self,
        broker: EvaluatorServingBrokerSession,
        broker_stopped: bool,
        container_id: str | None,
        removed: bool,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        if not broker_stopped:
            try:
                self._broker.abort(broker)
            except Exception as error:
                failures.append(_failure("sidecar_broker_cleanup_failed", error))
        if container_id is not None and not removed:
            try:
                self._docker.remove(container_id)
            except Exception:
                stop_error: Exception | None = None
                try:
                    self._docker.stop(container_id)
                except Exception as error:
                    stop_error = error
                try:
                    self._docker.remove(container_id)
                except Exception as error:
                    if stop_error is not None:
                        failures.append(_failure("sidecar_stop_failed", stop_error))
                    failures.append(_failure("sidecar_remove_failed", error))
        return tuple(failures)


def _main_container(
    container: DockerContainerObservation | None,
    prepared: PreparedLmEvalExecution,
) -> tuple[DockerContainerObservation, Path]:
    if (
        container is None
        or not container.running
        or container.workspace_mount is None
        or container.inferencex_mount != prepared.inferencex_projection.root
    ):
        raise _invalid("Magpie container is unavailable at evaluator handoff")
    return container, container.workspace_mount.resolve(strict=True)


def _require_created(
    spec: EvaluatorSidecarDockerSpec,
    identifier: str,
    observed: EvaluatorSidecarDockerObservation,
) -> None:
    validate_observation_against_spec(spec, observed)
    if (
        observed.container_id != identifier
        or observed.state != "created"
        or observed.running
        or observed.exit_code != 0
    ):
        raise _invalid("Evaluator sidecar was not observed in created state")


def _require_exited(
    spec: EvaluatorSidecarDockerSpec,
    identifier: str,
    observed: EvaluatorSidecarDockerObservation,
    result: ProcessResult,
) -> None:
    validate_observation_against_spec(spec, observed)
    valid_process = (
        result.exit_code == 0
        and not result.timed_out
        and not result.stdout_truncated
        and not result.stderr_truncated
        and result.cleanup_succeeded
    )
    if (
        not valid_process
        or observed.container_id != identifier
        or observed.state != "exited"
        or observed.running
        or observed.exit_code != 0
    ):
        raise _invalid("Evaluator sidecar did not exit successfully")


def _cleanup_receipt(
    container_id: str, exited: EvaluatorSidecarDockerObservation
) -> dict[str, object]:
    payload = {
        "schema": "apex.evaluator-sidecar-cleanup/v1",
        "container_id": container_id,
        "exited_observation_sha256": exited.sha256,
        "remove_argv": ["docker", "container", "rm", container_id],
        "removed": True,
        "verified": True,
    }
    return {**payload, "cleanup_sha256": sha256_json(payload)}


def _persist_run_evidence(root: Path, run: _SidecarRun) -> None:
    values = {
        "serving_listener_receipt.json": run.listener.to_dict(),
        "sidecar_spec.json": run.spec.to_dict(),
        "sidecar_created_observation.json": run.created.to_dict(),
        "sidecar_exited_observation.json": run.exited.to_dict(),
        "serving_broker_receipt.json": run.broker.to_dict(),
        "sidecar_cleanup_receipt.json": _cleanup_receipt(
            run.container_id, run.exited
        ),
    }
    for name, value in values.items():
        _write_new(root / name, value)


def _execution_receipt(
    prepared: PreparedLmEvalExecution,
    run: _SidecarRun,
    runtime: _RuntimePublication,
    published: PublishedEvaluatorOutputs,
) -> LmEvalExecutionReceipt:
    contract = prepared.contract
    return LmEvalExecutionReceipt(
        contract.sha256, contract.config_sha256, contract.policy_sha256,
        contract.policy_lock_sha256, contract.task_definition_sha256,
        contract.effective_task_definition_sha256,
        contract.task_materialization_receipt_sha256,
        contract.dataset_receipt_sha256, contract.dataset_revision,
        contract.runtime_sha256, contract.runtime_manifest_sha256,
        contract.runtime_lock_sha256, contract.launcher_sha256,
        contract.image_repo_digest, contract.image_id, run.container_id,
        run.listener.sha256, run.spec.sha256, run.created.sha256,
        run.exited.sha256, run.broker.sha256, run.cleanup_sha256,
        runtime.runtime_probe_sha256,
        str(_runtime_publication(runtime)["publication_sha256"]),
        published.result_artifacts, published.sample_artifacts,
    )


def _runtime_publication(runtime: _RuntimePublication) -> dict[str, object]:
    payload = {
        "schema": "apex.lm-eval-runtime-publication/v1",
        "runtime_probe_sha256": runtime.runtime_probe_sha256,
        "evidence": dict(runtime.evidence),
    }
    return {**payload, "publication_sha256": sha256_json(payload)}


def _publish_runtime(
    prepared: PreparedLmEvalExecution, workspace: Path
) -> _RuntimePublication:
    from .evaluator_runtime_publication import publish_lm_eval_runtime_evidence

    return publish_lm_eval_runtime_evidence(prepared, workspace)


def _write_new(path: Path, value: dict[str, object]) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
    )
    try:
        payload = canonical_json_bytes(value) + b"\n"
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise _invalid("Cannot persist evaluator sidecar evidence")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _failure(prefix: str, error: Exception) -> str:
    return f"{prefix}:{getattr(error, 'reason_code', type(error).__name__)}"


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_sidecar_authority_invalid")


__all__ = ["DockerEvaluatorSidecarAuthority"]
