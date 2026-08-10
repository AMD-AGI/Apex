"""Apex-owned live observer for Docker one-shot Magpie executions."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from apex.core import (
    ConfigurationError,
    ContractError,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from apex.ports import (
    BenchmarkPass,
    MagpieAttestationRequest,
    MagpieFormalMeasurementSupport,
    MagpieReportLocation,
)
from apex.runtime import DependencyReceipt
from apex.runtime.magpie_result_contract import EXECUTION_ATTESTATION_SCHEMA
from apex.runtime.repositories import BootstrapError, inspect_repository

from .docker_gpu_observation import ContainerGpuObserver, RocmContainerGpuObserver
from .docker_evaluator_preparation import (
    inferencex_dependency_entry,
    project_docker_evaluator_launch_argv,
    publish_inferencex_runtime,
    validate_prepared_docker_evaluator_inputs,
)
from .docker_observation import (
    DockerCliObservationClient,
    DockerContainerObservation,
    DockerImageObservation,
    DockerObservationClient,
)
from .evaluator_execution import LmEvalExecutionReceipt
from .evaluator_handoff import (
    CompletedEvaluatorHandoff,
    EvaluatorHandoffBarrier,
    EvaluatorHandoffSession,
    EvaluatorSidecarAuthority,
)
from .evaluator_inferencex_projection import verify_inferencex_projection
from .evaluator_preparation import LmEvalExecutionPreparer, PreparedLmEvalExecution
from .evaluator_runtime_publication import load_lm_eval_runtime_publication
from .evaluator_sidecar_authority import DockerEvaluatorSidecarAuthority
from .formal_measurement_support import docker_formal_measurement_support
from .magpie_attestation import expected_attestation_path, locate_local_magpie_report
from .quality import build_lm_eval_quality_gate


@dataclass(slots=True)
class _ObservationState:
    request: MagpieAttestationRequest
    image: DockerImageObservation
    dependencies: Mapping[str, object]
    stop: threading.Event = field(default_factory=threading.Event)
    observed: DockerContainerObservation | None = None
    gpu: Mapping[str, object] | None = None
    errors: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    launch_argv: tuple[str, ...] = ()
    prepared: PreparedLmEvalExecution | None = None
    handoff: EvaluatorHandoffSession | None = None
    completed_handoff: CompletedEvaluatorHandoff | None = None
    execution_receipt: LmEvalExecutionReceipt | None = None
    lm_eval_runtime_evidence: Mapping[str, object] | None = None
    inferencex_runtime_evidence: Mapping[str, object] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class DockerOneShotMagpieExecutionAttestor:
    """Observe one unchanged Magpie Docker launch without importing Magpie."""

    is_available = True

    @staticmethod
    def supports(execution_mode: str, lifecycle: str) -> bool:
        return execution_mode == "docker" and lifecycle == "one_shot"

    def __init__(
        self,
        dependency_receipt: DependencyReceipt,
        *,
        docker: DockerObservationClient | None = None,
        gpu: ContainerGpuObserver | None = None,
        dependency_observer: Callable[[DependencyReceipt], Mapping[str, object]] | None = None,
        evaluator_preparer: LmEvalExecutionPreparer | None = None,
        handoff_barrier: EvaluatorHandoffBarrier | None = None,
        sidecar_factory: Callable[
            [Callable[[], DockerContainerObservation | None]],
            EvaluatorSidecarAuthority,
        ] | None = None,
        runtime_loader: Callable[
            [Path, LmEvalExecutionReceipt], Mapping[str, object]
        ] = load_lm_eval_runtime_publication,
        inferencex_publisher: Callable[..., Mapping[str, object]] | None = None,
        poll_seconds: float = 0.1,
    ) -> None:
        if poll_seconds <= 0 or poll_seconds > 5:
            raise ValueError("poll_seconds must be in (0, 5]")
        self._receipt = dependency_receipt
        self._docker = docker or DockerCliObservationClient()
        self._gpu = gpu or RocmContainerGpuObserver()
        self._dependency_observer = dependency_observer or _dependency_snapshot
        self._evaluator_preparer = (
            evaluator_preparer or _default_evaluator_preparer(dependency_receipt)
        )
        self._handoff_barrier = handoff_barrier or EvaluatorHandoffBarrier()
        self._sidecar_factory = sidecar_factory or (
            lambda supplier: DockerEvaluatorSidecarAuthority(supplier)
        )
        self._runtime_loader = runtime_loader
        self._inferencex_publisher = (
            inferencex_publisher or publish_inferencex_runtime
        )
        self._poll_seconds = poll_seconds

    def formal_measurement_support(
        self, execution_mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport:
        if not self.supports(execution_mode, lifecycle):
            return docker_formal_measurement_support(execution_mode, lifecycle)
        if self._evaluator_preparer is None:
            return docker_formal_measurement_support(execution_mode, lifecycle)
        return MagpieFormalMeasurementSupport(
            True, None, "exact_image_sidecar", ()
        )

    def prepare(self, request: MagpieAttestationRequest) -> object:
        _validate_request(request)
        dependencies = self._dependency_observer(self._receipt)
        image = self._docker.resolve_image(str(request.requested_image))
        state = _ObservationState(
            request, image, dependencies, launch_argv=request.benchmark_argv
        )
        self._observe_once(state, initial=True)
        try:
            self._prepare_evaluator(state)
            state.thread = threading.Thread(
                target=self._observe_loop,
                args=(state,),
                name=f"apex-magpie-observer-{request.run_id}",
                daemon=True,
            )
            state.thread.start()
        except Exception:
            if state.handoff is not None:
                self._handoff_barrier.abort(
                    state.handoff, reason="magpie_observer_start_failed"
                )
            raise
        return state

    def _prepare_evaluator(self, state: _ObservationState) -> None:
        if (
            state.request.pass_type is not BenchmarkPass.MEASUREMENT
            or self._evaluator_preparer is None
        ):
            return
        prepared = self._evaluator_preparer.prepare(state.request)
        state.prepared = prepared
        state.launch_argv = project_docker_evaluator_launch_argv(
            state.request.benchmark_argv, prepared.launch_config_path
        )
        authority = self._sidecar_factory(lambda: _current_container(state))
        state.handoff = self._handoff_barrier.start(prepared, authority)

    def locate_report(self, session: object) -> MagpieReportLocation:
        if not isinstance(session, _ObservationState):
            raise ContractError("Magpie observer session is invalid", "docker_observer_failed")
        return locate_local_magpie_report(session.request.run_root)

    def launch_argv(self, session: object) -> tuple[str, ...]:
        if not isinstance(session, _ObservationState):
            raise ContractError(
                "Magpie observer session is invalid", "docker_observer_failed"
            )
        return session.launch_argv

    def abort(self, session: object, *, reason: str) -> None:
        if not isinstance(session, _ObservationState):
            raise ContractError(
                "Magpie observer session is invalid", "docker_observer_failed"
            )
        failures: list[str] = []
        if session.handoff is not None:
            try:
                self._handoff_barrier.abort(session.handoff, reason=reason)
            except Exception as error:
                failures.append(_error("evaluator_handoff_abort_failed", error))
        try:
            self._stop(session)
        except Exception as error:
            failures.append(_error("docker_observer_abort_failed", error))
        if session.thread is not None and session.thread.is_alive():
            failures.append("docker_observer_abort_incomplete")
        if failures:
            raise ContractError(";".join(failures), "docker_observer_abort_failed")

    def complete(
        self,
        session: object,
        *,
        report_path: Path | None,
        command_exit_code: int | None,
        timed_out: bool,
    ) -> Path | None:
        if not isinstance(session, _ObservationState):
            raise ContractError("Magpie observer session is invalid", "docker_observer_failed")
        self._complete_handoff(session)
        self._stop(session)
        self._complete_checks(session, report_path)
        self._complete_formal_artifacts(session, report_path)
        if report_path is None:
            return None
        report = _load_report(report_path, session.request.run_root)
        document = _attestation_document(
            session,
            report,
            report_path,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
        )
        path = expected_attestation_path(report_path)
        _write_new(path, document)
        return path

    def _complete_handoff(self, state: _ObservationState) -> None:
        if state.handoff is None:
            return
        try:
            completed = self._handoff_barrier.complete(state.handoff)
        except Exception as error:
            state.errors.append(_error("evaluator_handoff_failed", error))
            return
        state.completed_handoff = completed
        state.execution_receipt = completed.execution_receipt

    def _complete_formal_artifacts(
        self, state: _ObservationState, report_path: Path | None
    ) -> None:
        if state.prepared is None:
            return
        completed = state.completed_handoff
        if report_path is None or completed is None:
            state.errors.append("evaluator_formal_artifacts_missing")
            return
        try:
            verify_inferencex_projection(state.prepared.inferencex_projection)
            state.lm_eval_runtime_evidence = self._runtime_loader(
                state.prepared.authority_root, completed.execution_receipt
            )
            source = inferencex_dependency_entry(state.dependencies)
            state.inferencex_runtime_evidence = self._inferencex_publisher(
                state.prepared,
                report_path.parent.resolve(),
                source_root=Path(str(source["root"])),
                source_commit=str(source["commit"]),
                source_tree=str(source["tree"]),
                handoff_receipt_sha256=completed.handoff_receipt_sha256,
            )
        except Exception as error:
            state.errors.append(_error("evaluator_artifact_publication_failed", error))

    def _stop(self, state: _ObservationState) -> None:
        state.stop.set()
        if state.thread is None:
            raise ContractError("Magpie observer did not start", "docker_observer_failed")
        state.thread.join(timeout=max(2.0, self._poll_seconds * 4))
        if state.thread.is_alive():
            state.errors.append("docker_observer_thread_did_not_stop")

    def _complete_checks(
        self, state: _ObservationState, report_path: Path | None
    ) -> None:
        if sha256_file(state.request.config_path) != state.request.config_sha256:
            state.errors.append("benchmark_config_changed_during_execution")
        if state.prepared is not None:
            try:
                validate_prepared_docker_evaluator_inputs(state.prepared)
            except Exception as error:
                state.errors.append(_error("evaluator_prepared_input_drift", error))
        try:
            if self._dependency_observer(self._receipt) != state.dependencies:
                state.errors.append("dependency_changed_during_execution")
            image = self._docker.resolve_image(state.image.reference)
            if image != state.image:
                state.errors.append("docker_image_changed_during_execution")
        except Exception as error:
            state.errors.append(_error("docker_observer_completion_failed", error))
        observed = state.observed
        if observed is None:
            state.errors.append("magpie_container_not_observed")
        else:
            if (
                report_path is not None
                and observed.workspace_mount != report_path.parent.resolve()
            ):
                state.errors.append("magpie_report_container_mount_mismatch")
            try:
                remaining = self._docker.container_state(observed.container_id)
            except Exception as error:
                state.errors.append(_error("magpie_container_cleanup_unverified", error))
            else:
                if remaining is not None:
                    state.errors.append(f"magpie_container_not_removed:{remaining}")
        if state.gpu is None:
            state.errors.append("magpie_gpu_not_engaged")

    def _observe_loop(self, state: _ObservationState) -> None:
        while not state.stop.wait(self._poll_seconds):
            try:
                self._observe_once(state)
            except Exception as error:
                state.errors.append(_error("docker_observer_failed", error))
                state.stop.set()

    def _observe_once(self, state: _ObservationState, *, initial: bool = False) -> None:
        matches = _matching_containers(
            self._docker.running_containers(), state.request.run_root
        )
        if len(matches) > 1:
            raise ContractError("Multiple Magpie containers bind the run", "ambiguous_magpie_container")
        if initial and matches:
            raise ContractError("A Magpie container predates observer preparation", "stale_magpie_container")
        if not matches:
            return
        current = matches[0]
        _validate_container(current, state)
        with state.lock:
            if (
                state.observed is not None
                and state.observed.container_id != current.container_id
            ):
                raise ContractError(
                    "More than one container served a Magpie run",
                    "ambiguous_magpie_container",
                )
            state.observed = current
        assert state.request.gpu_lease is not None
        engagement = self._gpu.observe(current, state.request.gpu_lease)
        if engagement is not None:
            with state.lock:
                state.gpu = engagement


def _validate_request(request: MagpieAttestationRequest) -> None:
    if request.execution_mode != "docker":
        raise ContractError("Only Docker Magpie observation is available", "magpie_observer_mode_unavailable")
    if request.lifecycle != "one_shot":
        raise ContractError("Reusable Magpie lifecycle observation is unavailable", "magpie_observer_lifecycle_unavailable")
    if not isinstance(request.requested_image, str) or not request.requested_image:
        raise ContractError("Docker image is unresolved", "magpie_observer_image_unavailable")
    if request.gpu_lease is None:
        raise ContractError("GPU lease authority is missing", "magpie_gpu_lease_missing")
    if not request.run_root.is_absolute() or request.run_root.is_symlink():
        raise ContractError("Magpie run root is unsafe", "invalid_benchmark_output")
    if sha256_file(request.config_path) != request.config_sha256:
        raise ContractError("Benchmark config identity changed", "benchmark_config_changed_during_execution")


def _default_evaluator_preparer(
    receipt: DependencyReceipt,
) -> LmEvalExecutionPreparer | None:
    policy = receipt.evaluator_policy
    if policy is None or receipt.lm_eval_runtime is None:
        return None
    apex_root = policy.path.parent.parent
    dataset_root = (
        apex_root / ".cache" / "apex-evaluator-datasets"
        / policy.policy_id / policy.dataset_revision
    )
    preparer = LmEvalExecutionPreparer(receipt, dataset_root)
    try:
        preparer.verify_dataset()
    except ConfigurationError:
        return None
    return preparer


def _current_container(
    state: _ObservationState,
) -> DockerContainerObservation | None:
    with state.lock:
        return state.observed


def _matching_containers(
    containers: tuple[DockerContainerObservation, ...], run_root: Path
) -> tuple[DockerContainerObservation, ...]:
    root = run_root.resolve()
    matches = []
    for container in containers:
        mount = container.workspace_mount
        if not container.name.startswith("magpie-benchmark-") or mount is None:
            continue
        try:
            mount.relative_to(root)
        except ValueError:
            continue
        matches.append(container)
    return tuple(matches)


def _validate_container(
    container: DockerContainerObservation, state: _ObservationState
) -> None:
    inferencex = state.dependencies["dependencies"]
    assert isinstance(inferencex, Mapping)
    inferencex_entry = inferencex.get("inferencex")
    expected_root = (
        state.prepared.inferencex_projection.root
        if state.prepared is not None
        else Path(str(inferencex_entry["root"])).resolve()
        if isinstance(inferencex_entry, Mapping)
        else None
    )
    valid = (
        container.running
        and container.image_id == state.image.image_id
        and container.configured_image == state.image.reference
        and container.inferencex_mount == expected_root
        and container.kfd_exposed
        and container.dri_exposed
    )
    if not valid:
        raise ContractError("Magpie container differs from frozen inputs", "magpie_container_binding_mismatch")


def _dependency_snapshot(receipt: DependencyReceipt) -> Mapping[str, object]:
    dependencies: dict[str, object] = {}
    for name in ("magpie", "tracelens", "inferencex"):
        try:
            state = inspect_repository(receipt.root(name))
        except BootstrapError as error:
            raise ContractError("Dependency cannot be observed", "magpie_dependency_observation_failed") from error
        if state.commit != receipt.commits.get(name) or state.dirty_paths:
            raise ContractError("Dependency receipt drifted", "magpie_dependency_observation_failed")
        dependencies[name] = {
            "root": str(state.root),
            "commit": state.commit,
            "tree": state.tree,
        }
    return {"lock_sha256": receipt.lock_sha256, "dependencies": dependencies}


def _load_report(path: Path, run_root: Path) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(run_root.resolve())
    except ValueError as error:
        raise ContractError("Magpie report escapes run root", "benchmark_workspace_mismatch") from error
    if path.is_symlink() or path.stat().st_nlink != 1:
        raise ContractError("Magpie report is unsafe", "unsafe_benchmark_report")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("Magpie report is invalid", "invalid_benchmark_report") from error
    if not isinstance(value, Mapping):
        raise ContractError("Magpie report is invalid", "invalid_benchmark_report")
    return value


def _attestation_document(
    state: _ObservationState,
    report: Mapping[str, Any],
    report_path: Path,
    *,
    command_exit_code: int | None,
    timed_out: bool,
) -> Mapping[str, object]:
    process_ok = command_exit_code == 0 and not timed_out
    lane = report.get("profiling_enabled") is (
        state.request.pass_type is BenchmarkPass.DIAGNOSTIC
    )
    errors = list(dict.fromkeys(state.errors))
    verified = not errors and process_ok and lane
    runtime_receipt = _serving_receipt(state, report, process_ok, verified)
    quality_receipt = (
        None if state.request.pass_type is BenchmarkPass.DIAGNOSTIC
        else build_lm_eval_quality_gate(
            report_path.parent, execution_receipt=state.execution_receipt
        )
    )
    quality_ok = (
        state.request.pass_type is BenchmarkPass.DIAGNOSTIC
        or quality_receipt is not None
    )
    lm_eval = (
        _not_requested_lm_eval(state)
        if state.request.pass_type is BenchmarkPass.DIAGNOSTIC
        else state.lm_eval_runtime_evidence
    )
    formal_runtime = (
        state.request.pass_type is BenchmarkPass.DIAGNOSTIC
        or (
            state.execution_receipt is not None
            and state.inferencex_runtime_evidence is not None
            and lm_eval is not None
        )
    )
    runtime_ok = bool(
        runtime_receipt
        and runtime_receipt["verified"] is True
        and formal_runtime
    )
    return {
        "schema": EXECUTION_ATTESTATION_SCHEMA,
        "authority": "apex_evaluator",
        "official_report_path": report_path.relative_to(state.request.run_root).as_posix(),
        "official_report_size_bytes": report_path.stat().st_size,
        "report_sha256": sha256_file(report_path),
        "config_sha256": state.request.config_sha256,
        "run_id": state.request.run_id,
        "pass_type": state.request.pass_type.value,
        "lane_verified": lane,
        "reward_eligible": bool(
            state.request.pass_type is BenchmarkPass.MEASUREMENT
            and verified
            and quality_ok
        ),
        "profiling_enabled": state.request.pass_type is BenchmarkPass.DIAGNOSTIC,
        "process": _process_receipt(state, command_exit_code, timed_out),
        "dependencies": {"schema": "apex.magpie-dependency-attestation/v1", "verified": not any("dependency" in item for item in errors), "receipts": state.dependencies},
        "runtime": {"schema": "apex.magpie-runtime-attestation/v1", "verified": runtime_ok, "model_revision_receipt": None, "inferencex_runtime_receipt": state.inferencex_runtime_evidence, "lm_eval_runtime_receipt": lm_eval, "serving_runtime_receipt": runtime_receipt},
        "gpu_engagement": {"schema": "apex.magpie-gpu-engagement/v1", "verified": state.gpu is not None, "devices": list(state.gpu.get("devices", [])) if state.gpu else [], "processes": list(state.gpu.get("processes", [])) if state.gpu else []},
        "quality_gate": {"schema": "apex.magpie-quality-attestation/v1", "verified": quality_ok, "receipt": quality_receipt},
        "errors": errors,
    }


def _process_receipt(state, exit_code, timed_out) -> Mapping[str, object]:
    return {
        "schema": "apex.magpie-process-attestation/v1",
        "argv_sha256": sha256_json(list(state.launch_argv)),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "succeeded": exit_code == 0 and not timed_out,
        "verified": not any(item.startswith("docker_observer") for item in state.errors),
    }


def _not_requested_lm_eval(state: _ObservationState) -> Mapping[str, object] | None:
    if state.request.pass_type is not BenchmarkPass.DIAGNOSTIC:
        return None
    return {"schema": "magpie.lm-eval-runtime-evidence/v1", "requested": False, "status": "not_requested", "verified": False, "evidence_present": False, "runtime_sha256": None, "identity": None, "mount_mode": None, "manifest_artifact": None, "receipt_artifact": None, "errors": []}


def _serving_receipt(
    state: _ObservationState,
    report: Mapping[str, Any],
    process_succeeded: bool,
    verified: bool,
) -> Mapping[str, object] | None:
    container = state.observed
    if container is None:
        return None
    nulls = {name: None for name in ("runtime_schema", "tracelens_source_commit", "tracelens_source_tree", "patch_version", "patch_path", "patch_sha256", "dependency_wheel_manifest_sha256")}
    return {
        "schema": "apex.magpie-serving-runtime-observation/v3", "execution_mode": "docker",
        "input_config_sha256": state.request.config_sha256,
        "input_image": state.image.reference, "input_image_id": state.image.image_id,
        "requested_image": state.image.reference, "resolved_image_id": container.image_id,
        "image_derivation": {"kind": "direct", "framework": report.get("framework"), "base_image": state.image.reference, "base_image_id": state.image.image_id, "base_image_locator": state.image.reference, "derived_image": state.image.reference, "derived_image_id": container.image_id, **nulls, "validator": "docker-image-id", "verified": True},
        "container_name": container.name,
        "container_spec_sha256": container.container_spec_sha256,
        "process_succeeded": process_succeeded, "verified": verified, "errors": [],
    }


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, canonical_json_bytes(value) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _error(prefix: str, error: Exception) -> str:
    reason = getattr(error, "reason_code", type(error).__name__)
    return f"{prefix}:{reason}"


__all__ = ["DockerOneShotMagpieExecutionAttestor"]
