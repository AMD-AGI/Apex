"""Authority-backed execution attestation for one-shot Magpie Ray benchmarks."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from apex.core import ContractError, sha256_file, sha256_json
from apex.ports import (
    BenchmarkPass,
    MagpieAttestationRequest,
    MagpieReportLocation,
    RayNodeEvidenceAuthority,
    RayNodeEvidenceBinding,
    RayNodeEvidenceReceipt,
)
from apex.runtime import DependencyReceipt
from apex.runtime.magpie_result_contract import EXECUTION_ATTESTATION_SCHEMA

from .docker_magpie_attestor import _error, _load_report, _write_new
from .formal_measurement_support import ray_formal_measurement_support
from .magpie_attestation import expected_attestation_path
from .quality import build_lm_eval_quality_gate
from .ray_artifacts import (
    RayImportedWorkspace,
    RaySharedArtifactImporter,
    validate_remote_dependencies,
)
from .ray_observation import (
    LocalRayDriverProcessObserver,
    RayCliObservationClientFactory,
    RayDriverProcessObservation,
    RayDriverProcessObserver,
    RayJobObservation,
    RayJobSnapshot,
    RayObservationClient,
    RayObservationClientFactory,
    RayTaskObservation,
    RayTaskSnapshot,
)


class UnavailableRayNodeEvidenceAuthority:
    """Production default: a Ray worker may never attest itself."""

    is_available = False

    def prepare(self, request, *, ray_contract, cluster_identity_sha256):
        del request, ray_contract, cluster_identity_sha256
        raise ContractError("Ray node authority is unavailable", "ray_node_authority_unavailable")

    def complete(self, session, *, binding):
        del session, binding
        raise ContractError("Ray node authority is unavailable", "ray_node_authority_unavailable")

    def abort(self, session, *, reason: str) -> None:
        del session, reason


@dataclass(slots=True)
class _RayObservationState:
    request: MagpieAttestationRequest
    ray: RayObservationClient
    authority_session: object
    baseline: RayTaskSnapshot
    baseline_jobs: RayJobSnapshot
    baseline_keys: frozenset[tuple[str, int]]
    baseline_job_ids: frozenset[str]
    stop: threading.Event = field(default_factory=threading.Event)
    job: RayJobObservation | None = None
    final_job: RayJobObservation | None = None
    driver: RayDriverProcessObservation | None = None
    task: RayTaskObservation | None = None
    final_task: RayTaskObservation | None = None
    node_evidence: RayNodeEvidenceReceipt | None = None
    imported: RayImportedWorkspace | None = None
    errors: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    finalized: bool = False
    authority_closed: bool = False


class RayOneShotMagpieExecutionAttestor:
    """Bind Ray state to independent node evidence and immutable shared files."""

    @staticmethod
    def supports(execution_mode: str, lifecycle: str) -> bool:
        return execution_mode == "ray" and lifecycle == "one_shot"

    formal_measurement_support = staticmethod(ray_formal_measurement_support)

    def __init__(
        self,
        dependency_receipt: DependencyReceipt,
        *,
        ray_factory: RayObservationClientFactory | None = None,
        node_authority: RayNodeEvidenceAuthority | None = None,
        driver: RayDriverProcessObserver | None = None,
        importer: RaySharedArtifactImporter | None = None,
        poll_seconds: float = 0.1,
    ) -> None:
        if poll_seconds <= 0 or poll_seconds > 5:
            raise ValueError("poll_seconds must be in (0, 5]")
        self._receipt = dependency_receipt
        self._ray_factory = ray_factory or RayCliObservationClientFactory()
        self._authority = node_authority or UnavailableRayNodeEvidenceAuthority()
        self._driver = driver or LocalRayDriverProcessObserver()
        self._importer = importer or RaySharedArtifactImporter()
        self._poll_seconds = poll_seconds

    @property
    def is_available(self) -> bool:
        return self._ray_factory.is_available and self._authority.is_available

    def prepare(self, request: MagpieAttestationRequest) -> object:
        _validate_request(request)
        if not self.is_available:
            raise ContractError("Ray node authority is unavailable", "ray_node_authority_unavailable")
        assert request.ray_contract is not None
        ray = self._ray_factory.create(request.ray_contract.cluster_address)
        baseline_jobs = ray.jobs()
        baseline = ray.tasks()
        _require_cluster(request, baseline_jobs, baseline)
        authority_session = self._authority.prepare(
            request,
            ray_contract=request.ray_contract,
            cluster_identity_sha256=baseline.identity_digest,
        )
        state = _RayObservationState(
            request,
            ray,
            authority_session,
            baseline,
            baseline_jobs,
            frozenset(task.key for task in baseline.tasks),
            frozenset(job.job_id for job in baseline_jobs.jobs),
        )
        state.thread = threading.Thread(
            target=self._observe_loop,
            args=(state,),
            name=f"apex-magpie-ray-observer-{request.run_id}",
            daemon=True,
        )
        state.thread.start()
        return state

    def launch_argv(self, session: object) -> tuple[str, ...]:
        if not isinstance(session, _RayObservationState):
            raise ContractError(
                "Ray Magpie observer session is invalid", "ray_observer_failed"
            )
        return session.request.benchmark_argv

    def abort(self, session: object, *, reason: str) -> None:
        state = _state(session)
        try:
            self._stop_observation(state)
        finally:
            state.finalized = True
            self._abort_authority(state, reason)

    def locate_report(self, session: object) -> MagpieReportLocation:
        state = _state(session)
        self._finalize_observation(state)
        if state.errors:
            self._abort_authority(state, state.errors[0])
            return MagpieReportLocation(None, state.errors[0])
        binding = _node_binding(state)
        try:
            evidence = self._authority.complete(
                state.authority_session, binding=binding
            )
            state.authority_closed = True
            validate_remote_dependencies(evidence.dependencies, self._receipt)
            imported = self._importer.import_workspace(
                state.request, binding, evidence
            )
        except Exception as error:
            state.errors.append(_error("ray_node_evidence_failed", error))
            self._abort_authority(state, state.errors[-1])
            return MagpieReportLocation(None, state.errors[-1])
        state.node_evidence, state.imported = evidence, imported
        return MagpieReportLocation(imported.report_path)

    def complete(
        self,
        session: object,
        *,
        report_path: Path | None,
        command_exit_code: int | None,
        timed_out: bool,
    ) -> Path | None:
        state = _state(session)
        if not state.finalized:
            self._finalize_observation(state)
        if report_path is None or state.imported is None or state.node_evidence is None:
            self._abort_authority(state, "ray_report_unavailable")
            return None
        report = _load_report(report_path, state.request.run_root)
        document = _attestation_document(
            state,
            report,
            report_path,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
        )
        path = expected_attestation_path(report_path)
        _write_new(path, document)
        return path

    def _finalize_observation(self, state: _RayObservationState) -> None:
        if state.finalized:
            return
        self._stop_observation(state)
        self._complete_checks(state)
        state.finalized = True

    def _stop_observation(self, state: _RayObservationState) -> None:
        state.stop.set()
        if state.thread is None:
            raise ContractError("Ray observer did not start", "ray_observer_failed")
        state.thread.join(timeout=max(2.0, self._poll_seconds * 4))
        if state.thread.is_alive():
            state.errors.append("ray_observer_thread_did_not_stop")

    def _complete_checks(self, state: _RayObservationState) -> None:
        if sha256_file(state.request.config_path) != state.request.config_sha256:
            state.errors.append("benchmark_config_changed_during_execution")
        if state.job is None or state.driver is None or state.task is None:
            state.errors.append("magpie_ray_task_not_observed")
            return
        try:
            jobs, tasks = state.ray.jobs(), state.ray.tasks()
            _require_cluster(state.request, state.baseline_jobs, jobs)
            _require_cluster(state.request, state.baseline, tasks)
            self._bind_terminal_state(state, jobs, tasks)
        except Exception as error:
            state.errors.append(_error("ray_observer_completion_failed", error))

    @staticmethod
    def _bind_terminal_state(
        state: _RayObservationState,
        jobs: RayJobSnapshot,
        tasks: RayTaskSnapshot,
    ) -> None:
        assert state.job is not None and state.task is not None
        job_matches = tuple(job for job in jobs.jobs if job.job_id == state.job.job_id)
        task_matches = tuple(task for task in tasks.tasks if task.key == state.task.key)
        if len(job_matches) != 1 or job_matches[0].driver_pid != state.job.driver_pid:
            raise ContractError("Ray job identity changed", "magpie_ray_job_identity_changed")
        if job_matches[0].is_dead is not True:
            raise ContractError("Ray job did not finish", "magpie_ray_job_not_finished")
        if len(task_matches) != 1 or task_matches[0].identity != state.task.identity:
            raise ContractError("Ray task identity changed", "magpie_ray_task_identity_changed")
        if task_matches[0].state != "FINISHED":
            raise ContractError("Ray task did not finish", "magpie_ray_task_not_finished")
        state.final_job, state.final_task = job_matches[0], task_matches[0]

    def _observe_loop(self, state: _RayObservationState) -> None:
        while not state.stop.wait(self._poll_seconds):
            try:
                self._observe_once(state)
            except Exception as error:
                state.errors.append(_error("ray_observer_failed", error))
                state.stop.set()

    def _observe_once(self, state: _RayObservationState) -> None:
        jobs, tasks = state.ray.jobs(), state.ray.tasks()
        _require_cluster(state.request, state.baseline_jobs, jobs)
        _require_cluster(state.request, state.baseline, tasks)
        match = self._matching_job(state, jobs)
        if match is None:
            return
        job, driver = match
        if state.job is not None and job != state.job:
            raise ContractError("Ray job identity changed", "magpie_ray_job_identity_changed")
        if state.driver is not None and driver != state.driver:
            raise ContractError("Ray driver identity changed", "magpie_ray_driver_changed")
        state.job, state.driver = job, driver
        current = _new_task(state, tasks, job)
        if current is None:
            return
        if state.task is not None and current.identity != state.task.identity:
            raise ContractError("Ray task identity changed", "magpie_ray_task_identity_changed")
        state.task = current

    def _matching_job(
        self, state: _RayObservationState, snapshot: RayJobSnapshot
    ) -> tuple[RayJobObservation, RayDriverProcessObservation] | None:
        matches = []
        for job in snapshot.jobs:
            if job.job_id in state.baseline_job_ids or job.is_dead:
                continue
            try:
                driver = self._driver.freeze(job, state.request.benchmark_argv)
            except ContractError as error:
                if error.reason_code in {
                    "ray_driver_process_mismatch",
                    "ray_worker_process_unavailable",
                }:
                    continue
                raise
            matches.append((job, driver))
        if len(matches) > 1:
            raise ContractError("Multiple Ray jobs match", "ambiguous_magpie_ray_job")
        return matches[0] if matches else None

    def _abort_authority(self, state: _RayObservationState, reason: str) -> None:
        if state.authority_closed:
            return
        try:
            self._authority.abort(state.authority_session, reason=reason)
        finally:
            state.authority_closed = True


def _state(session: object) -> _RayObservationState:
    if not isinstance(session, _RayObservationState):
        raise ContractError("Ray observer session is invalid", "ray_observer_failed")
    return session


def _validate_request(request: MagpieAttestationRequest) -> None:
    if request.execution_mode != "ray":
        raise ContractError("Only Ray observation is available", "magpie_observer_mode_unavailable")
    if request.lifecycle != "one_shot":
        raise ContractError("Ray lifecycle is unavailable", "magpie_observer_lifecycle_unavailable")
    if request.requested_image is not None:
        raise ContractError("Ray cannot claim a Docker image", "magpie_ray_image_invalid")
    if request.gpu_lease is None:
        raise ContractError("GPU lease authority is missing", "magpie_gpu_lease_missing")
    if request.ray_contract is None:
        raise ContractError("Resolved Ray contract is missing", "invalid_magpie_ray_config")
    if not request.run_root.is_absolute() or request.run_root.is_symlink():
        raise ContractError("Magpie run root is unsafe", "invalid_benchmark_output")
    if sha256_file(request.config_path) != request.config_sha256:
        raise ContractError("Benchmark config changed", "benchmark_config_changed_during_execution")


def _require_cluster(
    request: MagpieAttestationRequest,
    first: RayTaskSnapshot | RayJobSnapshot,
    second: RayTaskSnapshot | RayJobSnapshot,
) -> None:
    assert request.ray_contract is not None
    if (
        first.identity_digest != second.identity_digest
        or first.address_sha256 != request.ray_contract.address_sha256
        or second.address_sha256 != request.ray_contract.address_sha256
    ):
        raise ContractError("Ray cluster identity changed", "ray_cluster_identity_changed")


def _new_task(
    state: _RayObservationState,
    snapshot: RayTaskSnapshot,
    job: RayJobObservation,
) -> RayTaskObservation | None:
    matches = tuple(
        task
        for task in snapshot.tasks
        if task.key not in state.baseline_keys and task.job_id == job.job_id
    )
    if len(matches) > 1:
        raise ContractError("Multiple Ray tasks match", "ambiguous_magpie_ray_task")
    return matches[0] if matches else None


def _node_binding(state: _RayObservationState) -> RayNodeEvidenceBinding:
    if state.job is None or state.driver is None or state.task is None:
        raise ContractError("Ray task identity is incomplete", "ray_node_evidence_invalid")
    assert state.request.ray_contract is not None
    task = state.final_task or state.task
    return RayNodeEvidenceBinding(
        state.request.run_id,
        state.request.pass_type,
        state.request.config_sha256,
        sha256_json(list(state.request.benchmark_argv)),
        sha256_json(state.request.gpu_lease),
        state.request.ray_contract,
        state.baseline.identity_digest,
        (state.final_job or state.job).to_dict(),
        state.driver.to_dict(),
        task.to_dict(),
    )


def _attestation_document(
    state: _RayObservationState,
    report: Mapping[str, object],
    report_path: Path,
    *,
    command_exit_code: int | None,
    timed_out: bool,
) -> Mapping[str, object]:
    assert state.node_evidence is not None and state.imported is not None
    process_ok = command_exit_code == 0 and not timed_out
    lane = report.get("profiling_enabled") is (
        state.request.pass_type is BenchmarkPass.DIAGNOSTIC
    )
    errors = list(dict.fromkeys(state.errors))
    runtime_receipt = _ray_runtime_receipt(state, process_ok, not errors and lane)
    quality_receipt = (
        None
        if state.request.pass_type is BenchmarkPass.DIAGNOSTIC
        else build_lm_eval_quality_gate(report_path.parent, execution_receipt=None)
    )
    quality_ok = state.request.pass_type is BenchmarkPass.DIAGNOSTIC or quality_receipt is not None
    verified = not errors and process_ok and lane and runtime_receipt["verified"] is True
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
            state.request.pass_type is BenchmarkPass.MEASUREMENT and verified and quality_ok
        ),
        "profiling_enabled": state.request.pass_type is BenchmarkPass.DIAGNOSTIC,
        "process": _process_receipt(state, command_exit_code, timed_out),
        "dependencies": {
            "schema": "apex.magpie-dependency-attestation/v1",
            "verified": verified,
            "receipts": state.node_evidence.dependencies,
        },
        "runtime": {
            "schema": "apex.magpie-runtime-attestation/v1",
            "verified": runtime_receipt["verified"] is True,
            "model_revision_receipt": state.node_evidence.runtime["model_revision_receipt"],
            "inferencex_runtime_receipt": state.node_evidence.runtime["inferencex_runtime_receipt"],
            "lm_eval_runtime_receipt": state.node_evidence.runtime["lm_eval_runtime_receipt"],
            "serving_runtime_receipt": runtime_receipt,
        },
        "gpu_engagement": {
            "schema": "apex.magpie-gpu-engagement/v1",
            "verified": verified,
            "devices": list(state.node_evidence.gpu_devices),
            "processes": list(state.node_evidence.gpu_processes),
        },
        "quality_gate": {
            "schema": "apex.magpie-quality-attestation/v1",
            "verified": quality_ok,
            "receipt": quality_receipt,
        },
        "errors": errors,
    }


def _process_receipt(state, exit_code, timed_out) -> Mapping[str, object]:
    return {
        "schema": "apex.magpie-process-attestation/v1",
        "argv_sha256": sha256_json(list(state.request.benchmark_argv)),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "succeeded": exit_code == 0 and not timed_out,
        "verified": not state.errors and state.final_task is not None,
    }


def _ray_runtime_receipt(
    state: _RayObservationState, process_succeeded: bool, verified: bool
) -> Mapping[str, object]:
    assert state.node_evidence is not None and state.imported is not None
    binding = _node_binding(state)
    return {
        "schema": "apex.magpie-ray-runtime-observation/v2",
        "execution_mode": "ray",
        "input_config_sha256": state.request.config_sha256,
        "ray_config_sha256": binding.ray_contract.ray_config_sha256,
        "ray_address_sha256": binding.ray_contract.address_sha256,
        "gpu_lease_sha256": binding.gpu_lease_sha256,
        "cluster_identity_sha256": state.baseline.identity_digest,
        "node_authority_sha256": state.node_evidence.authority_sha256,
        "node_evidence_binding_sha256": binding.digest,
        "magpie_task_id": state.node_evidence.magpie_task_id,
        "artifact_import": state.imported.to_dict(),
        "job": dict(binding.job),
        "driver_process": dict(binding.driver_process),
        "task": dict(binding.task),
        "node_receipts": [dict(value) for value in state.node_evidence.node_receipts],
        "process_succeeded": process_succeeded,
        "verified": bool(verified and state.node_evidence.runtime["verified"] is True),
        "errors": list(dict.fromkeys(state.errors)),
    }


__all__ = [
    "RayOneShotMagpieExecutionAttestor",
    "UnavailableRayNodeEvidenceAuthority",
]
