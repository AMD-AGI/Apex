"""Apex-owned observer for local Magpie one-shot and server lifecycles."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from apex.core import ContractError, sha256_file, sha256_json
from apex.ports import (
    BenchmarkPass,
    MagpieAttestationRequest,
    MagpieReportLocation,
)
from apex.runtime import DependencyReceipt
from apex.runtime.magpie_result_contract import EXECUTION_ATTESTATION_SCHEMA

from .local_gpu_observation import (
    LocalGpuEngagementObserver,
    LocalGpuLeaseAuthority,
    RocmLocalGpuEngagementObserver,
    validate_active_local_gpu_lease,
)
from .formal_measurement_support import local_formal_measurement_support
from .local_port_observation import (
    LocalPortObservationClient,
    ProcfsLocalPortObservationClient,
)
from .local_magpie_contract import (
    LocalMagpieContract,
    MAX_CONFIG_BYTES,
    MAX_LIFECYCLE_BYTES,
    dependency_snapshot,
    json_mapping,
    load_local_contract,
    read_regular,
    validate_local_request,
    write_new,
)
from .local_process_observation import (
    LocalProcessIdentity,
    LocalProcessObservationClient,
    ProcfsLocalProcessObservationClient,
    belongs_to_root,
    descendant_closure,
    matching_processes,
    same_process,
)
from .local_runtime_receipt import build_local_runtime_receipt
from .magpie_attestation import expected_attestation_path, locate_local_magpie_report
from .quality import build_lm_eval_quality_gate


@dataclass(frozen=True, slots=True)
class _ServerState:
    process: LocalProcessIdentity
    listener_pids: tuple[int, ...]
    compatibility_sha256: str


@dataclass(slots=True)
class _ObservationState:
    request: MagpieAttestationRequest
    dependencies: Mapping[str, object]
    contract: LocalMagpieContract
    lease: LocalGpuLeaseAuthority
    initial_server: _ServerState | None
    stop: threading.Event = field(default_factory=threading.Event)
    benchmark_process: LocalProcessIdentity | None = None
    server_process: LocalProcessIdentity | None = None
    final_server: _ServerState | None = None
    gpu: Mapping[str, object] | None = None
    quiescence: Mapping[str, object] | None = None
    listener_pids: tuple[int, ...] = ()
    observed: dict[tuple[int, int], LocalProcessIdentity] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None


class LocalMagpieExecutionAttestor:
    """Observe local Magpie without importing, patching, or signaling Magpie."""

    is_available = True

    @staticmethod
    def supports(execution_mode: str, lifecycle: str) -> bool:
        return execution_mode == "local" and lifecycle in {"one_shot", "reuse", "cleanup"}

    formal_measurement_support = staticmethod(local_formal_measurement_support)

    def __init__(
        self,
        dependency_receipt: DependencyReceipt,
        *,
        processes: LocalProcessObservationClient | None = None,
        ports: LocalPortObservationClient | None = None,
        gpu: LocalGpuEngagementObserver | None = None,
        dependency_observer: Callable[[DependencyReceipt], Mapping[str, object]] | None = None,
        lease_validator: Callable[[Mapping[str, object], str], LocalGpuLeaseAuthority] | None = None,
        poll_seconds: float = 0.1,
    ) -> None:
        if poll_seconds <= 0 or poll_seconds > 5:
            raise ValueError("poll_seconds must be in (0, 5]")
        self._receipt = dependency_receipt
        self._processes = processes or ProcfsLocalProcessObservationClient()
        self._ports = ports or ProcfsLocalPortObservationClient()
        self._gpu = gpu or RocmLocalGpuEngagementObserver(self._processes)
        self._dependency_observer = dependency_observer or dependency_snapshot
        self._lease_validator = lease_validator or self._validate_lease
        self._poll_seconds = poll_seconds

    def prepare(self, request: MagpieAttestationRequest) -> object:
        validate_local_request(request)
        dependencies = self._dependency_observer(self._receipt)
        contract = load_local_contract(request, self._receipt, dependencies)
        assert request.gpu_lease is not None
        lease = self._lease_validator(request.gpu_lease, request.run_id)
        initial = self._processes.snapshot()
        matches = matching_processes(
            initial,
            argv=request.benchmark_argv,
            cwd=self._receipt.root("magpie"),
        )
        if matches:
            raise ContractError(
                "A matching Magpie process predates observer preparation",
                "stale_magpie_local_process",
            )
        initial_server = self._server_state(contract, lease, required=False)
        if contract.lifecycle == "cleanup" and initial_server is None:
            raise ContractError(
                "Cleanup requires one previously attested persistent server",
                "magpie_local_cleanup_target_missing",
            )
        state = _ObservationState(request, dependencies, contract, lease, initial_server)
        if initial_server is not None:
            state.server_process = initial_server.process
            state.observed[_key(initial_server.process)] = initial_server.process
        state.thread = threading.Thread(
            target=self._observe_loop,
            args=(state,),
            name=f"apex-magpie-local-observer-{request.run_id}",
            daemon=True,
        )
        state.thread.start()
        return state

    def launch_argv(self, session: object) -> tuple[str, ...]:
        if not isinstance(session, _ObservationState):
            raise ContractError(
                "Local Magpie observer session is invalid",
                "magpie_local_observer_failed",
            )
        return session.request.benchmark_argv

    def abort(self, session: object, *, reason: str) -> None:
        del reason
        if not isinstance(session, _ObservationState):
            raise ContractError("Local observer session is invalid", "local_observer_failed")
        self._stop(session)

    def locate_report(self, session: object) -> MagpieReportLocation:
        if not isinstance(session, _ObservationState):
            raise ContractError("Local observer session is invalid", "local_observer_failed")
        return locate_local_magpie_report(session.request.run_root)

    def complete(
        self,
        session: object,
        *,
        report_path: Path | None,
        command_exit_code: int | None,
        timed_out: bool,
    ) -> Path | None:
        if not isinstance(session, _ObservationState):
            raise ContractError(
                "Local Magpie observer session is invalid",
                "magpie_local_observer_failed",
            )
        self._stop(session)
        self._complete_checks(session)
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
        write_new(path, document)
        return path

    def _validate_lease(
        self, value: Mapping[str, object], run_id: str
    ) -> LocalGpuLeaseAuthority:
        return validate_active_local_gpu_lease(
            value, run_id=run_id, processes=self._processes
        )

    def _stop(self, state: _ObservationState) -> None:
        state.stop.set()
        if state.thread is None:
            raise ContractError(
                "Local Magpie observer did not start", "magpie_local_observer_failed"
            )
        state.thread.join(timeout=max(2.0, self._poll_seconds * 4))
        if state.thread.is_alive():
            state.errors.append("magpie_local_observer_thread_did_not_stop")

    def _complete_checks(self, state: _ObservationState) -> None:
        self._stable_inputs(state)
        if state.benchmark_process is None:
            state.errors.append("magpie_local_process_not_observed")
        self._residual_checks(state)
        if state.contract.lifecycle == "reuse":
            self._complete_reuse(state)
        elif state.contract.lifecycle == "cleanup":
            self._complete_cleanup(state)
        else:
            self._complete_one_shot(state)
        if state.gpu is None:
            state.errors.append("magpie_local_gpu_not_engaged")
        if not state.listener_pids:
            state.errors.append("magpie_local_server_listener_not_observed")

    def _stable_inputs(self, state: _ObservationState) -> None:
        if sha256_file(state.request.config_path) != state.request.config_sha256:
            state.errors.append("benchmark_config_changed_during_execution")
        try:
            if self._dependency_observer(self._receipt) != state.dependencies:
                state.errors.append("dependency_changed_during_execution")
        except Exception as error:
            state.errors.append(_error("magpie_local_dependency_check_failed", error))
        try:
            assert state.request.gpu_lease is not None
            current = self._lease_validator(
                state.request.gpu_lease, state.request.run_id
            )
            if (
                current.run_id != state.lease.run_id
                or current.lease_digest != state.lease.lease_digest
                or current.selector_scope != state.lease.selector_scope
                or current.devices != state.lease.devices
                or current.owner != state.lease.owner
            ):
                state.errors.append("magpie_local_gpu_lease_changed")
            else:
                state.lease = current
        except Exception as error:
            state.errors.append(_error("magpie_local_gpu_lease_completion_failed", error))

    def _residual_checks(self, state: _ObservationState) -> None:
        for identity in state.observed.values():
            if (
                state.contract.lifecycle == "reuse"
                and state.server_process is not None
                and belongs_to_root(identity, state.observed, state.server_process)
            ):
                continue
            try:
                current = self._processes.process(identity.pid)
            except Exception as error:
                state.errors.append(_error("magpie_local_residual_check_failed", error))
                continue
            if same_process(identity, current):
                state.errors.append(f"magpie_local_residual_process:{identity.pid}")

    def _complete_reuse(self, state: _ObservationState) -> None:
        try:
            final = self._server_state(state.contract, state.lease, required=True)
        except Exception as error:
            state.errors.append(_error("magpie_local_reuse_unverified", error))
            return
        assert final is not None
        if state.initial_server is not None and final.process != state.initial_server.process:
            state.errors.append("magpie_local_reuse_server_identity_changed")
        state.final_server = final
        state.server_process = final.process

    def _complete_cleanup(self, state: _ObservationState) -> None:
        assert state.contract.pid_file is not None and state.contract.meta_file is not None
        if _exists(state.contract.pid_file) or _exists(state.contract.meta_file):
            state.errors.append("magpie_local_cleanup_state_remains")
        self._require_quiescent(state)

    def _complete_one_shot(self, state: _ObservationState) -> None:
        self._require_quiescent(state)

    def _require_quiescent(self, state: _ObservationState) -> None:
        try:
            state.quiescence = self._gpu.require_quiescent(state.lease)
        except Exception as error:
            state.errors.append(_error("magpie_local_cleanup_unverified", error))

    def _observe_loop(self, state: _ObservationState) -> None:
        while not state.stop.wait(self._poll_seconds):
            try:
                self._observe_once(state)
            except Exception as error:
                state.errors.append(_error("magpie_local_observer_failed", error))
                state.stop.set()

    def _observe_once(self, state: _ObservationState) -> None:
        processes = self._processes.snapshot()
        matches = matching_processes(
            processes,
            argv=state.request.benchmark_argv,
            cwd=self._receipt.root("magpie"),
        )
        if len(matches) > 1:
            raise ContractError(
                "Multiple exact local Magpie processes were observed",
                "ambiguous_magpie_local_process",
            )
        if matches:
            self._bind_benchmark_process(state, matches[0])
        server = self._server_pid(state.contract, state.lease)
        if server is not None:
            if state.server_process is not None and server != state.server_process:
                raise ContractError(
                    "Persistent server identity changed during the run",
                    "magpie_local_server_identity_changed",
                )
            state.server_process = server
        roots = tuple(
            item for item in (state.benchmark_process, state.server_process)
            if item is not None
        )
        if not roots:
            return
        closure = descendant_closure(processes, roots)
        for item in closure:
            if not _same_cgroup(item, state.lease.owner):
                raise ContractError(
                    "A local Magpie descendant escaped the lease-owner cgroup",
                    "magpie_local_process_cgroup_escape",
                )
            state.observed[_key(item)] = item
        listeners = self._ports.listener_owners(state.contract.port, closure)
        if listeners:
            state.listener_pids = tuple(sorted(item.pid for item in listeners))
        engagement = self._gpu.observe(roots, state.lease)
        if engagement is not None:
            state.gpu = engagement

    def _bind_benchmark_process(
        self, state: _ObservationState, process: LocalProcessIdentity
    ) -> None:
        valid = (
            process.uid == state.lease.owner.uid
            and process.ppid == state.lease.owner.pid
            and process.process_group == process.pid
            and process.session_id == process.pid
            and _same_cgroup(process, state.lease.owner)
        )
        if not valid:
            raise ContractError(
                "Local Magpie process is outside the exact Apex session",
                "magpie_local_process_binding_mismatch",
            )
        if state.benchmark_process is not None and process != state.benchmark_process:
            raise ContractError(
                "More than one local Magpie process served the run",
                "ambiguous_magpie_local_process",
            )
        state.benchmark_process = process
        state.observed[_key(process)] = process

    def _server_pid(
        self, contract: LocalMagpieContract, lease: LocalGpuLeaseAuthority
    ) -> LocalProcessIdentity | None:
        if contract.pid_file is None or not _exists(contract.pid_file):
            return None
        raw = read_regular(contract.pid_file, MAX_LIFECYCLE_BYTES)
        try:
            pid = int(raw.decode("ascii").strip())
        except (UnicodeError, ValueError) as error:
            raise ContractError(
                "Persistent server PID file is invalid", "magpie_local_server_state_invalid"
            ) from error
        process = self._processes.process(pid)
        if process is None or not _same_cgroup(process, lease.owner):
            raise ContractError(
                "Persistent server process identity is unavailable",
                "magpie_local_server_state_invalid",
            )
        return process

    def _server_state(
        self,
        contract: LocalMagpieContract,
        lease: LocalGpuLeaseAuthority,
        *,
        required: bool,
    ) -> _ServerState | None:
        if contract.pid_file is None or contract.meta_file is None:
            return None
        if not _exists(contract.pid_file) and not _exists(contract.meta_file):
            if required:
                raise ContractError(
                    "Persistent server state is missing", "magpie_local_server_state_missing"
                )
            return None
        if not _exists(contract.pid_file) or not _exists(contract.meta_file):
            if required:
                raise ContractError(
                    "Persistent server state is partial", "magpie_local_server_state_invalid"
                )
            return None
        process = self._server_pid(contract, lease)
        assert process is not None
        metadata_raw = read_regular(contract.meta_file, MAX_LIFECYCLE_BYTES)
        metadata = json_mapping(metadata_raw, "magpie_local_server_state_invalid")
        if metadata.get("server_pid") != process.pid or any(
            metadata.get(key) != value for key, value in contract.metadata.items()
        ):
            raise ContractError(
                "Persistent server metadata differs from the frozen config",
                "magpie_local_server_metadata_mismatch",
            )
        processes = self._processes.snapshot()
        closure = descendant_closure(processes, (process,))
        listeners = self._ports.listener_owners(contract.port, closure)
        if not listeners:
            if required:
                raise ContractError(
                    "Persistent server does not own the configured listener",
                    "magpie_local_server_listener_unverified",
                )
            return None
        return _ServerState(
            process,
            tuple(sorted(item.pid for item in listeners)),
            sha256_json(dict(contract.metadata)),
        )


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
    quality = (
        None if state.request.pass_type is BenchmarkPass.DIAGNOSTIC
        else build_lm_eval_quality_gate(report_path.parent, execution_receipt=None)
    )
    errors = list(dict.fromkeys(state.errors))
    lifecycle_ok = state.contract.lifecycle != "cleanup" or state.quiescence is not None
    runtime_verified = not errors and process_ok and lane and lifecycle_ok
    quality_ok = (
        state.request.pass_type is BenchmarkPass.DIAGNOSTIC or quality is not None
    )
    runtime_receipt = _local_runtime_receipt(state, process_ok, runtime_verified)
    reward_eligible = (
        state.request.pass_type is BenchmarkPass.MEASUREMENT
        and state.contract.lifecycle != "cleanup"
        and runtime_verified
        and quality_ok
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
        "reward_eligible": reward_eligible,
        "profiling_enabled": state.request.pass_type is BenchmarkPass.DIAGNOSTIC,
        "process": _process_receipt(state, command_exit_code, timed_out),
        "dependencies": {
            "schema": "apex.magpie-dependency-attestation/v1",
            "verified": not any("dependency" in item for item in errors),
            "receipts": state.dependencies,
        },
        "runtime": {
            "schema": "apex.magpie-runtime-attestation/v1",
            "verified": runtime_receipt["verified"],
            "model_revision_receipt": None,
            "inferencex_runtime_receipt": None,
            "lm_eval_runtime_receipt": _not_requested_lm_eval(state),
            "serving_runtime_receipt": runtime_receipt,
        },
        "gpu_engagement": {
            "schema": "apex.magpie-gpu-engagement/v1",
            "verified": state.gpu is not None and lifecycle_ok,
            "devices": list(state.gpu.get("devices", [])) if state.gpu else [],
            "processes": list(state.gpu.get("processes", [])) if state.gpu else [],
        },
        "quality_gate": {
            "schema": "apex.magpie-quality-attestation/v1",
            "verified": quality_ok,
            "receipt": quality,
        },
        "errors": errors,
    }


def _local_runtime_receipt(
    state: _ObservationState, process_succeeded: bool, verified: bool
) -> Mapping[str, object]:
    server = state.final_server or state.initial_server
    runtime_processes = [
        item.to_dict()
        for item in sorted(
            state.observed.values(),
            key=lambda value: (value.pid, value.start_time_ticks),
        )
    ]
    return build_local_runtime_receipt(
        config_sha256=state.request.config_sha256,
        gpu_lease_digest=state.lease.lease_digest,
        dependencies=state.dependencies,
        lifecycle=state.contract.lifecycle,
        port=state.contract.port,
        server_source_generation_sha256=(
            state.contract.server_source_generation_sha256
        ),
        benchmark_process=(
            state.benchmark_process.to_dict() if state.benchmark_process else None
        ),
        runtime_processes=runtime_processes,
        server_process=server.process.to_dict() if server else None,
        server_listener_pids=server.listener_pids if server else (),
        compatibility_sha256=server.compatibility_sha256 if server else None,
        observed_listener_pids=state.listener_pids,
        quiescence_receipt=state.quiescence,
        process_succeeded=process_succeeded,
        verified=verified,
        errors=state.errors,
    )


def _process_receipt(state, exit_code, timed_out) -> Mapping[str, object]:
    return {
        "schema": "apex.magpie-process-attestation/v1",
        "argv_sha256": sha256_json(list(state.request.benchmark_argv)),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "succeeded": exit_code == 0 and not timed_out,
        "verified": state.benchmark_process is not None and not any(
            item.startswith("magpie_local_observer") for item in state.errors
        ),
    }


def _not_requested_lm_eval(state: _ObservationState) -> Mapping[str, object] | None:
    if state.request.pass_type is not BenchmarkPass.DIAGNOSTIC:
        return None
    return {
        "schema": "magpie.lm-eval-runtime-evidence/v1", "requested": False,
        "status": "not_requested", "verified": False, "evidence_present": False,
        "runtime_sha256": None, "identity": None, "mount_mode": None,
        "manifest_artifact": None, "receipt_artifact": None, "errors": [],
    }


def _load_report(path: Path, run_root: Path) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(run_root.resolve())
    except ValueError as error:
        raise ContractError("Magpie report escapes run root", "benchmark_workspace_mismatch") from error
    raw = read_regular(path, MAX_CONFIG_BYTES * 16)
    return json_mapping(raw, "invalid_benchmark_report")


def _same_cgroup(left: LocalProcessIdentity, right: LocalProcessIdentity) -> bool:
    return left.cgroup_sha256 == right.cgroup_sha256 and left.cgroup_lines == right.cgroup_lines


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _key(value: LocalProcessIdentity) -> tuple[int, int]:
    return value.pid, value.start_time_ticks


def _error(prefix: str, error: Exception) -> str:
    reason = getattr(error, "reason_code", type(error).__name__)
    return f"{prefix}:{reason}"


__all__ = ["LocalMagpieExecutionAttestor"]
