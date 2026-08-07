"""Composition root for the kernel-only E2E optimization closed loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.benchmark import (
    BenchmarkConfigViews,
    MagpieBenchmarkAdapter,
    build_config_views,
    validate_resolved_view,
)
from apex.core import (
    ApexError,
    ContractError,
    IntegrityError,
    TaskStatus,
    new_identifier,
    sha256_json,
)
from apex.diagnostics import MagpieTraceEvidenceAdapter
from apex.evaluation import E2EAcceptancePolicy
from apex.intake import E2EOptimizeSpec
from apex.orchestration import RunPhase, SearchStage
from apex.ports import BenchmarkPass
from apex.runtime import (
    DependencyReceipt,
    GpuLeaseManager,
    GpuLeaseReceipt,
    LocalGpuLeaseManager,
    ProvenanceResolver,
    RunProvenance,
)
from apex.storage import ArtifactReceipt

from .benchmarking import (
    BenchmarkAdapter,
    Diagnosis,
    DiagnosticsAdapter,
    E2EBenchmarkSession,
    measurement_metrics,
)
from .candidate import CandidateWorker
from .context import E2EContextBuilder
from .finalization import E2EFinalizer
from .oracles import CorrectnessOracleRegistry
from .result import E2EOptimizationResult, load_bound_terminal_result
from .recovery import (
    RecoveredRunRequest,
    load_run_request,
    persist_run_request,
    recover_baseline,
    recover_diagnosis,
    recover_record,
    recover_uncommitted_diagnosis,
)
from .run_record import E2ERunRecord
from .search import E2ESearchLoop
from .services import (
    CandidateDeploymentPort,
    CandidateSafetyPort,
    FinalDeliveryPort,
    MicroQualificationPort,
    NoToolSafetyVerifier,
    UnavailableDeployment,
    UnavailableFinalDelivery,
    UnavailableMicroQualifier,
)


class ProvenancePort(Protocol):
    def resolve(
        self,
        config_path: Path,
        *,
        gpu_arch: str,
        hints: Mapping[str, Any] | None = None,
    ) -> RunProvenance: ...


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    spec: E2EOptimizeSpec
    record: E2ERunRecord
    views: BenchmarkConfigViews
    provenance: RunProvenance
    session: E2EBenchmarkSession
    gpu_device_scope: str


class E2EOptimizeUseCase:
    """Own one GPU lease while state, search, and final proof remain modular."""

    def __init__(
        self,
        *,
        dependency_receipt: DependencyReceipt,
        benchmark: BenchmarkAdapter | None = None,
        diagnostics: DiagnosticsAdapter | None = None,
        provenance: ProvenancePort | None = None,
        candidate_worker: CandidateWorker | None = None,
        contexts: E2EContextBuilder | None = None,
        micro: MicroQualificationPort | None = None,
        safety: CandidateSafetyPort | None = None,
        deployments: CandidateDeploymentPort | None = None,
        final_delivery: FinalDeliveryPort | None = None,
        gpu_leases: GpuLeaseManager | None = None,
        correctness_oracles: CorrectnessOracleRegistry | None = None,
    ) -> None:
        self._receipt = dependency_receipt
        self._benchmark = benchmark or MagpieBenchmarkAdapter(dependency_receipt)
        self._diagnostics = diagnostics or MagpieTraceEvidenceAdapter()
        self._provenance = provenance or ProvenanceResolver()
        self._candidate_worker = candidate_worker
        self._contexts = contexts or E2EContextBuilder()
        self._micro = micro or UnavailableMicroQualifier()
        self._safety = safety or NoToolSafetyVerifier()
        self._deployments = deployments or UnavailableDeployment()
        self._final_delivery = final_delivery or UnavailableFinalDelivery()
        self._gpu_leases = gpu_leases or LocalGpuLeaseManager()
        self._correctness_oracles = correctness_oracles

    def run(self, spec: E2EOptimizeSpec) -> E2EOptimizationResult:
        run_id = new_identifier("e2e")
        with self._gpu_leases.acquire(
            run_id, requested_devices=_gpu_devices(spec)
        ) as lease:
            return self._run_leased(spec, run_id, lease.receipt)

    def resume(self, run_root: Path) -> E2EOptimizationResult:
        """Resume a crash at a durable baseline/diagnostic boundary."""

        request = load_run_request(run_root)
        if request.dependency_lock_sha256 != self._receipt.lock_sha256:
            raise ContractError(
                "Runtime dependency lock differs from the interrupted run",
                "resume_dependency_mismatch",
            )
        record = recover_record(request)
        bound = load_bound_terminal_result(
            record,
            expected_provenance_hash=request.provenance_digest,
            expected_views=request.views,
        )
        if bound is not None:
            if record.controller.state.phase is RunPhase.RUNNING:
                record.controller.finish(bound.phase, reason=bound.stop_reason)
            elif record.controller.state.phase is not bound.phase:
                raise IntegrityError(
                    "Terminal result conflicts with run state",
                    "e2e_result_binding_mismatch",
                )
            _verify_terminal_phase(record.controller.state.phase, bound.result.status)
            return bound.result
        if record.controller.state.phase is not RunPhase.RUNNING:
            raise IntegrityError(
                "Terminal run has no journal-bound result",
                "e2e_result_binding_missing",
            )
        with self._gpu_leases.acquire(
            request.run_id, requested_devices=_gpu_devices(request.spec)
        ) as lease:
            return self._resume_leased(request, record, lease.receipt)

    def _run_leased(
        self,
        spec: E2EOptimizeSpec,
        run_id: str,
        gpu_lease: GpuLeaseReceipt,
    ) -> E2EOptimizationResult:
        prepared: _PreparedRun | None = None
        baseline = None
        diagnosis: Diagnosis | None = None
        try:
            prepared = self._prepare(spec, run_id, gpu_lease)
            result, baseline, receipt = prepared.session.measure(
                "baseline-measurement", prepared.views.measurement
            )
            if not result.succeeded or baseline is None:
                return self._finalizer(prepared, gpu_lease).failure(
                    initial=None,
                    baseline=None,
                    status=TaskStatus.BASELINE_INVALID,
                    reason="baseline_quality_or_measurement_failed",
                )
            prepared.record.controller.commit_e2e_baseline(
                receipt=receipt.digest,
                metrics=measurement_metrics(baseline),
                quality_passed=True,
            )
            diagnosis = prepared.session.diagnose(
                "diagnostic-0", prepared.views.diagnostic
            )
            prepared.record.controller.commit_e2e_diagnostics(
                receipt=diagnosis.state_receipt.digest,
                opportunity_ids=tuple(
                    item.opportunity_id for item in diagnosis.plan.eligible
                ),
            )
            search = self._search(spec, prepared).run(diagnosis, baseline)
            return self._finalizer(prepared, gpu_lease).run(
                initial=diagnosis,
                baseline=baseline,
                search=search,
            )
        except ApexError as error:
            if prepared is None:
                raise
            return self._finalizer(prepared, gpu_lease).failure(
                initial=diagnosis,
                baseline=baseline,
                status=TaskStatus.INFRASTRUCTURE_ERROR,
                reason=error.reason_code,
            )

    def _prepare(
        self, spec: E2EOptimizeSpec, run_id: str, gpu_lease: GpuLeaseReceipt
    ) -> _PreparedRun:
        provenance = self._provenance.resolve(
            spec.config_path,
            gpu_arch=spec.gpu_arch,
            hints=spec.deployment_hints,
        )
        record = E2ERunRecord.create(
            run_id=run_id,
            root=spec.results_dir,
            initial_anchor_id=f"anchor-{provenance.digest[:16]}",
            dataset_split=spec.dataset_split,
            data_visibility=spec.data_visibility,
        )
        self._record_run_identity(record, gpu_lease, provenance)
        views = build_config_views(
            spec.config_path,
            record.root / "configs",
            dependency_receipt=self._receipt,
            source_repository_roots=tuple(
                Path(lock.path) for lock in provenance.source_locks if lock.exact
            ),
            model_revision=provenance.model_revision,
            hf_cache_path=_hf_cache_path(spec),
            gpu_devices=_gpu_devices(spec),
            hf_offline=_hf_offline(spec),
        )
        record.controller.initialize_e2e(
            workload_id=f"workload-{provenance.benchmark_config_sha256[:16]}",
            provenance_hash=provenance.digest,
            objective_policy_hash=_objective_hash(spec),
            accuracy_contract_hash=_accuracy_hash(
                views, spec, self._correctness_oracles
            ),
            measurement_protocol_hash=views.workload_semantics_sha256,
            candidate_limit=spec.max_kernels * spec.max_iterations,
            cycle_limit=spec.max_iterations,
        )
        persist_run_request(
            record,
            spec=spec,
            dependency_lock_sha256=self._receipt.lock_sha256,
            provenance_digest=provenance.digest,
            views=views,
            correctness_oracle_policy_sha256=getattr(
                self._correctness_oracles, "policy_sha256", None
            ),
        )
        session = E2EBenchmarkSession(
            benchmark=self._benchmark,
            diagnostics=self._diagnostics,
            record=record,
            provenance=provenance,
            protocol_hash=views.workload_semantics_sha256,
            max_kernels=spec.max_kernels,
            correctness_oracles=self._correctness_oracles,
        )
        return _PreparedRun(
            spec, record, views, provenance, session, gpu_lease.device_scope
        )

    def _resume_leased(
        self,
        request: RecoveredRunRequest,
        record: E2ERunRecord,
        gpu_lease: GpuLeaseReceipt,
    ) -> E2EOptimizationResult:
        self._record_resume_lease(record, gpu_lease)
        prepared = self._recover_prepared(request, record)
        if record.controller.state.pending_action is not None:
            record.controller.abort_pending("interrupted_before_completion_receipt")
        search_state = record.controller.state.e2e
        assert search_state is not None
        baseline = recover_baseline(record) if search_state.baseline_receipt else None
        stage = search_state.stage
        if stage is SearchStage.BASELINING:
            result, baseline, receipt = prepared.session.measure(
                f"baseline-resume-{record.controller.state.sequence}",
                prepared.views.measurement,
            )
            if not result.succeeded or baseline is None:
                return self._finalizer(prepared, gpu_lease).failure(
                    initial=None,
                    baseline=None,
                    status=TaskStatus.BASELINE_INVALID,
                    reason="baseline_quality_or_measurement_failed",
                )
            record.controller.commit_e2e_baseline(
                receipt=receipt.digest,
                metrics=measurement_metrics(baseline),
                quality_passed=True,
            )
            stage = record.controller.state.e2e.stage
        if baseline is None:
            raise ContractError("Resume baseline is missing", "baseline_not_committed")
        if stage is SearchStage.DIAGNOSING:
            recovered = recover_uncommitted_diagnosis(record)
            if recovered is None:
                diagnosis = prepared.session.diagnose(
                    f"diagnostic-resume-{record.controller.state.sequence}",
                    prepared.views.diagnostic,
                )
            else:
                plan, path, evidence, lineage = recovered
                diagnosis = Diagnosis(plan, path, evidence, lineage)
            record.controller.commit_e2e_diagnostics(
                receipt=diagnosis.state_receipt.digest,
                opportunity_ids=tuple(
                    item.opportunity_id for item in diagnosis.plan.eligible
                ),
            )
        elif stage in {SearchStage.PLANNING, SearchStage.FINALIZING}:
            plan, path, evidence, lineage = recover_diagnosis(record)
            diagnosis = Diagnosis(plan, path, evidence, lineage)
        else:
            raise ContractError(
                f"Safe resume is unavailable from stage {stage.value}",
                "resume_stage_unsupported",
            )
        if record.controller.state.e2e.budget.candidates_used:
            raise ContractError(
                "Mid-candidate recovery requires a frozen candidate checkpoint",
                "resume_stage_unsupported",
            )
        outcome = self._search(request.spec, prepared).run(diagnosis, baseline)
        return self._finalizer(prepared, gpu_lease).run(
            initial=diagnosis,
            baseline=baseline,
            search=outcome,
        )

    def _recover_prepared(
        self, request: RecoveredRunRequest, record: E2ERunRecord
    ) -> _PreparedRun:
        spec = request.spec
        provenance = self._provenance.resolve(
            spec.config_path, gpu_arch=spec.gpu_arch, hints=spec.deployment_hints
        )
        if provenance.digest != request.provenance_digest:
            raise ContractError("Run provenance changed", "resume_provenance_mismatch")
        observed_oracle = getattr(self._correctness_oracles, "policy_sha256", None)
        if observed_oracle != request.correctness_oracle_policy_sha256:
            raise ContractError("Oracle policy changed", "resume_oracle_policy_mismatch")
        for path, pass_type in (
            (request.views.measurement, BenchmarkPass.MEASUREMENT),
            (request.views.diagnostic, BenchmarkPass.DIAGNOSTIC),
            (request.views.replay, BenchmarkPass.MEASUREMENT),
        ):
            validate_resolved_view(
                path, pass_type=pass_type, dependency_receipt=self._receipt
            )
        session = E2EBenchmarkSession(
            benchmark=self._benchmark,
            diagnostics=self._diagnostics,
            record=record,
            provenance=provenance,
            protocol_hash=request.views.workload_semantics_sha256,
            max_kernels=spec.max_kernels,
            correctness_oracles=self._correctness_oracles,
        )
        return _PreparedRun(spec, record, request.views, provenance, session)

    @staticmethod
    def _record_run_identity(
        record: E2ERunRecord,
        gpu_lease: GpuLeaseReceipt,
        provenance: RunProvenance,
    ) -> None:
        lease_receipt = record.put_json(gpu_lease.to_dict())
        provenance_receipt = record.put_json(provenance.to_dict())
        record.controller.record_domain_event(
            "dependency_verified",
            {
                "kind": "gpu_lease",
                "device_scope": gpu_lease.device_scope,
                "lease_digest": gpu_lease.digest,
                "artifacts": [_binding("gpu_lease", lease_receipt)],
            },
            idempotency_key="gpu_lease.acquired",
        )
        record.controller.record_domain_event(
            "provenance_observed",
            {
                "status": provenance.status,
                "provenance_hash": provenance.digest,
                "missing_evidence": list(provenance.missing_evidence),
                "provenance": provenance.to_dict(),
                "artifacts": [_binding("run_provenance", provenance_receipt)],
            },
            idempotency_key="provenance.observed",
        )

    @staticmethod
    def _record_resume_lease(
        record: E2ERunRecord, gpu_lease: GpuLeaseReceipt
    ) -> None:
        receipt = record.put_json(gpu_lease.to_dict())
        record.controller.record_domain_event(
            "dependency_verified",
            {
                "kind": "gpu_lease_resume",
                "device_scope": gpu_lease.device_scope,
                "lease_digest": gpu_lease.digest,
                "artifacts": [_binding("gpu_lease", receipt)],
            },
            idempotency_key=f"gpu_lease.resume.{record.controller.state.sequence}",
        )

    def _search(self, spec: E2EOptimizeSpec, prepared: _PreparedRun) -> E2ESearchLoop:
        return E2ESearchLoop(
            spec=spec,
            record=prepared.record,
            session=prepared.session,
            provenance=prepared.provenance,
            views=prepared.views,
            candidate_worker=self._candidate_worker,
            contexts=self._contexts,
            micro=self._micro,
            safety=self._safety,
            deployments=self._deployments,
            gpu_device_scope=prepared.gpu_device_scope,
        )

    def _finalizer(
        self, prepared: _PreparedRun, gpu_lease: GpuLeaseReceipt
    ) -> E2EFinalizer:
        policy = E2EAcceptancePolicy(prepared.spec.goal.gates)
        return E2EFinalizer(
            record=prepared.record,
            session=prepared.session,
            views=prepared.views,
            provenance=prepared.provenance,
            delivery=self._final_delivery,
            gpu_lease=gpu_lease,
            acceptance_policy=policy,
            accuracy_policy_sha256=_accuracy_hash(
                prepared.views, prepared.spec, self._correctness_oracles
            ),
            performance_policy_sha256=policy.digest,
            safety_policy_sha256=getattr(
                self._safety, "policy_fingerprint", None
            ),
        )


def _objective_hash(spec: E2EOptimizeSpec) -> str:
    return sha256_json(spec.to_dict()["goal"])


def _hf_cache_path(spec: E2EOptimizeSpec) -> Path | None:
    raw = spec.deployment_hints.get("hf_cache_path")
    if raw is None:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        raise ContractError(
            "deployment_hints.hf_cache_path must be absolute",
            "invalid_hf_cache_path",
        )
    return path


def _gpu_devices(spec: E2EOptimizeSpec) -> str | None:
    raw = spec.deployment_hints.get("gpu_devices")
    return str(raw) if raw is not None else None


def _hf_offline(spec: E2EOptimizeSpec) -> bool:
    raw = spec.deployment_hints.get("hf_offline", False)
    if not isinstance(raw, bool):
        raise ContractError(
            "deployment_hints.hf_offline must be a boolean",
            "invalid_hf_offline",
        )
    return raw


def _accuracy_hash(
    views: BenchmarkConfigViews,
    spec: E2EOptimizeSpec,
    correctness_oracles: CorrectnessOracleRegistry | None = None,
) -> str:
    policy = {
        "schema": "apex.e2e-quality-policy-binding.v1",
        "quality_tasks": views.quality_tasks,
        "evaluator_policy_sha256": views.evaluator_policy_sha256,
        "regression_gates": asdict(spec.goal.gates),
    }
    if correctness_oracles is not None:
        policy["correctness_oracle_policy_sha256"] = (
            correctness_oracles.policy_sha256
        )
    return sha256_json(policy)


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def _verify_terminal_phase(phase: RunPhase, status: TaskStatus) -> None:
    successful = {TaskStatus.SUCCEEDED, TaskStatus.NO_GAIN}
    if (phase is RunPhase.SUCCEEDED) != (status in successful):
        raise ContractError("Terminal result conflicts with run state", "e2e_result_binding_mismatch")


__all__ = ["BenchmarkAdapter", "E2EOptimizeUseCase", "ProvenancePort"]
