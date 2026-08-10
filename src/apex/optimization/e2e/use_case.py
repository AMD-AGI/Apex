"""Composition root for the kernel-only E2E optimization closed loop."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apex.benchmark import (
    BenchmarkConfigViews,
    MagpieBenchmarkAdapter,
)
from apex.core import (
    ApexError,
    ContractError,
    IntegrityError,
    TaskStatus,
    new_identifier,
)
from apex.diagnostics import (
    MagpieTraceEvidenceAdapter,
    PinnedTraceLensComparisonAdapter,
)
from apex.evaluation import E2EAcceptancePolicy
from apex.intake import E2EOptimizeSpec
from apex.orchestration import RunPhase, SearchStage
from apex.ports import TraceComparisonPort
from apex.runtime import (
    DependencyReceipt,
    FormalResultsRootValidator,
    GpuLease,
    GpuLeaseManager,
    GpuLeaseReceipt,
    LocalGpuLeaseManager,
    MagpieMainConfigAdapter,
    ProvenanceResolver,
    ReleaseCandidateReceipt,
    RunProvenance,
    formal_results_validator,
)

from ..baseline_recording import (
    record_campaign_baseline,
    validate_resume_campaign_baseline,
)
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
from .preflight import (
    E2EPreflightResult,
    MagpieConfigResolutionPort,
    ProvenancePort,
    build_preflight_views,
    compose_preflight_result,
    gpu_devices,
    resolve_preflight_provenance,
    resolve_preflight_contract,
    require_benchmark_execution_available,
    require_formal_measurement_available,
    validate_resume_preflight,
)
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
from .run_contracts import (
    accuracy_hash as _accuracy_hash,
    artifact_binding as _binding,
    objective_hash as _objective_hash,
    relocate_views as _relocate_views,
    require_optimizable_contract as _require_optimizable_contract,
    verify_resume_gpu_lease as _verify_resume_gpu_lease,
    verify_terminal_phase as _verify_terminal_phase,
)
from .recovery_search import recover_search
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


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    spec: E2EOptimizeSpec
    record: E2ERunRecord
    views: BenchmarkConfigViews
    provenance: RunProvenance
    session: E2EBenchmarkSession
    gpu_lease: GpuLeaseReceipt


class E2EOptimizeUseCase:
    """Own one GPU lease while state, search, and final proof remain modular."""
    def __init__(
        self,
        *,
        dependency_receipt: DependencyReceipt,
        benchmark: BenchmarkAdapter | None = None,
        diagnostics: DiagnosticsAdapter | None = None,
        trace_comparison: TraceComparisonPort | None = None,
        provenance: ProvenancePort | None = None,
        resolved_plans: MagpieConfigResolutionPort | None = None,
        candidate_worker: CandidateWorker | None = None,
        contexts: E2EContextBuilder | None = None,
        micro: MicroQualificationPort | None = None,
        safety: CandidateSafetyPort | None = None,
        deployments: CandidateDeploymentPort | None = None,
        final_delivery: FinalDeliveryPort | None = None,
        gpu_leases: GpuLeaseManager | None = None,
        correctness_oracles: CorrectnessOracleRegistry | None = None,
        results_validator: FormalResultsRootValidator | None = None,
    ) -> None:
        self._receipt = dependency_receipt
        self._benchmark = benchmark or MagpieBenchmarkAdapter(dependency_receipt)
        self._diagnostics = diagnostics or MagpieTraceEvidenceAdapter()
        self._trace_comparison = trace_comparison or PinnedTraceLensComparisonAdapter(
            root=dependency_receipt.root("tracelens"),
            commit=dependency_receipt.commits["tracelens"],
        )
        self._provenance = provenance or ProvenanceResolver()
        self._resolved_plans = resolved_plans or MagpieMainConfigAdapter(
            dependency_receipt
        )
        self._candidate_worker = candidate_worker
        self._contexts = contexts or E2EContextBuilder()
        self._micro = micro or UnavailableMicroQualifier()
        self._safety = safety or NoToolSafetyVerifier()
        self._deployments = deployments or UnavailableDeployment()
        self._final_delivery = final_delivery or UnavailableFinalDelivery()
        self._gpu_leases = gpu_leases or LocalGpuLeaseManager()
        self._correctness_oracles = correctness_oracles
        self._results_validator = results_validator or _results_policy(
            dependency_receipt
        )

    def preview(self, spec: E2EOptimizeSpec) -> E2EPreflightResult:
        """Resolve config/capabilities without creating a run or acquiring a GPU."""

        resolved = resolve_preflight_contract(self._resolved_plans, spec)
        provenance = resolve_preflight_provenance(
            self._provenance, spec, resolved
        )
        parent = spec.results_dir.expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".apex-e2e-preflight-", dir=parent
        ) as root:
            views = (
                build_preflight_views(
                    self._receipt, spec, provenance, resolved, Path(root)
                )
                if resolved.status == "config_compatible"
                else None
            )
            return compose_preflight_result(
                self._receipt,
                views,
                provenance,
                resolved,
                benchmark=self._benchmark,
                deployment=self._deployments,
                micro=self._micro,
                final_delivery=self._final_delivery,
            )

    def run(self, spec: E2EOptimizeSpec) -> E2EOptimizationResult:
        self._results_validator.validate(spec.results_dir, require_new=True)
        run_id = new_identifier("e2e")
        resolved = resolve_preflight_contract(self._resolved_plans, spec)
        _require_optimizable_contract(resolved)
        require_benchmark_execution_available(self._benchmark, resolved)
        require_formal_measurement_available(self._benchmark, resolved)
        provenance = resolve_preflight_provenance(
            self._provenance, spec, resolved
        )
        staging_parent = spec.results_dir.expanduser().resolve().parent
        staging_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".apex-e2e-configs-", dir=staging_parent
        ) as staging_root:
            staged = build_preflight_views(
                self._receipt, spec, provenance, resolved, Path(staging_root)
            )
            with self._gpu_leases.acquire(
                run_id, requested_devices=gpu_devices(spec)
            ) as lease:
                return self._run_leased(
                    spec, run_id, lease, provenance, staged
                )

    def resume(
        self,
        run_root: Path,
        *,
        campaign_baseline: ReleaseCandidateReceipt | None = None,
    ) -> E2EOptimizationResult:
        """Resume a crash at a durable baseline/diagnostic boundary."""

        self._results_validator.validate(run_root)
        request = load_run_request(run_root)
        validate_resume_campaign_baseline(
            request.spec.campaign_baseline_receipt,
            campaign_baseline,
        )
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
        resolved = resolve_preflight_contract(self._resolved_plans, request.spec)
        provenance = validate_resume_preflight(
            self._provenance,
            self._resolved_plans,
            self._receipt,
            request,
            getattr(self._correctness_oracles, "policy_sha256", None),
        )
        require_benchmark_execution_available(self._benchmark, resolved)
        require_formal_measurement_available(self._benchmark, resolved)
        with self._gpu_leases.acquire(
            request.run_id, requested_devices=gpu_devices(request.spec)
        ) as lease:
            return self._resume_leased(
                request, record, lease, provenance
            )

    def _run_leased(
        self,
        spec: E2EOptimizeSpec,
        run_id: str,
        gpu_lease: GpuLease,
        provenance: RunProvenance,
        staged_views: BenchmarkConfigViews,
    ) -> E2EOptimizationResult:
        prepared: _PreparedRun | None = None
        baseline = None
        diagnosis: Diagnosis | None = None
        try:
            prepared = self._prepare(
                spec, run_id, gpu_lease, provenance, staged_views
            )
            result, baseline, receipt = prepared.session.measure(
                "baseline-measurement", prepared.views.measurement
            )
            if not result.succeeded or baseline is None:
                return self._finalizer(prepared, gpu_lease.receipt).failure(
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
                "diagnostic-0",
                prepared.views.diagnostic,
                preserve_raw_trace=True,
            )
            prepared.record.controller.commit_e2e_diagnostics(
                receipt=diagnosis.state_receipt.digest,
                opportunity_ids=tuple(
                    item.opportunity_id for item in diagnosis.plan.eligible
                ),
            )
            search = self._search(spec, prepared).run(diagnosis, baseline)
            return self._finalizer(prepared, gpu_lease.receipt).run(
                initial=diagnosis,
                baseline=baseline,
                search=search,
            )
        except ApexError as error:
            if prepared is None:
                raise
            return self._finalizer(prepared, gpu_lease.receipt).failure(
                initial=diagnosis,
                baseline=baseline,
                status=TaskStatus.INFRASTRUCTURE_ERROR,
                reason=error.reason_code,
                details={
                    "reason_code": error.reason_code,
                    "evidence": dict(error.details or {}),
                },
            )

    def _prepare(
        self,
        spec: E2EOptimizeSpec,
        run_id: str,
        gpu_lease: GpuLease,
        provenance: RunProvenance,
        staged: BenchmarkConfigViews,
    ) -> _PreparedRun:
        record = E2ERunRecord.create(
            run_id=run_id,
            root=spec.results_dir,
            initial_anchor_id=f"anchor-{provenance.digest[:16]}",
            dataset_split=spec.dataset_split,
            data_visibility=spec.data_visibility,
        )
        destination = record.root / "configs"
        os.replace(staged.original.parent, destination)
        views = _relocate_views(staged, destination)
        receipt = gpu_lease.receipt
        self._record_run_identity(record, receipt, provenance)
        if spec.campaign_baseline_receipt is not None:
            record_campaign_baseline(
                record.artifacts,
                record.controller,
                spec.campaign_baseline_receipt,
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
            gpu_device_scope=receipt.execution_scope,
        )
        session = E2EBenchmarkSession(
            benchmark=self._benchmark,
            diagnostics=self._diagnostics,
            record=record,
            provenance=provenance,
            protocol_hash=views.workload_semantics_sha256,
            max_kernels=spec.max_kernels,
            trace_comparison=self._trace_comparison,
            gpu_lease=gpu_lease,
            correctness_oracles=self._correctness_oracles,
        )
        return _PreparedRun(spec, record, views, provenance, session, receipt)

    def _resume_leased(
        self,
        request: RecoveredRunRequest,
        record: E2ERunRecord,
        gpu_lease: GpuLease,
        provenance: RunProvenance,
    ) -> E2EOptimizationResult:
        _verify_resume_gpu_lease(request, gpu_lease.receipt, record.iter_events())
        self._record_resume_lease(record, gpu_lease.receipt)
        prepared = self._recover_prepared(
            request, record, gpu_lease=gpu_lease, provenance=provenance
        )
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
                return self._finalizer(prepared, gpu_lease.receipt).failure(
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
                    preserve_raw_trace=True,
                )
            else:
                plan, path, evidence, lineage, comparison = recovered
                diagnosis = Diagnosis(plan, path, evidence, lineage, comparison)
            record.controller.commit_e2e_diagnostics(
                receipt=diagnosis.state_receipt.digest,
                opportunity_ids=tuple(
                    item.opportunity_id for item in diagnosis.plan.eligible
                ),
            )
        elif stage is SearchStage.BASELINING:
            raise ContractError(
                "Baseline recovery did not advance", "baseline_not_committed"
            )
        projection = recover_search(
            record,
            spec=request.spec,
            views=request.views,
            baseline=baseline,
        )
        diagnosis = projection.initial_diagnosis
        outcome = self._search(request.spec, prepared).run(
            projection.diagnosis,
            baseline,
            recovery=projection,
        )
        return self._finalizer(prepared, gpu_lease.receipt).run(
            initial=diagnosis,
            baseline=baseline,
            search=outcome,
            measurement_action_id=(
                f"final-measurement-resume-"
                f"{record.controller.state.sequence}"
            ),
        )

    def _recover_prepared(
        self,
        request: RecoveredRunRequest,
        record: E2ERunRecord,
        *,
        gpu_lease: GpuLease,
        provenance: RunProvenance,
    ) -> _PreparedRun:
        spec = request.spec
        session = E2EBenchmarkSession(
            benchmark=self._benchmark,
            diagnostics=self._diagnostics,
            record=record,
            provenance=provenance,
            protocol_hash=request.views.workload_semantics_sha256,
            max_kernels=spec.max_kernels,
            trace_comparison=self._trace_comparison,
            gpu_lease=gpu_lease,
            correctness_oracles=self._correctness_oracles,
        )
        return _PreparedRun(
            spec, record, request.views, provenance, session, gpu_lease.receipt
        )

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
                "execution_scope": gpu_lease.execution_scope,
                "physical_scope": gpu_lease.physical_scope,
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
                "execution_scope": gpu_lease.execution_scope,
                "physical_scope": gpu_lease.physical_scope,
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
            gpu_lease=prepared.gpu_lease,
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


__all__ = ["BenchmarkAdapter", "E2EOptimizeUseCase", "ProvenancePort"]


def _results_policy(receipt: DependencyReceipt) -> FormalResultsRootValidator:
    source_roots = (
        tuple(receipt.source_locks.roots.values())
        if receipt.source_locks is not None
        else ()
    )
    return formal_results_validator(
        apex_root=Path(__file__).resolve().parents[4],
        dependency_roots=receipt.roots.values(),
        source_roots=source_roots,
    )
