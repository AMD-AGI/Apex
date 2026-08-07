"""Terminal measurement, source-bundle delivery, and clean-replay proof."""

from __future__ import annotations

from typing import Any

from apex.benchmark import BenchmarkConfigViews
from apex.core import ContractError, TaskStatus, ValidationLevel
from apex.evaluation import E2EAcceptancePolicy, E2EMeasurement, evaluate_no_regression
from apex.orchestration import RunPhase, SearchStage
from apex.runtime import GpuLeaseReceipt, RunProvenance

from ..projections import publish_terminal_projections
from .benchmarking import Diagnosis, E2EBenchmarkSession, measurement_metrics
from .result import E2EOptimizationResult, build_e2e_result, write_e2e_result
from .run_record import E2ERunRecord
from .search import SearchOutcome
from .services import AcceptedCandidate, FinalDeliveryPort, FinalDeliveryRequest


class E2EFinalizer:
    """Turn primary E2E evidence into a terminal result without overstating proof."""

    def __init__(
        self,
        *,
        record: E2ERunRecord,
        session: E2EBenchmarkSession,
        views: BenchmarkConfigViews,
        provenance: RunProvenance,
        delivery: FinalDeliveryPort,
        gpu_lease: GpuLeaseReceipt,
        acceptance_policy: E2EAcceptancePolicy,
        accuracy_policy_sha256: str,
        performance_policy_sha256: str,
        safety_policy_sha256: str | None,
    ) -> None:
        self.record = record
        self.session = session
        self.views = views
        self.provenance = provenance
        self.delivery = delivery
        self.gpu_lease = gpu_lease
        self.acceptance_policy = acceptance_policy
        self.accuracy_policy_sha256 = accuracy_policy_sha256
        self.performance_policy_sha256 = performance_policy_sha256
        self.safety_policy_sha256 = safety_policy_sha256

    def run(
        self,
        *,
        initial: Diagnosis,
        baseline: E2EMeasurement,
        search: SearchOutcome,
    ) -> E2EOptimizationResult:
        if _stage(self.record) is not SearchStage.FINALIZING:
            raise ContractError(
                "E2E search did not reach finalization", "illegal_e2e_transition"
            )
        benchmark, final, receipt = self.session.measure(
            "final-measurement", search.measurement_config
        )
        if not benchmark.succeeded or final is None:
            return self.failure(
                initial=initial,
                baseline=baseline,
                status=TaskStatus.VERIFICATION_FAILED,
                reason="final_measurement_failed",
            )
        if not search.accepted:
            return self._without_winner(initial, baseline, final, receipt.digest, search)
        cumulative = evaluate_no_regression(
            search.anchor, final, policy=self.acceptance_policy
        )
        if not cumulative.keep:
            return self._cumulative_regression(
                initial, baseline, final, receipt.digest, search, cumulative.to_dict()
            )
        return self._deliver(
            initial, baseline, final, receipt.digest, search, cumulative.to_dict()
        )

    def failure(
        self,
        *,
        initial: Diagnosis | None,
        baseline: E2EMeasurement | None,
        status: TaskStatus,
        reason: str,
    ) -> E2EOptimizationResult:
        if self.record.controller.state.phase is RunPhase.RUNNING:
            if self.record.controller.state.pending_action is not None:
                self.record.controller.abort_pending(reason)
            self.record.controller.finish(RunPhase.FAILED, reason=reason)
        return self._write(
            initial=initial,
            status=status,
            reason=reason,
            validation=ValidationLevel.NONE,
            baseline=baseline,
            final=None,
            no_regression=None,
            details={"gpu_lease": self.gpu_lease.to_dict()},
        )

    def _without_winner(
        self,
        initial: Diagnosis,
        baseline: E2EMeasurement,
        final: E2EMeasurement,
        final_receipt: str,
        search: SearchOutcome,
    ) -> E2EOptimizationResult:
        verdict = evaluate_no_regression(
            baseline, final, policy=self.acceptance_policy
        )
        self.record.controller.commit_e2e_final(
            receipt=final_receipt, clean_replay_verified=False
        )
        reason = _exit_reason(self.record) or "no_source_candidate_improved_workload"
        if verdict.keep:
            unsupported = reason in _UNSUPPORTED_REASONS
            status = TaskStatus.UNSUPPORTED if unsupported else TaskStatus.NO_GAIN
            phase = RunPhase.FAILED if unsupported else RunPhase.SUCCEEDED
            self.record.controller.finish(phase, reason=status.value)
        else:
            status = TaskStatus.VERIFICATION_FAILED
            reason = verdict.reason_code
            self.record.controller.finish(RunPhase.FAILED, reason=reason)
        return self._write(
            initial=initial,
            status=status,
            reason=reason,
            validation=ValidationLevel.NONE,
            baseline=baseline,
            final=final,
            no_regression=verdict.keep,
            details={
                "replay_verdict": verdict.to_dict(),
                "diagnostic_evidence_history": list(search.diagnostic_history),
                "accepted_candidates": [],
                "gpu_lease": self.gpu_lease.to_dict(),
            },
        )

    def _cumulative_regression(
        self,
        initial: Diagnosis,
        baseline: E2EMeasurement,
        final: E2EMeasurement,
        final_receipt: str,
        search: SearchOutcome,
        verdict: dict[str, Any],
    ) -> E2EOptimizationResult:
        lineage = self.record.put_json(
            {
                "final_benchmark_receipt": final_receipt,
                "verdict": verdict,
                "clean_replay_verified": False,
            }
        )
        self.record.controller.commit_e2e_final(
            receipt=lineage.digest, clean_replay_verified=False
        )
        self.record.controller.finish(
            RunPhase.FAILED, reason="final_cumulative_replay_regression"
        )
        return self._write(
            initial=initial,
            status=TaskStatus.VERIFICATION_FAILED,
            reason=str(verdict["reason_code"]),
            validation=_max_validation(search.accepted),
            baseline=baseline,
            final=final,
            no_regression=False,
            details=self._search_details(search, "cumulative_verdict", verdict),
        )

    def _deliver(
        self,
        initial: Diagnosis,
        baseline: E2EMeasurement,
        final: E2EMeasurement,
        final_receipt: str,
        search: SearchOutcome,
        cumulative: dict[str, Any],
    ) -> E2EOptimizationResult:
        agent_backend, agent_model = _accepted_agent_identity(search.accepted)
        result = self.delivery.finalize(
            FinalDeliveryRequest(
                self.record.run_id,
                search.accepted,
                self.provenance,
                self.views.original,
                search.measurement_config,
                search.diagnostic_config,
                search.replay_config,
                baseline,
                final,
                self.record.root / "delivery" / "second-clean-replay",
                agent_backend,
                agent_model,
                self.accuracy_policy_sha256,
                self.performance_policy_sha256,
                self.safety_policy_sha256,
            )
        )
        delivery_receipt = self.record.record_final_delivery(result)
        lineage = self.record.put_json(
            {
                "schema_version": 1,
                "final_benchmark_receipt": final_receipt,
                "final_delivery_receipt": delivery_receipt.digest,
                "clean_replay_verified": result.clean_replay_verified,
                "bundle_digest": result.bundle_digest,
            }
        )
        self.record.controller.commit_e2e_final(
            receipt=lineage.digest,
            clean_replay_verified=result.verified and result.clean_replay_verified,
        )
        self.record.controller.finish(
            RunPhase.SUCCEEDED if result.verified else RunPhase.FAILED,
            reason=(
                "source_rebuild_and_second_clean_replay_verified"
                if result.verified
                else result.reason_code
            ),
        )
        details = self._search_details(search, "cumulative_verdict", cumulative)
        details["final_delivery"] = result.to_dict()
        return self._write(
            initial=initial,
            status=result.status,
            reason=result.reason_code,
            validation=result.validation_level,
            baseline=baseline,
            final=final,
            no_regression=True,
            details=details,
        )

    def _search_details(
        self,
        search: SearchOutcome,
        verdict_name: str,
        verdict: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            verdict_name: verdict,
            "diagnostic_evidence_history": list(search.diagnostic_history),
            "accepted_candidates": _accepted_details(search.accepted),
            "gpu_lease": self.gpu_lease.to_dict(),
        }

    def _write(
        self,
        *,
        initial: Diagnosis | None,
        status: TaskStatus,
        reason: str,
        validation: ValidationLevel,
        baseline: E2EMeasurement | None,
        final: E2EMeasurement | None,
        no_regression: bool | None,
        details: dict[str, Any],
    ) -> E2EOptimizationResult:
        result = build_e2e_result(
            record=self.record,
            views=self.views,
            provenance=self.provenance,
            status=status,
            reason=reason,
            validation_level=validation,
            baseline=baseline,
            final=final,
            plan=initial.plan if initial else None,
            evidence_path=str(initial.evidence_path) if initial else None,
            no_regression=no_regression,
            details=details,
        )
        write_e2e_result(result, self.record.root / "result.json")
        _write_projections(self.record)
        return result


def _accepted_details(accepted: tuple[AcceptedCandidate, ...]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.candidate.candidate_id,
            "opportunity_id": item.opportunity.opportunity_id,
            "changed_files": list(item.candidate.changed_files),
            "srobust": item.micro.srobust,
            "safety_certified": item.safety.safety_certified,
            "validation_level": item.deployment.validation_level.value,
            "primary_metrics": measurement_metrics(item.primary_measurement),
            "decision_receipt": item.decision_receipt,
        }
        for item in accepted
    ]


def _max_validation(accepted: tuple[AcceptedCandidate, ...]) -> ValidationLevel:
    if any(
        item.deployment.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
        for item in accepted
    ):
        return ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    return ValidationLevel.NONE


def _accepted_agent_identity(
    accepted: tuple[AcceptedCandidate, ...],
) -> tuple[str | None, str | None]:
    identities = {
        (item.candidate.agent_result.backend.value, item.candidate.agent_result.model)
        for item in accepted
    }
    if len(identities) != 1:
        return None, None
    backend, model = next(iter(identities))
    return (backend or None), (model or None)


def _stage(record: E2ERunRecord) -> SearchStage:
    search = record.controller.state.e2e
    if search is None:
        raise ContractError("E2E state is not initialized", "e2e_not_initialized")
    return search.stage


def _exit_reason(record: E2ERunRecord) -> str | None:
    search = record.controller.state.e2e
    return search.exit_reason if search is not None else None


def _write_projections(record: E2ERunRecord) -> None:
    publish_terminal_projections(
        root=record.root,
        run_id=record.run_id,
        artifacts=record.artifacts,
        workload_state=record.controller.state,
    )


_UNSUPPORTED_REASONS = {
    "candidate_worker_unavailable",
    "micro_verifier_unavailable",
    "delivery_adapter_unavailable",
}


__all__ = ["E2EFinalizer"]
