"""Bounded kernel-candidate search against the current live E2E anchor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apex.benchmark import BenchmarkConfigViews
from apex.core import ApexError, ContractError, IntegrityError
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EMeasurement,
)
from apex.intake import E2EOptimizeSpec
from apex.orchestration import SearchStage
from apex.runtime import GpuLeaseReceipt, RunProvenance
from apex.storage import ArtifactReceipt

from .benchmarking import Diagnosis, E2EBenchmarkSession
from .candidate import CandidateWorker, E2ECandidate, E2ECandidateRequest
from .context import E2EContextBuilder
from .kernel_lane import KernelOpportunity
from .outcomes import commit_e2e_reject, commit_measured_e2e_outcome
from .promotion import MatchedPromotion, MatchedPromotionRunner
from .recovery_search import RecoveredSearch
from .run_record import E2ERunRecord
from .search_recovery import SearchRecovery
from .search_support import (
    QualifiedAttempt as _QualifiedAttempt,
    candidate_id as _candidate_id,
    commit_qualified_reject as _commit_qualified_reject,
    opportunity_map as _opportunity_map,
    promotion_artifacts as _promotion_artifacts,
    promotion_receipts as _promotion_receipts,
    raise_agent_teardown_infrastructure as _raise_agent_teardown_infrastructure,
    search_stage as _stage,
    source_key as _source_key,
    validate_deployment as _validate_deployment,
)
from .services import (
    AcceptedCandidate,
    CandidateDeployment,
    CandidateDeploymentPort,
    CandidateDeploymentRequest,
    CandidateSafetyPort,
    MicroQualification,
    MicroQualificationPort,
    MicroQualificationRequest,
    SafetyQualification,
    SafetyQualificationRequest,
)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Durable search output consumed by finalization."""

    accepted: tuple[AcceptedCandidate, ...]
    anchor: E2EMeasurement
    measurement_config: Path
    diagnostic_config: Path
    replay_config: Path
    diagnostic_history: tuple[str, ...]

class E2ESearchLoop:
    """Consume a dynamic opportunity queue using fresh stateless agent calls."""

    def __init__(
        self,
        *,
        spec: E2EOptimizeSpec,
        record: E2ERunRecord,
        session: E2EBenchmarkSession,
        provenance: RunProvenance,
        views: BenchmarkConfigViews,
        candidate_worker: CandidateWorker | None,
        contexts: E2EContextBuilder,
        micro: MicroQualificationPort,
        safety: CandidateSafetyPort,
        deployments: CandidateDeploymentPort,
        gpu_lease: GpuLeaseReceipt,
    ) -> None:
        self.spec = spec
        self.record = record
        self.session = session
        self.provenance = provenance
        self.views = views
        self.worker = candidate_worker
        self.contexts = contexts
        self.micro = micro
        self.safety = safety
        self.deployments = deployments
        self.gpu_lease = gpu_lease
        self.gpu_device_scope = gpu_lease.execution_scope
        self.promotions = MatchedPromotionRunner(
            session=session,
            record=record,
            gpu_lease=gpu_lease,
            policy=E2EAcceptancePolicy(spec.goal.gates),
        )

    def run(
        self,
        initial: Diagnosis,
        baseline: E2EMeasurement,
        *,
        recovery: RecoveredSearch | None = None,
    ) -> SearchOutcome:
        recovery_driver = SearchRecovery(self)
        progress = recovery_driver.progress(initial, baseline, recovery)
        if recovery is not None:
            recovery_driver.reconcile(progress, recovery.active)
        opportunities = _opportunity_map(progress.diagnosis)
        while _stage(self.record) is SearchStage.PLANNING:
            opportunity, reason = self._select_available(opportunities)
            if opportunity is None:
                self.record.controller.request_e2e_finalization(
                    reason=reason or "no_eligible_kernel_source"
                )
                break
            winner = self._attempt(
                opportunity,
                progress.anchor,
                progress.diagnosis.evidence_receipt,
                progress.configs,
                (
                    progress.accepted[-1].deployment.deployed_image_id
                    if progress.accepted
                    else None
                ),
                tuple(progress.accepted),
            )
            if winner is None:
                continue
            recovery_driver.accept(progress, winner)
            progress.diagnosis = self._reprofile(
                progress.configs[1],
                len(progress.accepted),
                frozenset(progress.accepted_sources),
            )
            recovery_driver.append_history(progress, progress.diagnosis)
            opportunities = _opportunity_map(progress.diagnosis)
        return SearchOutcome(
            tuple(progress.accepted),
            progress.anchor,
            *progress.configs,
            tuple(progress.history),
        )

    def _select_available(
        self, opportunities: Mapping[str, KernelOpportunity]
    ) -> tuple[KernelOpportunity | None, str | None]:
        search = self.record.controller.state.e2e
        assert search is not None
        if self.worker is None:
            return None, "candidate_worker_unavailable"
        unsupported_micro = False
        unsupported_delivery = False
        for opportunity_id in search.opportunity_queue:
            opportunity = opportunities.get(opportunity_id)
            if opportunity is None or not opportunity.eligible:
                continue
            if not self.micro.supports(opportunity):
                unsupported_micro = True
                continue
            if not self.deployments.supports(opportunity, self.provenance):
                unsupported_delivery = True
                continue
            return opportunity, None
        if unsupported_micro:
            return None, "micro_verifier_unavailable"
        if unsupported_delivery:
            return None, "delivery_adapter_unavailable"
        return None, "no_eligible_kernel_source"

    def _attempt(
        self,
        opportunity: KernelOpportunity,
        anchor: E2EMeasurement,
        diagnostic_receipt: ArtifactReceipt,
        configs: tuple[Path, Path, Path],
        anchor_image_id: str | None,
        accepted_stack: tuple[AcceptedCandidate, ...],
    ) -> AcceptedCandidate | None:
        generated = self._generate(opportunity, anchor, diagnostic_receipt)
        if generated is None:
            return None
        attempt_id, candidate, candidate_receipt = generated
        micro = self._micro_gate(
            attempt_id, candidate, candidate_receipt, opportunity
        )
        if micro is None:
            return None
        safety = self._safety_gate(
            attempt_id, candidate, candidate_receipt, opportunity
        )
        if safety is None:
            return None
        qualified = self._deployment_gate(
            attempt_id,
            candidate,
            candidate_receipt,
            opportunity,
            micro,
            safety,
            configs,
            accepted_stack,
        )
        if qualified is None:
            return None
        return self._e2e_gate(qualified, configs[0], anchor_image_id)

    def _generate(
        self,
        opportunity: KernelOpportunity,
        anchor: E2EMeasurement,
        diagnostic_receipt: ArtifactReceipt,
    ) -> tuple[str, E2ECandidate, ArtifactReceipt] | None:
        search = self.record.controller.state.e2e
        assert search is not None and self.worker is not None
        attempt_id = f"attempt-{search.budget.candidates_used + 1}"
        context = self.contexts.compile(
            spec=self.spec,
            record=self.record,
            opportunity=opportunity,
            attempt_id=attempt_id,
            anchor=anchor,
            diagnostic_receipt=diagnostic_receipt,
            qualification_mode=str(
                getattr(self.micro, "qualification_mode", "strict_micro")
            ),
        )
        self.record.controller.select_e2e_opportunity(
            attempt_id=attempt_id,
            opportunity_id=opportunity.opportunity_id,
            context_packet_id=context.compiled.packet.context_packet_id,
        )
        try:
            candidate = self.worker.generate(
                self._candidate_request(attempt_id, opportunity, context.prompt)
            )
        except ApexError:
            raise
        except Exception as error:
            raise ContractError(
                "Candidate generation infrastructure failed",
                "candidate_generation_infrastructure_failed",
                {"error_type": type(error).__name__},
            ) from error
        if candidate.attempt_id != attempt_id:
            raise IntegrityError(
                "Candidate worker returned another attempt",
                "candidate_attempt_mismatch",
                {
                    "expected_attempt_id": attempt_id,
                    "observed_attempt_id": candidate.attempt_id,
                },
            )
        receipt = self.record.record_candidate(candidate)
        if not candidate.succeeded or candidate.candidate_id is None:
            _raise_agent_teardown_infrastructure(candidate, receipt)
            self.record.controller.reject_e2e_execution(
                candidate_id=None,
                receipt=receipt.digest,
                reason=candidate.reason_code,
            )
            commit_e2e_reject(
                self.record,
                attempt_id=attempt_id,
                opportunity_id=opportunity.opportunity_id,
                candidate_id=None,
                candidate_manifest=receipt,
                reason=candidate.reason_code,
            )
            self.record.controller.complete_e2e_update(stop=False, reason=candidate.reason_code)
            return None
        self.record.controller.freeze_e2e_candidate(
            candidate_id=candidate.candidate_id,
            artifact_ref=receipt.digest,
        )
        return attempt_id, candidate, receipt

    def _candidate_request(
        self, attempt_id: str, opportunity: KernelOpportunity, prompt: str
    ) -> E2ECandidateRequest:
        return E2ECandidateRequest(
            run_id=self.record.run_id,
            attempt_id=attempt_id,
            opportunity=opportunity,
            prompt=prompt,
            destination=self.record.root / "worktrees" / attempt_id,
            backend=self.spec.agent_backend,
            model=self.spec.agent_model,
            effort=self.spec.agent_effort,
            max_turns=self.spec.max_turns,
            timeout_seconds=self.spec.agent_timeout_seconds,
        )

    def _micro_gate(
        self,
        attempt_id: str,
        candidate: E2ECandidate,
        candidate_receipt: ArtifactReceipt,
        opportunity: KernelOpportunity,
    ) -> tuple[MicroQualification, ArtifactReceipt] | None:
        result = self.micro.verify(
            MicroQualificationRequest(
                self.record.run_id,
                candidate,
                opportunity,
                self.record.root / "verification" / attempt_id / "micro",
                self.record.controller.state.anchor_generation,
                self.gpu_device_scope,
            )
        )
        receipt = self.record.record_micro(attempt_id, result)
        assert candidate.candidate_id is not None
        self.record.controller.commit_e2e_micro_verification(
            candidate_id=candidate.candidate_id,
            receipt=receipt.digest,
            qualified=result.qualified,
            reason=result.reason_code,
        )
        if result.qualified:
            return result, receipt
        commit_e2e_reject(
            self.record,
            attempt_id=attempt_id,
            opportunity_id=opportunity.opportunity_id,
            candidate_id=_candidate_id(candidate),
            candidate_manifest=candidate_receipt,
            reason=result.reason_code,
            evidence_receipts={"micro_receipt": receipt.digest},
            evidence_artifacts=(("micro_qualification", receipt),),
        )
        self.record.controller.complete_e2e_update(stop=False, reason=result.reason_code)
        return None

    def _safety_gate(
        self,
        attempt_id: str,
        candidate: E2ECandidate,
        candidate_receipt: ArtifactReceipt,
        opportunity: KernelOpportunity,
    ) -> tuple[SafetyQualification, ArtifactReceipt] | None:
        result = self.safety.verify(
            SafetyQualificationRequest(
                self.record.run_id,
                candidate,
                opportunity,
                self.record.root / "verification" / attempt_id / "safety",
                self.record.controller.state.anchor_generation,
            )
        )
        receipt = self.record.record_safety(attempt_id, result)
        reason = result.reason_codes[0] if result.reason_codes else "safety_verified"
        assert candidate.candidate_id is not None
        self.record.controller.commit_e2e_safety_verification(
            candidate_id=candidate.candidate_id,
            receipt=receipt.digest,
            finding=result.finding,
            allowed_to_measure=result.allowed_to_measure,
            promotion_eligible=result.promotion_eligible,
            reason=reason,
        )
        if result.qualified:
            return result, receipt
        commit_e2e_reject(
            self.record,
            attempt_id=attempt_id,
            opportunity_id=opportunity.opportunity_id,
            candidate_id=_candidate_id(candidate),
            candidate_manifest=candidate_receipt,
            reason=reason,
            evidence_receipts={"safety_receipt": receipt.digest},
            evidence_artifacts=(("safety_qualification", receipt),),
        )
        self.record.controller.complete_e2e_update(stop=False, reason=reason)
        return None

    def _deployment_gate(
        self,
        attempt_id: str,
        candidate: E2ECandidate,
        candidate_receipt: ArtifactReceipt,
        opportunity: KernelOpportunity,
        micro_pair: tuple[MicroQualification, ArtifactReceipt],
        safety_pair: tuple[SafetyQualification, ArtifactReceipt],
        configs: tuple[Path, Path, Path],
        accepted_stack: tuple[AcceptedCandidate, ...],
    ) -> _QualifiedAttempt | None:
        safety, safety_receipt = safety_pair
        result = self.deployments.deploy(
            CandidateDeploymentRequest(
                run_id=self.record.run_id,
                candidate=candidate,
                opportunity=opportunity,
                provenance=self.provenance,
                benchmark_measurement=configs[0],
                benchmark_diagnostic=configs[1],
                workload_semantics_sha256=self.views.workload_semantics_sha256,
                artifact_root=self.record.root / "delivery" / attempt_id,
                anchor_generation=self.record.controller.state.anchor_generation,
                safety=safety,
                benchmark_replay=configs[2],
                accepted_stack=accepted_stack,
            )
        )
        _validate_deployment(result, candidate, self.views)
        receipt = self.record.record_delivery(attempt_id, result)
        assert candidate.candidate_id is not None
        if not result.qualified:
            self._reject_deployment(
                attempt_id,
                candidate,
                candidate_receipt,
                opportunity,
                micro_pair,
                safety_receipt,
                result,
                receipt,
            )
            return None
        self.record.controller.commit_e2e_delivery_verification(
            candidate_id=candidate.candidate_id,
            receipt=receipt.digest,
            verified=True,
            reason=result.reason_code,
        )
        micro, micro_receipt = micro_pair
        return _QualifiedAttempt(
            attempt_id,
            opportunity,
            candidate,
            candidate_receipt,
            micro,
            micro_receipt,
            safety,
            safety_receipt,
            result,
            receipt,
        )

    def _reject_deployment(
        self,
        attempt_id: str,
        candidate: E2ECandidate,
        candidate_receipt: ArtifactReceipt,
        opportunity: KernelOpportunity,
        micro_pair: tuple[MicroQualification, ArtifactReceipt],
        safety_receipt: ArtifactReceipt,
        result: CandidateDeployment,
        receipt: ArtifactReceipt,
    ) -> None:
        self.deployments.rollback(result)
        if result.infrastructure_failure:
            raise IntegrityError(
                "Candidate deployment infrastructure failed",
                "deployment_infrastructure_failed",
                {
                    "candidate_id": candidate.candidate_id,
                    "deployment_reason_code": result.reason_code,
                    "delivery_receipt": receipt.digest,
                    "deployment_evidence": dict(result.evidence),
                },
            )
        self.record.controller.commit_e2e_delivery_verification(
            candidate_id=_candidate_id(candidate),
            receipt=receipt.digest,
            verified=False,
            reason=result.reason_code,
        )
        _, micro_receipt = micro_pair
        commit_e2e_reject(
            self.record,
            attempt_id=attempt_id,
            opportunity_id=opportunity.opportunity_id,
            candidate_id=_candidate_id(candidate),
            candidate_manifest=candidate_receipt,
            reason=result.reason_code,
            evidence_receipts={
                "micro_receipt": micro_receipt.digest,
                "safety_receipt": safety_receipt.digest,
                "delivery_receipt": receipt.digest,
            },
            evidence_artifacts=(
                ("micro_qualification", micro_receipt),
                ("safety_qualification", safety_receipt),
                ("primary_delivery", receipt),
            ),
        )
        self.record.controller.complete_e2e_update(
            stop=False,
            reason=result.reason_code,
        )

    def _e2e_gate(
        self,
        attempt: _QualifiedAttempt,
        anchor_config: Path,
        anchor_image_id: str | None,
    ) -> AcceptedCandidate | None:
        candidate_id = _candidate_id(attempt.candidate)
        try:
            result = self.promotions.run(
                attempt_id=attempt.attempt_id,
                candidate_id=candidate_id,
                opportunity_id=attempt.opportunity.opportunity_id,
                anchor_config=anchor_config,
                anchor_image_id=anchor_image_id,
                deployment=attempt.deployment,
            )
        except IntegrityError:
            self.deployments.rollback(attempt.deployment)
            raise
        if result.promotion is None:
            _commit_qualified_reject(
                self.record, attempt, result.evidence_receipt, result.reason_code
            )
            return self._rollback(attempt, result.reason_code)
        return self._decide_promotion(attempt, result.promotion)

    def _decide_promotion(
        self,
        attempt: _QualifiedAttempt,
        promotion: MatchedPromotion,
    ) -> AcceptedCandidate | None:
        candidate_id = _candidate_id(attempt.candidate)
        verdict = promotion.verdict
        candidate_source = attempt.candidate.candidate_source_sha256
        decision = commit_measured_e2e_outcome(
            self.record,
            attempt_id=attempt.attempt_id,
            opportunity_id=attempt.opportunity.opportunity_id,
            candidate_id=candidate_id,
            candidate_manifest=attempt.candidate_receipt,
            verdict=verdict,
            evidence_receipts=_promotion_receipts(attempt, promotion.receipt),
            evidence_artifacts=_promotion_artifacts(attempt, promotion.receipt),
            new_anchor_id=(
                f"anchor-{attempt.deployment.deployed_source_sha256[:16]}"
                if verdict.keep
                else None
            ),
            accepted_patch_id=(
                f"patch-{candidate_source[:16]}"
                if verdict.keep and candidate_source is not None
                else None
            ),
        )
        if not verdict.keep:
            return self._rollback(attempt, verdict.reason_code)
        return AcceptedCandidate(
            attempt.candidate,
            attempt.opportunity,
            attempt.micro,
            attempt.safety,
            attempt.deployment,
            promotion.primary_measurement,
            decision.digest,
        )

    def _rollback(
        self,
        attempt: _QualifiedAttempt,
        reason: str,
    ) -> None:
        self.deployments.rollback(attempt.deployment)
        self.record.controller.complete_e2e_update(stop=False, reason=reason)
        return None

    def _reprofile(
        self,
        config: Path,
        accepted_count: int,
        accepted_sources: frozenset[str],
    ) -> Diagnosis:
        diagnosis = self.session.diagnose(
            f"reprofile-diagnostic-{accepted_count}", config
        )
        eligible = tuple(
            item
            for item in diagnosis.plan.eligible
            if _source_key(item) not in accepted_sources
        )
        self.record.controller.commit_e2e_reprofile(
            receipt=diagnosis.state_receipt.digest,
            opportunity_ids=tuple(item.opportunity_id for item in eligible),
        )
        self.record.controller.complete_e2e_update(
            stop=False,
            reason="no_reprofiled_opportunities" if not eligible else "continue",
        )
        return diagnosis

__all__ = ["E2ESearchLoop", "SearchOutcome"]
