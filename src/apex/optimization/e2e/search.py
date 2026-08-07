"""Bounded kernel-candidate search against the current live E2E anchor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apex.benchmark import BenchmarkConfigViews
from apex.core import ApexError, ContractError
from apex.evaluation import E2EAcceptancePolicy, E2EMeasurement, evaluate_current_anchor
from apex.intake import E2EOptimizeSpec
from apex.orchestration import SearchStage
from apex.runtime import RunProvenance
from apex.storage import ArtifactReceipt

from .benchmarking import Diagnosis, E2EBenchmarkSession
from .candidate import CandidateWorker, E2ECandidate, E2ECandidateRequest
from .context import E2EContextBuilder
from .kernel_lane import KernelOpportunity
from .run_record import E2ERunRecord
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


@dataclass(frozen=True, slots=True)
class _QualifiedAttempt:
    attempt_id: str
    opportunity: KernelOpportunity
    candidate: E2ECandidate
    micro: MicroQualification
    micro_receipt: ArtifactReceipt
    safety: SafetyQualification
    safety_receipt: ArtifactReceipt
    deployment: CandidateDeployment
    delivery_receipt: ArtifactReceipt


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

    def run(self, initial: Diagnosis, baseline: E2EMeasurement) -> SearchOutcome:
        opportunities = _opportunity_map(initial)
        diagnosis = initial
        accepted: list[AcceptedCandidate] = []
        accepted_sources: set[str] = set()
        anchor = baseline
        configs = (self.views.measurement, self.views.diagnostic, self.views.replay)
        history = [str(initial.evidence_path)]
        while _stage(self.record) is SearchStage.PLANNING:
            opportunity, reason = self._select_available(opportunities)
            if opportunity is None:
                self.record.controller.request_e2e_finalization(
                    reason=reason or "no_eligible_kernel_source"
                )
                break
            winner = self._attempt(
                opportunity,
                anchor,
                diagnosis.evidence_receipt,
                configs,
            )
            if winner is None:
                continue
            accepted.append(winner)
            accepted_sources.add(_source_key(winner.opportunity))
            anchor = winner.primary_measurement
            configs = _candidate_configs(winner)
            diagnosis = self._reprofile(
                configs[1], len(accepted), frozenset(accepted_sources)
            )
            history.append(str(diagnosis.evidence_path))
            opportunities = _opportunity_map(diagnosis)
        return SearchOutcome(tuple(accepted), anchor, *configs, tuple(history))

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
    ) -> AcceptedCandidate | None:
        generated = self._generate(opportunity, anchor, diagnostic_receipt)
        if generated is None:
            return None
        attempt_id, candidate = generated
        micro = self._micro_gate(attempt_id, candidate, opportunity)
        if micro is None:
            return None
        safety = self._safety_gate(attempt_id, candidate, opportunity)
        if safety is None:
            return None
        qualified = self._deployment_gate(
            attempt_id,
            candidate,
            opportunity,
            micro,
            safety,
            configs,
        )
        if qualified is None:
            return None
        return self._e2e_gate(qualified, anchor)

    def _generate(
        self,
        opportunity: KernelOpportunity,
        anchor: E2EMeasurement,
        diagnostic_receipt: ArtifactReceipt,
    ) -> tuple[str, E2ECandidate] | None:
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
            opportunity_id=opportunity.opportunity_id,
            context_packet_id=context.compiled.packet.context_packet_id,
        )
        try:
            candidate = self.worker.generate(self._candidate_request(attempt_id, opportunity, context.prompt))
            receipt = self.record.record_candidate(candidate)
        except ApexError as error:
            self._reject_generation(attempt_id, error.reason_code)
            return None
        if not candidate.succeeded or candidate.candidate_id is None:
            self.record.controller.reject_e2e_execution(
                candidate_id=attempt_id,
                receipt=receipt.digest,
                reason=candidate.reason_code,
            )
            self.record.controller.complete_e2e_update(stop=False, reason=candidate.reason_code)
            return None
        self.record.controller.freeze_e2e_candidate(
            candidate_id=candidate.candidate_id,
            artifact_ref=receipt.digest,
        )
        return attempt_id, candidate

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

    def _reject_generation(self, attempt_id: str, reason: str) -> None:
        receipt = self.record.put_json(
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "reason_code": reason,
                "stage": "candidate_generation",
            }
        )
        self.record.controller.reject_e2e_execution(
            candidate_id=attempt_id,
            receipt=receipt.digest,
            reason=reason,
        )
        self.record.controller.complete_e2e_update(stop=False, reason=reason)

    def _micro_gate(
        self,
        attempt_id: str,
        candidate: E2ECandidate,
        opportunity: KernelOpportunity,
    ) -> tuple[MicroQualification, ArtifactReceipt] | None:
        result = self.micro.verify(
            MicroQualificationRequest(
                self.record.run_id,
                candidate,
                opportunity,
                self.record.root / "verification" / attempt_id / "micro",
                self.record.controller.state.anchor_generation,
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
        self.record.controller.complete_e2e_update(stop=False, reason=result.reason_code)
        return None

    def _safety_gate(
        self,
        attempt_id: str,
        candidate: E2ECandidate,
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
        self.record.controller.complete_e2e_update(stop=False, reason=reason)
        return None

    def _deployment_gate(
        self,
        attempt_id: str,
        candidate: E2ECandidate,
        opportunity: KernelOpportunity,
        micro_pair: tuple[MicroQualification, ArtifactReceipt],
        safety_pair: tuple[SafetyQualification, ArtifactReceipt],
        configs: tuple[Path, Path, Path],
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
            )
        )
        _validate_deployment(result, candidate, self.views)
        receipt = self.record.record_delivery(attempt_id, result)
        assert candidate.candidate_id is not None
        self.record.controller.commit_e2e_delivery_verification(
            candidate_id=candidate.candidate_id,
            receipt=receipt.digest,
            verified=result.qualified,
            reason=result.reason_code,
        )
        if not result.qualified:
            self.deployments.rollback(result)
            self.record.controller.complete_e2e_update(stop=False, reason=result.reason_code)
            return None
        micro, micro_receipt = micro_pair
        return _QualifiedAttempt(
            attempt_id,
            opportunity,
            candidate,
            micro,
            micro_receipt,
            safety,
            safety_receipt,
            result,
            receipt,
        )

    def _e2e_gate(
        self, attempt: _QualifiedAttempt, anchor: E2EMeasurement
    ) -> AcceptedCandidate | None:
        result, measurement, receipt = self.session.measure(
            f"candidate-measurement-{attempt.attempt_id}",
            attempt.deployment.measurement_config,
        )
        candidate_id = _candidate_id(attempt.candidate)
        if not result.succeeded or measurement is None:
            return self._revert(attempt, receipt, "candidate_e2e_measurement_failed")
        verdict = evaluate_current_anchor(
            anchor,
            measurement,
            E2EAcceptancePolicy(self.spec.goal.gates),
        )
        decision = self.record.record_decision(
            attempt.attempt_id,
            candidate_id=candidate_id,
            verdict="keep" if verdict.keep else "revert",
            reason=verdict.reason_code,
            evidence=_decision_evidence(attempt, receipt, verdict.to_dict()),
        )
        if not verdict.keep:
            return self._revert(attempt, decision, verdict.reason_code, recorded=True)
        assert attempt.candidate.candidate_source_sha256 is not None
        self.record.controller.decide_e2e_candidate(
            candidate_id=candidate_id,
            receipt=decision.digest,
            verdict="keep",
            reason=verdict.reason_code,
            new_anchor_id=f"anchor-{attempt.deployment.deployed_source_sha256[:16]}",
            accepted_patch_id=f"patch-{attempt.candidate.candidate_source_sha256[:16]}",
        )
        return AcceptedCandidate(
            attempt.candidate,
            attempt.opportunity,
            attempt.micro,
            attempt.safety,
            attempt.deployment,
            measurement,
            decision.digest,
        )

    def _revert(
        self,
        attempt: _QualifiedAttempt,
        receipt: ArtifactReceipt,
        reason: str,
        *,
        recorded: bool = False,
    ) -> None:
        candidate_id = _candidate_id(attempt.candidate)
        decision = receipt
        if not recorded:
            decision = self.record.record_decision(
                attempt.attempt_id,
                candidate_id=candidate_id,
                verdict="revert",
                reason=reason,
                evidence={
                    "schema_version": 1,
                    "candidate_id": candidate_id,
                    "benchmark_receipt": receipt.digest,
                    "verdict": "revert",
                },
            )
        self.deployments.rollback(attempt.deployment)
        self.record.controller.decide_e2e_candidate(
            candidate_id=candidate_id,
            receipt=decision.digest,
            verdict="revert",
            reason=reason,
        )
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
            receipt=diagnosis.evidence_receipt.digest,
            opportunity_ids=tuple(item.opportunity_id for item in eligible),
        )
        self.record.controller.complete_e2e_update(
            stop=False,
            reason="no_reprofiled_opportunities" if not eligible else "continue",
        )
        return diagnosis


def _opportunity_map(diagnosis: Diagnosis) -> dict[str, KernelOpportunity]:
    return {item.opportunity_id: item for item in diagnosis.plan.opportunities}


def _candidate_configs(candidate: AcceptedCandidate) -> tuple[Path, Path, Path]:
    deployment = candidate.deployment
    return (
        deployment.measurement_config,
        deployment.diagnostic_config,
        deployment.replay_config,
    )


def _candidate_id(candidate: E2ECandidate) -> str:
    if candidate.candidate_id is None:
        raise ContractError("Candidate is not frozen", "invalid_frozen_candidate")
    return candidate.candidate_id


def _source_key(opportunity: KernelOpportunity) -> str:
    if opportunity.source_root is None or opportunity.source_path is None:
        raise ContractError("Kernel source is unresolved", "source_unresolved")
    root = opportunity.source_root.resolve(strict=True)
    relative = opportunity.source_path.resolve(strict=True).relative_to(root)
    return f"{root}:{relative.as_posix()}"


def _stage(record: E2ERunRecord) -> SearchStage:
    search = record.controller.state.e2e
    if search is None:
        raise ContractError("E2E state is not initialized", "e2e_not_initialized")
    return search.stage


def _validate_deployment(
    deployment: CandidateDeployment,
    candidate: E2ECandidate,
    views: BenchmarkConfigViews,
) -> None:
    if candidate.candidate_id != deployment.candidate_id:
        raise ContractError("Deployment targets another candidate", "candidate_id_mismatch")
    if deployment.workload_semantics_sha256 != views.workload_semantics_sha256:
        raise ContractError("Deployment changed workload semantics", "benchmark_semantics_changed")
    if not deployment.deployed:
        return
    if deployment.deployed_source_sha256 != candidate.candidate_source_sha256:
        raise ContractError("Deployed source differs from frozen candidate", "candidate_lineage_mismatch")
    for path in (
        deployment.measurement_config,
        deployment.diagnostic_config,
        deployment.replay_config,
    ):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ContractError("Deployment config is missing or unsafe", "invalid_replay_config")


def _decision_evidence(
    attempt: _QualifiedAttempt,
    benchmark_receipt: ArtifactReceipt,
    verdict: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "candidate_id": _candidate_id(attempt.candidate),
        "opportunity_id": attempt.opportunity.opportunity_id,
        "micro_receipt": attempt.micro_receipt.digest,
        "safety_receipt": attempt.safety_receipt.digest,
        "delivery_receipt": attempt.delivery_receipt.digest,
        "benchmark_receipt": benchmark_receipt.digest,
        "verdict": dict(verdict),
    }


__all__ = ["E2ESearchLoop", "SearchOutcome"]
