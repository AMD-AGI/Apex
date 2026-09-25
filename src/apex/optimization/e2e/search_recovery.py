"""Stage reconciliation for a CAS-projected interrupted E2E search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apex.core import ContractError, IntegrityError
from apex.evaluation import E2EObservation
from apex.orchestration import SearchStage
from apex.storage import ArtifactReceipt

from .benchmarking import Diagnosis
from .candidate import E2ECandidate
from .outcomes import commit_e2e_reject
from .recovery import recover_uncommitted_diagnosis
from .recovery_search import RecoveredAttempt, RecoveredSearch
from .search_support import (
    candidate_configs,
    candidate_id,
    source_key,
)
from .services import (
    AcceptedCandidate,
    MicroQualification,
    SafetyQualification,
)


@dataclass(slots=True)
class SearchProgress:
    diagnosis: Diagnosis
    accepted: list[AcceptedCandidate]
    accepted_sources: set[str]
    anchor: E2EObservation
    configs: tuple[Path, Path, Path]
    history: list[str]


class SearchRecovery:
    """Drive only durable stage transitions; search policy remains in the loop."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def progress(
        self,
        initial: Diagnosis,
        baseline: E2EObservation,
        recovery: RecoveredSearch | None,
    ) -> SearchProgress:
        if recovery is None:
            return SearchProgress(
                initial,
                [],
                set(),
                baseline,
                (
                    self.loop.views.measurement,
                    self.loop.views.diagnostic,
                    self.loop.views.replay,
                ),
                [str(initial.evidence_path)],
            )
        return SearchProgress(
            recovery.diagnosis,
            list(recovery.accepted),
            {source_key(item.opportunity) for item in recovery.accepted},
            recovery.anchor,
            (
                recovery.measurement_config,
                recovery.diagnostic_config,
                recovery.replay_config,
            ),
            list(recovery.diagnostic_history),
        )

    def reconcile(
        self, progress: SearchProgress, active: RecoveredAttempt | None
    ) -> None:
        while True:
            stage = self._stage()
            if stage in {SearchStage.PLANNING, SearchStage.FINALIZING}:
                return
            if stage in _ACTIVE_STAGES:
                if active is None:
                    raise IntegrityError(
                        "Search stage lacks an active recovery attempt",
                        "recovery_lineage_incomplete",
                    )
                winner = self._resume_attempt(
                    active,
                    progress.configs,
                    (
                        progress.accepted[-1].deployment.deployed_image_id
                        if progress.accepted
                        else None
                    ),
                    tuple(progress.accepted),
                )
                if winner is not None:
                    self.accept(progress, winner)
                active = None
                continue
            if stage is SearchStage.REPROFILING:
                progress.diagnosis = self._resume_reprofile(progress)
                self.append_history(progress, progress.diagnosis)
                continue
            if stage is SearchStage.UPDATING:
                self._resume_update(active)
                active = None
                continue
            raise ContractError(
                f"Search recovery cannot reconcile {stage.value}",
                "resume_stage_unsupported",
            )

    @staticmethod
    def accept(progress: SearchProgress, winner: AcceptedCandidate) -> None:
        if all(
            item.candidate.candidate_id != winner.candidate.candidate_id
            for item in progress.accepted
        ):
            progress.accepted.append(winner)
        progress.accepted_sources.add(source_key(winner.opportunity))
        progress.anchor = winner.primary_measurement
        progress.configs = candidate_configs(winner)

    @staticmethod
    def append_history(progress: SearchProgress, diagnosis: Diagnosis) -> None:
        path = str(diagnosis.evidence_path)
        if not progress.history or progress.history[-1] != path:
            progress.history.append(path)

    def _resume_attempt(
        self,
        recovered: RecoveredAttempt,
        configs: tuple[Path, Path, Path],
        anchor_image_id: str | None,
        accepted_stack: tuple[AcceptedCandidate, ...],
    ) -> AcceptedCandidate | None:
        if self._stage() is SearchStage.DECIDING:
            self._close_recovered_reject(recovered)
            return None
        candidate, receipt = self._resume_generation(recovered)
        if candidate is None or receipt is None:
            return None
        micro = self._resume_micro(recovered, candidate, receipt)
        if micro is None:
            return None
        safety = self._resume_safety(recovered, candidate, receipt)
        if safety is None:
            return None
        qualified = self._resume_delivery(
            recovered,
            candidate,
            receipt,
            micro,
            safety,
            configs,
            accepted_stack,
        )
        if qualified is None:
            return None
        if recovered.promotion is None:
            return self.loop._e2e_gate(qualified, configs[0], anchor_image_id)
        return self.loop._decide_promotion(qualified, recovered.promotion)

    def _resume_generation(
        self, recovered: RecoveredAttempt
    ) -> tuple[E2ECandidate | None, ArtifactReceipt | None]:
        candidate = recovered.candidate
        receipt = recovered.candidate_receipt
        if self._stage() is not SearchStage.EXECUTING:
            return candidate, receipt
        if candidate is None or receipt is None:
            receipt = self._record_interrupted_generation(recovered)
            self._close_execution_reject(
                recovered, receipt, "interrupted_candidate_generation"
            )
            return None, None
        if not candidate.succeeded or candidate.candidate_id is None:
            self._close_execution_reject(
                recovered, receipt, candidate.reason_code
            )
            return None, None
        self.loop.record.controller.freeze_e2e_candidate(
            candidate_id=candidate.candidate_id,
            artifact_ref=receipt.digest,
        )
        return candidate, receipt

    def _close_execution_reject(
        self,
        recovered: RecoveredAttempt,
        receipt: ArtifactReceipt,
        reason: str,
    ) -> None:
        self.loop.record.controller.reject_e2e_execution(
            candidate_id=None, receipt=receipt.digest, reason=reason
        )
        commit_e2e_reject(
            self.loop.record,
            attempt_id=recovered.attempt_id,
            opportunity_id=recovered.opportunity.opportunity_id,
            candidate_id=None,
            candidate_manifest=receipt,
            reason=reason,
        )
        self.loop.record.controller.complete_e2e_update(
            stop=False, reason=reason
        )

    def _resume_micro(
        self,
        recovered: RecoveredAttempt,
        candidate: E2ECandidate,
        candidate_receipt: ArtifactReceipt,
    ) -> tuple[MicroQualification, ArtifactReceipt] | None:
        pair = recovered.micro_pair
        if self._stage() is not SearchStage.MICRO_VERIFYING:
            return pair
        if pair is None:
            return self.loop._micro_gate(
                recovered.attempt_id,
                candidate,
                candidate_receipt,
                recovered.opportunity,
            )
        result, receipt = pair
        self.loop.record.controller.commit_e2e_micro_verification(
            candidate_id=candidate_id(candidate),
            receipt=receipt.digest,
            qualified=result.qualified,
            reason=result.reason_code,
        )
        if not result.qualified:
            self._close_recovered_reject(recovered)
            return None
        return pair

    def _resume_safety(
        self,
        recovered: RecoveredAttempt,
        candidate: E2ECandidate,
        candidate_receipt: ArtifactReceipt,
    ) -> tuple[SafetyQualification, ArtifactReceipt] | None:
        pair = recovered.safety_pair
        if self._stage() is not SearchStage.SAFETY_VERIFYING:
            return pair
        if pair is None:
            return self.loop._safety_gate(
                recovered.attempt_id,
                candidate,
                candidate_receipt,
                recovered.opportunity,
            )
        result, receipt = pair
        reason = result.reason_codes[0] if result.reason_codes else "safety_verified"
        self.loop.record.controller.commit_e2e_safety_verification(
            candidate_id=candidate_id(candidate),
            receipt=receipt.digest,
            finding=result.finding,
            allowed_to_measure=result.allowed_to_measure,
            promotion_eligible=result.promotion_eligible,
            reason=reason,
        )
        if not result.qualified:
            self._close_recovered_reject(recovered)
            return None
        return pair

    def _resume_delivery(
        self,
        recovered: RecoveredAttempt,
        candidate: E2ECandidate,
        candidate_receipt: ArtifactReceipt,
        micro_pair: tuple[MicroQualification, ArtifactReceipt],
        safety_pair: tuple[SafetyQualification, ArtifactReceipt],
        configs: tuple[Path, Path, Path],
        accepted_stack: tuple[AcceptedCandidate, ...],
    ) -> Any:
        pair = recovered.deployment_pair
        if self._stage() is not SearchStage.DELIVERY_VERIFYING:
            return recovered.qualified() if pair is not None else None
        if pair is None:
            return self.loop._deployment_gate(
                recovered.attempt_id,
                candidate,
                candidate_receipt,
                recovered.opportunity,
                micro_pair,
                safety_pair,
                configs,
                accepted_stack,
            )
        result, receipt = pair
        if not result.qualified:
            self.loop._reject_deployment(
                recovered.attempt_id,
                candidate,
                candidate_receipt,
                recovered.opportunity,
                micro_pair,
                safety_pair[1],
                result,
                receipt,
            )
            return None
        self.loop.record.controller.commit_e2e_delivery_verification(
            candidate_id=candidate_id(candidate),
            receipt=receipt.digest,
            verified=True,
            reason=result.reason_code,
        )
        return recovered.qualified()

    def _record_interrupted_generation(
        self, recovered: RecoveredAttempt
    ) -> ArtifactReceipt:
        record = self.loop.record
        manifest = record.put_json(
            {
                "schema_version": 1,
                "attempt_id": recovered.attempt_id,
                "candidate_id": None,
                "succeeded": False,
                "reason_code": "interrupted_candidate_generation",
                "workspace": str(record.root / "worktrees" / recovered.attempt_id),
                "editable_files": [],
                "changed_files": [],
                "baseline_source_sha256": "0" * 64,
                "candidate_source_sha256": None,
                "frozen_sources": [],
                "source_receipts": [],
            }
        )
        record.controller.record_domain_event(
            "candidate_frozen",
            {
                "attempt_id": recovered.attempt_id,
                "anchor_generation": record.controller.state.anchor_generation,
                "split": record.dataset_split,
                "visibility": record.data_visibility,
                "candidate_id": None,
                "succeeded": False,
                "reason_code": "interrupted_candidate_generation",
                "changed_files": [],
                "artifacts": [
                    {"role": "candidate_manifest", "receipt": manifest.to_dict()}
                ],
            },
            idempotency_key=f"attempt.{recovered.attempt_id}.candidate",
        )
        return manifest

    def _close_recovered_reject(self, recovered: RecoveredAttempt) -> None:
        search = self.loop.record.controller.state.e2e
        assert search is not None
        receipt = recovered.candidate_receipt
        if receipt is None:
            raise IntegrityError(
                "Reject recovery lacks candidate evidence",
                "recovery_lineage_incomplete",
            )
        reason = search.exit_reason or "interrupted_before_decision"
        bindings = self._recovered_evidence(recovered)
        if recovered.deployment_pair is not None:
            self.loop.deployments.rollback(recovered.deployment_pair[0])
        commit_e2e_reject(
            self.loop.record,
            attempt_id=recovered.attempt_id,
            opportunity_id=recovered.opportunity.opportunity_id,
            candidate_id=(
                recovered.candidate.candidate_id
                if recovered.candidate is not None
                else None
            ),
            candidate_manifest=receipt,
            reason=reason,
            evidence_receipts={name: item.digest for name, _, item in bindings},
            evidence_artifacts=tuple((role, item) for _, role, item in bindings),
        )
        self.loop.record.controller.complete_e2e_update(
            stop=False, reason=reason
        )

    @staticmethod
    def _recovered_evidence(
        recovered: RecoveredAttempt,
    ) -> tuple[tuple[str, str, ArtifactReceipt], ...]:
        values = (
            ("micro_receipt", "micro_qualification", recovered.micro_pair),
            ("safety_receipt", "safety_qualification", recovered.safety_pair),
            ("delivery_receipt", "primary_delivery", recovered.deployment_pair),
        )
        values = tuple(
            (name, role, pair[1])
            for name, role, pair in values
            if pair is not None
        )
        if recovered.promotion is None:
            return values
        return (*values, (
            "paired_promotion_receipt",
            "paired_promotion",
            recovered.promotion.receipt,
        ))

    def _resume_reprofile(self, progress: SearchProgress) -> Diagnosis:
        recovered = recover_uncommitted_diagnosis(self.loop.record)
        if recovered is None:
            action_id = (
                f"reprofile-diagnostic-{len(progress.accepted)}-resume-"
                f"{self.loop.record.controller.state.sequence}"
            )
            diagnosis = self.loop.session.diagnose(action_id, progress.configs[1])
        else:
            plan, path, evidence, lineage, comparison = recovered
            diagnosis = Diagnosis(plan, path, evidence, lineage, comparison)
        eligible = tuple(
            item
            for item in diagnosis.plan.eligible
            if source_key(item) not in progress.accepted_sources
        )
        self.loop.record.controller.commit_e2e_reprofile(
            receipt=diagnosis.state_receipt.digest,
            opportunity_ids=tuple(item.opportunity_id for item in eligible),
        )
        self.loop.record.controller.complete_e2e_update(
            stop=False,
            reason="no_reprofiled_opportunities" if not eligible else "continue",
        )
        return diagnosis

    def _resume_update(self, active: RecoveredAttempt | None) -> None:
        search = self.loop.record.controller.state.e2e
        assert search is not None
        decision = search.decisions[-1] if search.decisions else None
        if (
            decision is not None
            and decision.verdict != "keep"
            and active is not None
            and active.deployment_pair is not None
        ):
            self.loop.deployments.rollback(active.deployment_pair[0])
        self.loop.record.controller.complete_e2e_update(
            stop=False, reason=search.exit_reason or "continue"
        )

    def _stage(self) -> SearchStage:
        search = self.loop.record.controller.state.e2e
        if search is None:
            raise ContractError("E2E state is absent", "e2e_not_initialized")
        return search.stage


_ACTIVE_STAGES = {
    SearchStage.EXECUTING,
    SearchStage.MICRO_VERIFYING,
    SearchStage.SAFETY_VERIFYING,
    SearchStage.DELIVERY_VERIFYING,
    SearchStage.E2E_VERIFYING,
    SearchStage.DECIDING,
}


__all__ = ["SearchProgress", "SearchRecovery"]
