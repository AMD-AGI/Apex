"""RunController: the sole writer of replayable workload state."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Sequence

from apex.core import (
    ContractError,
    StateTransitionError,
    validate_identifier,
)

from .atomic import AtomicJournalPort, ProposedEvent, append_reduced_transaction
from .e2e_reward_validation import validate_e2e_reward_payload
from .replay import replay_workload_state
from .state import E2ESearchState, RunPhase, WorkloadState
from .transitions import DOMAIN_EVENT_TYPES, EventLike, reduce_event


class JournalPort(AtomicJournalPort, Protocol):
    def append(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
        parent_event_id: str | None = None,
    ) -> EventLike: ...

    def iter_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        verify: bool = True,
    ) -> Iterable[EventLike]: ...

    def verify_run(self, run_id: str) -> None: ...


class SnapshotLike(Protocol):
    high_water_mark: int
    payload: Mapping[str, Any]


class SnapshotPort(Protocol):
    def save(self, *, high_water_mark: int, payload: Mapping[str, Any]) -> SnapshotLike: ...

    def load(self) -> SnapshotLike | None: ...

    def delete(self) -> None: ...


class RunController:
    """Append an event, reduce it, then publish a disposable snapshot."""

    def __init__(
        self,
        journal: JournalPort,
        snapshots: SnapshotPort,
        state: WorkloadState,
    ) -> None:
        self._journal = journal
        self._snapshots = snapshots
        self._state = state

    @property
    def state(self) -> WorkloadState:
        return self._state

    @classmethod
    def create(
        cls,
        run_id: str,
        journal: JournalPort,
        snapshots: SnapshotPort,
        *,
        initial_anchor_id: str = "anchor-0",
    ) -> "RunController":
        validate_identifier(run_id, field_name="run_id")
        validate_identifier(initial_anchor_id, field_name="initial_anchor_id")
        if tuple(journal.iter_events(run_id)):
            raise ContractError("Run already exists", "run_already_exists")
        controller = cls(journal, snapshots, WorkloadState.initial(run_id))
        controller._record(
            "run.started",
            {"initial_anchor_id": initial_anchor_id},
            idempotency_key="run.started",
        )
        return controller

    @classmethod
    def recover(
        cls,
        run_id: str,
        journal: JournalPort,
        snapshots: SnapshotPort,
    ) -> "RunController":
        validate_identifier(run_id, field_name="run_id")
        journal.verify_run(run_id)
        events = tuple(journal.iter_events(run_id))
        if not events:
            raise ContractError("Run does not exist", "run_not_found")
        state = replay_workload_state(run_id, events)
        controller = cls(journal, snapshots, state)
        controller._save_snapshot()
        return controller

    def queue_action(
        self,
        action_id: str,
        action_type: str,
        *,
        parent_anchor_id: str | None = None,
        parent_anchor_generation: int | None = None,
    ) -> WorkloadState:
        payload = {
            "action_id": action_id,
            "action_type": action_type,
            "parent_anchor_id": (
                self.state.anchor_id if parent_anchor_id is None else parent_anchor_id
            ),
            "parent_anchor_generation": (
                self.state.anchor_generation
                if parent_anchor_generation is None
                else parent_anchor_generation
            ),
        }
        return self._record("action.queued", payload, f"action.{action_id}.queued")

    def start_action(self, action_id: str) -> WorkloadState:
        return self._record(
            "action.started",
            {"action_id": action_id},
            f"action.{action_id}.started",
        )

    def mark_artifacts_ready(
        self,
        action_id: str,
        artifact_refs: Sequence[str],
    ) -> WorkloadState:
        return self._record(
            "action.artifacts_ready",
            {"action_id": action_id, "artifact_refs": list(artifact_refs)},
            f"action.{action_id}.artifacts_ready",
        )

    def verify_action(self, action_id: str, verification_id: str) -> WorkloadState:
        return self._record(
            "action.verified",
            {"action_id": action_id, "verification_id": verification_id},
            f"action.{action_id}.verified",
        )

    def commit_action(
        self,
        action_id: str,
        *,
        new_anchor_id: str,
        accepted_patch_id: str,
    ) -> WorkloadState:
        return self._record(
            "action.committed",
            {
                "action_id": action_id,
                "new_anchor_id": new_anchor_id,
                "accepted_patch_id": accepted_patch_id,
            },
            f"action.{action_id}.committed",
        )

    def complete_action(self, action_id: str) -> WorkloadState:
        """Commit a verified non-anchor action such as benchmark or analysis."""

        return self._record(
            "action.completed",
            {"action_id": action_id},
            f"action.{action_id}.completed",
        )

    def record_domain_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> WorkloadState:
        """Append allowlisted evidence without granting it control-state authority."""

        if event_type not in DOMAIN_EVENT_TYPES:
            raise ContractError("Unknown domain evidence event", "unknown_domain_event")
        return self._record(event_type, payload, idempotency_key)

    def initialize_e2e(
        self,
        *,
        workload_id: str,
        provenance_hash: str,
        objective_policy_hash: str,
        accuracy_contract_hash: str,
        measurement_protocol_hash: str,
        candidate_limit: int,
        cycle_limit: int,
    ) -> WorkloadState:
        return self._record(
            "e2e.initialized",
            {
                "workload_id": workload_id,
                "provenance_hash": provenance_hash,
                "objective_policy_hash": objective_policy_hash,
                "accuracy_contract_hash": accuracy_contract_hash,
                "measurement_protocol_hash": measurement_protocol_hash,
                "candidate_limit": candidate_limit,
                "cycle_limit": cycle_limit,
            },
            "e2e.initialized",
        )

    def commit_e2e_baseline(
        self,
        *,
        receipt: str,
        metrics: Mapping[str, float],
        quality_passed: bool,
    ) -> WorkloadState:
        return self._record(
            "e2e.baseline_committed",
            {"receipt": receipt, "metrics": dict(metrics), "quality_passed": quality_passed},
            "e2e.baseline_committed",
        )

    def commit_e2e_diagnostics(
        self, *, receipt: str, opportunity_ids: Sequence[str]
    ) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.diagnostics_committed",
            {"receipt": receipt, "opportunity_ids": list(opportunity_ids)},
            f"e2e.diagnostics.{search.bottleneck_generation + 1}",
        )

    def select_e2e_opportunity(
        self, *, attempt_id: str, opportunity_id: str, context_packet_id: str
    ) -> WorkloadState:
        search = self._e2e()
        if attempt_id in {item.attempt_id for item in search.decisions}:
            raise StateTransitionError(
                "Attempt ID was already used",
                "attempt_id_reused",
            )
        return self._record(
            "e2e.opportunity_selected",
            {
                "attempt_id": attempt_id,
                "opportunity_id": opportunity_id,
                "context_packet_id": context_packet_id,
                **self._generation_payload(search),
            },
            f"e2e.attempt.{attempt_id}.selected",
        )

    def freeze_e2e_candidate(self, *, candidate_id: str, artifact_ref: str) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.candidate_frozen",
            {
                **self._active_attempt_payload(search),
                "candidate_id": candidate_id,
                "artifact_ref": artifact_ref,
            },
            f"e2e.candidate.{search.budget.candidates_used}.frozen",
        )

    def reject_e2e_execution(
        self, *, candidate_id: str | None, receipt: str, reason: str
    ) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.execution_rejected",
            {
                **self._active_attempt_payload(search),
                **({"candidate_id": candidate_id} if candidate_id is not None else {}),
                "receipt": receipt,
                "reason": reason,
            },
            f"e2e.candidate.{search.budget.candidates_used}.execution_rejected",
        )

    def commit_e2e_micro_verification(
        self, *, candidate_id: str, receipt: str, qualified: bool, reason: str = "qualified"
    ) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.micro_verified",
            {
                **self._active_attempt_payload(search),
                "candidate_id": candidate_id,
                "receipt": receipt,
                "qualified": qualified,
                "reason": reason,
            },
            f"e2e.candidate.{search.budget.candidates_used}.micro",
        )

    def commit_e2e_safety_verification(
        self,
        *,
        candidate_id: str,
        receipt: str,
        finding: bool,
        allowed_to_measure: bool = True,
        promotion_eligible: bool = True,
        reason: str = "no_finding",
    ) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.safety_verified",
            {
                **self._active_attempt_payload(search),
                "candidate_id": candidate_id,
                "receipt": receipt,
                "finding": finding,
                "allowed_to_measure": allowed_to_measure,
                "promotion_eligible": promotion_eligible,
                "reason": reason,
            },
            f"e2e.candidate.{search.budget.candidates_used}.safety",
        )

    def commit_e2e_delivery_verification(
        self,
        *,
        candidate_id: str,
        receipt: str,
        verified: bool = True,
        reason: str = "delivery_verified",
    ) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.delivery_verified",
            {
                **self._active_attempt_payload(search),
                "candidate_id": candidate_id,
                "receipt": receipt,
                "verified": verified,
                "reason": reason,
            },
            f"e2e.candidate.{search.budget.candidates_used}.delivery",
        )

    def decide_e2e_candidate(
        self,
        *,
        candidate_id: str | None,
        receipt: str,
        verdict: str,
        reason: str,
        reward_payload: Mapping[str, Any],
        decision_artifacts: Sequence[Mapping[str, Any]] = (),
        new_anchor_id: str | None = None,
        accepted_patch_id: str | None = None,
    ) -> WorkloadState:
        search = self._e2e()
        payload: dict[str, Any] = {
            **self._active_attempt_payload(search),
            "receipt": receipt,
            "verdict": verdict,
            "reason": reason,
            **self._generation_payload(search),
            "artifacts": [dict(item) for item in decision_artifacts],
        }
        if candidate_id is not None:
            payload["candidate_id"] = candidate_id
        if new_anchor_id is not None:
            payload["new_anchor_id"] = new_anchor_id
        if accepted_patch_id is not None:
            payload["accepted_patch_id"] = accepted_patch_id
        lineage = self._active_attempt_payload(search)
        reward = dict(reward_payload)
        for key, expected in lineage.items():
            observed = reward.get(key)
            if observed is not None and observed != expected:
                raise ContractError(
                    "Reward targets another E2E attempt",
                    "attempt_id_mismatch",
                )
            reward[key] = expected
        if candidate_id is not None:
            observed_candidate = reward.get("candidate_id")
            if observed_candidate is not None and observed_candidate != candidate_id:
                raise ContractError(
                    "Reward targets another E2E candidate",
                    "candidate_id_mismatch",
                )
            reward["candidate_id"] = candidate_id
        elif reward.get("candidate_id") is not None:
            raise ContractError(
                "Reward claims a candidate for a source-free decision",
                "candidate_id_mismatch",
            )
        validate_e2e_reward_payload(
            reward,
            verdict=verdict,
            reason=reason,
            candidate_present=candidate_id is not None,
        )
        attempt = str(lineage["attempt_id"])
        return self._record_transaction(
            (
                ("reward_committed", reward, f"e2e.attempt.{attempt}.reward"),
                (
                    "e2e.candidate_decided",
                    payload,
                    f"e2e.attempt.{attempt}.decision",
                ),
            )
        )

    def decide_e2e_untrainable(
        self,
        *,
        candidate_id: str,
        receipt: str,
        reason: str,
        decision_artifacts: Sequence[Mapping[str, Any]] = (),
    ) -> WorkloadState:
        """Commit an inconclusive decision without fabricating a reward event."""

        search = self._e2e()
        attempt = str(self._active_attempt_payload(search)["attempt_id"])
        return self._record(
            "e2e.candidate_decided",
            {
                **self._active_attempt_payload(search),
                "candidate_id": candidate_id,
                "receipt": receipt,
                "verdict": "needs_more_measurement",
                "reason": reason,
                "trainability": "untrainable",
                "untrainable_reason": reason,
                **self._generation_payload(search),
                "artifacts": [dict(item) for item in decision_artifacts],
            },
            f"e2e.attempt.{attempt}.decision",
        )

    def commit_e2e_reprofile(
        self, *, receipt: str, opportunity_ids: Sequence[str]
    ) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.reprofiled",
            {"receipt": receipt, "opportunity_ids": list(opportunity_ids)},
            f"e2e.reprofile.{search.bottleneck_generation + 1}",
        )

    def complete_e2e_update(self, *, stop: bool, reason: str = "continue") -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.updated",
            {"stop": stop, "reason": reason},
            f"e2e.update.{search.budget.candidates_used}.{search.state_generation}",
        )

    def request_e2e_finalization(self, *, reason: str) -> WorkloadState:
        search = self._e2e()
        return self._record(
            "e2e.finalization_requested",
            {"reason": reason},
            f"e2e.finalization.{search.state_generation}",
        )

    def commit_e2e_final(
        self, *, receipt: str, clean_replay_verified: bool = False
    ) -> WorkloadState:
        return self._record(
            "e2e.final_committed",
            {
                "receipt": receipt,
                "clean_replay_verified": clean_replay_verified,
            },
            "e2e.final_committed",
        )

    def fail_action(self, action_id: str, error: str) -> WorkloadState:
        return self._record(
            "action.failed",
            {"action_id": action_id, "error": error},
            f"action.{action_id}.failed",
        )

    def abort_pending(self, reason: str) -> WorkloadState:
        action = self.state.pending_action
        if action is None:
            raise StateTransitionError("No action is pending", "action_not_pending")
        return self._record(
            "action.aborted",
            {"action_id": action.action_id, "reason": reason},
            f"action.{action.action_id}.aborted",
        )

    def finish(self, phase: RunPhase, *, reason: str) -> WorkloadState:
        event_types = {
            RunPhase.SUCCEEDED: "run.succeeded",
            RunPhase.FAILED: "run.failed",
            RunPhase.CANCELLED: "run.cancelled",
        }
        if phase not in event_types:
            raise ContractError("Finish phase must be terminal", "invalid_terminal_phase")
        return self._record(event_types[phase], {"reason": reason}, f"run.{phase.value}")

    def rebuild_snapshot(self) -> WorkloadState:
        self._journal.verify_run(self.state.run_id)
        events = tuple(self._journal.iter_events(self.state.run_id))
        self._state = replay_workload_state(self.state.run_id, events)
        self._save_snapshot()
        return self._state

    def _record(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> WorkloadState:
        existing = self._journal.get_by_idempotency_key(self.state.run_id, idempotency_key)
        if existing is None:
            proposal = ProposedEvent(
                self.state.sequence + 1,
                "proposed-event",
                self.state.run_id,
                event_type,
                payload,
                self.state.last_event_id,
            )
            reduce_event(self.state, proposal)
        event = self._journal.append(
            run_id=self.state.run_id,
            event_type=event_type,
            payload=payload,
            idempotency_key=idempotency_key,
            parent_event_id=self.state.last_event_id if existing is None else existing.parent_event_id,
        )
        if event.sequence <= self.state.sequence:
            return self.state
        self._state = reduce_event(self.state, event)
        self._save_snapshot()
        return self.state

    def _record_transaction(
        self,
        values: Sequence[tuple[str, Mapping[str, Any], str]],
    ) -> WorkloadState:
        """Validate and atomically append a short causal event chain."""

        successor = append_reduced_transaction(self._journal, self.state, values)
        if successor.sequence > self.state.sequence:
            self._state = successor
            self._save_snapshot()
        return self.state

    def _save_snapshot(self) -> None:
        self._snapshots.save(
            high_water_mark=self.state.sequence,
            payload=self.state.to_dict(),
        )

    def _e2e(self) -> E2ESearchState:
        if self.state.e2e is None:
            raise ContractError("E2E workload is not initialized", "e2e_not_initialized")
        return self.state.e2e

    def _generation_payload(self, search: E2ESearchState) -> dict[str, Any]:
        return {
            "parent_anchor_id": self.state.anchor_id,
            "parent_anchor_generation": self.state.anchor_generation,
            "state_generation": search.state_generation,
        }

    @staticmethod
    def _active_attempt_payload(search: E2ESearchState) -> dict[str, str]:
        if search.active_attempt_id is None or search.active_opportunity_id is None:
            raise ContractError("No E2E attempt is active", "attempt_not_active")
        return {
            "attempt_id": search.active_attempt_id,
            "opportunity_id": search.active_opportunity_id,
        }
__all__ = ["JournalPort", "RunController", "SnapshotLike", "SnapshotPort"]
