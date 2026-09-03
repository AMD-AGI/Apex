"""Immutable controller state for a single optimization workload."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from apex.core import ContractError, validate_identifier


class RunPhase(str, Enum):
    NEW = "new"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionStatus(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    ARTIFACTS_READY = "artifacts_ready"
    VERIFIED = "verified"
    COMMITTED = "committed"
    FAILED = "failed"
    ABORTED = "aborted"


class SearchStage(str, Enum):
    """Hierarchical E2E stage; independent of the run terminal phase."""

    BASELINING = "baselining"
    DIAGNOSING = "diagnosing"
    PLANNING = "planning"
    EXECUTING = "executing"
    MICRO_VERIFYING = "micro_verifying"
    SAFETY_VERIFYING = "safety_verifying"
    DELIVERY_VERIFYING = "delivery_verifying"
    E2E_VERIFYING = "e2e_verifying"
    DECIDING = "deciding"
    REPROFILING = "reprofiling"
    UPDATING = "updating"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Deterministic search budget accounting stored in authoritative state."""

    candidate_limit: int
    cycle_limit: int
    candidates_used: int = 0
    cycles_used: int = 0

    def __post_init__(self) -> None:
        if min(self.candidate_limit, self.cycle_limit) < 1:
            raise ContractError("Search budget limits must be positive", "invalid_search_budget")
        if not 0 <= self.candidates_used <= self.candidate_limit:
            raise ContractError("Candidate budget accounting is invalid", "invalid_search_budget")
        if not 0 <= self.cycles_used <= self.cycle_limit:
            raise ContractError("Cycle budget accounting is invalid", "invalid_search_budget")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SearchBudget":
        try:
            return cls(
                candidate_limit=int(value["candidate_limit"]),
                cycle_limit=int(value["cycle_limit"]),
                candidates_used=int(value.get("candidates_used", 0)),
                cycles_used=int(value.get("cycles_used", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed search budget", "invalid_search_budget") from error


@dataclass(frozen=True, slots=True)
class SearchDecision:
    attempt_id: str
    opportunity_id: str
    candidate_id: str | None
    verdict: str
    reason: str
    evidence_ref: str
    anchor_generation: int
    candidate_artifact_ref: str
    context_packet_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.attempt_id, field_name="attempt_id")
        validate_identifier(self.opportunity_id, field_name="opportunity_id")
        if self.candidate_id is not None:
            validate_identifier(self.candidate_id, field_name="candidate_id")
        if self.verdict not in {"keep", "revert", "reject", "needs_more_measurement"}:
            raise ContractError("Unknown search verdict", "invalid_search_verdict")
        if (
            not self.reason
            or not self.evidence_ref
            or not self.candidate_artifact_ref
            or not self.context_packet_id
            or self.anchor_generation < 0
        ):
            raise ContractError("Search decision is incomplete", "invalid_search_decision")
        if self.candidate_id is None and self.verdict != "reject":
            raise ContractError("Source-free decision must reject", "invalid_search_decision")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SearchDecision":
        try:
            return cls(
                attempt_id=str(value["attempt_id"]),
                opportunity_id=str(value["opportunity_id"]),
                candidate_id=_optional_string(value.get("candidate_id")),
                verdict=str(value["verdict"]),
                reason=str(value["reason"]),
                evidence_ref=str(value["evidence_ref"]),
                anchor_generation=int(value["anchor_generation"]),
                candidate_artifact_ref=str(value["candidate_artifact_ref"]),
                context_packet_id=str(value["context_packet_id"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed search decision", "invalid_search_decision") from error


@dataclass(frozen=True, slots=True)
class E2ESearchState:
    """Replayable E2E memory; agent conversations are deliberately absent."""

    workload_id: str
    stage: SearchStage
    state_generation: int
    cycle: int
    provenance_hash: str
    objective_policy_hash: str
    accuracy_contract_hash: str
    measurement_protocol_hash: str
    budget: SearchBudget
    baseline_receipt: str | None = None
    baseline_metrics: tuple[tuple[str, float], ...] = ()
    diagnostic_receipt: str | None = None
    opportunity_queue: tuple[str, ...] = ()
    opportunity_attempts: tuple[tuple[str, int], ...] = ()
    bottleneck_generation: int = 0
    active_attempt_id: str | None = None
    active_opportunity_id: str | None = None
    active_candidate_id: str | None = None
    context_packet_id: str | None = None
    candidate_artifact_ref: str | None = None
    verification_receipts: tuple[str, ...] = ()
    decisions: tuple[SearchDecision, ...] = ()
    exit_reason: str | None = None
    final_clean_replay_verified: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.workload_id, field_name="workload_id")
        if self.state_generation < 1 or self.cycle < 0 or self.bottleneck_generation < 0:
            raise ContractError("E2E generation counters are invalid", "invalid_e2e_state")
        for value in (
            self.provenance_hash,
            self.objective_policy_hash,
            self.accuracy_contract_hash,
            self.measurement_protocol_hash,
        ):
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ContractError("E2E contract hashes must be SHA-256", "invalid_e2e_contract_hash")
        attempts = dict(self.opportunity_attempts)
        if (
            len(attempts) != len(self.opportunity_attempts)
            or self.opportunity_attempts != tuple(sorted(self.opportunity_attempts))
            or any(count < 1 or count > self.budget.cycle_limit for count in attempts.values())
        ):
            raise ContractError("Opportunity attempt accounting is invalid", "invalid_e2e_state")
        for opportunity_id in attempts:
            validate_identifier(opportunity_id, field_name="opportunity_id")
        for opportunity_id in self.opportunity_queue:
            validate_identifier(opportunity_id, field_name="opportunity_id")
        active_lineage = (
            self.active_attempt_id,
            self.active_opportunity_id,
            self.context_packet_id,
        )
        if any(item is None for item in active_lineage) != all(
            item is None for item in active_lineage
        ):
            raise ContractError("Active attempt lineage is partial", "invalid_e2e_state")
        if self.active_attempt_id is not None:
            validate_identifier(self.active_attempt_id, field_name="attempt_id")
            assert self.active_opportunity_id is not None
            validate_identifier(self.active_opportunity_id, field_name="opportunity_id")
        if self.active_candidate_id is not None:
            validate_identifier(self.active_candidate_id, field_name="candidate_id")
        decision_attempts = tuple(item.attempt_id for item in self.decisions)
        if len(set(decision_attempts)) != len(decision_attempts):
            raise ContractError("Attempt decisions must be unique", "invalid_e2e_state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "stage": self.stage.value,
            "state_generation": self.state_generation,
            "cycle": self.cycle,
            "provenance_hash": self.provenance_hash,
            "objective_policy_hash": self.objective_policy_hash,
            "accuracy_contract_hash": self.accuracy_contract_hash,
            "measurement_protocol_hash": self.measurement_protocol_hash,
            "budget": self.budget.to_dict(),
            "baseline_receipt": self.baseline_receipt,
            "baseline_metrics": dict(self.baseline_metrics),
            "diagnostic_receipt": self.diagnostic_receipt,
            "opportunity_queue": list(self.opportunity_queue),
            "opportunity_attempts": dict(self.opportunity_attempts),
            "bottleneck_generation": self.bottleneck_generation,
            "active_attempt_id": self.active_attempt_id,
            "active_opportunity_id": self.active_opportunity_id,
            "active_candidate_id": self.active_candidate_id,
            "context_packet_id": self.context_packet_id,
            "candidate_artifact_ref": self.candidate_artifact_ref,
            "verification_receipts": list(self.verification_receipts),
            "decisions": [item.to_dict() for item in self.decisions],
            "exit_reason": self.exit_reason,
            "final_clean_replay_verified": self.final_clean_replay_verified,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "E2ESearchState":
        try:
            metrics = value.get("baseline_metrics", {})
            if not isinstance(metrics, Mapping):
                raise TypeError("baseline metrics")
            return cls(
                workload_id=str(value["workload_id"]),
                stage=SearchStage(str(value["stage"])),
                state_generation=int(value["state_generation"]),
                cycle=int(value["cycle"]),
                provenance_hash=str(value["provenance_hash"]),
                objective_policy_hash=str(value["objective_policy_hash"]),
                accuracy_contract_hash=str(value["accuracy_contract_hash"]),
                measurement_protocol_hash=str(value["measurement_protocol_hash"]),
                budget=SearchBudget.from_dict(value["budget"]),
                baseline_receipt=_optional_string(value.get("baseline_receipt")),
                baseline_metrics=tuple(sorted((str(k), float(v)) for k, v in metrics.items())),
                diagnostic_receipt=_optional_string(value.get("diagnostic_receipt")),
                opportunity_queue=tuple(str(item) for item in value.get("opportunity_queue", ())),
                opportunity_attempts=tuple(
                    sorted(
                        (str(key), int(item))
                        for key, item in dict(value.get("opportunity_attempts", {})).items()
                    )
                ),
                bottleneck_generation=int(value.get("bottleneck_generation", 0)),
                active_attempt_id=_optional_string(value.get("active_attempt_id")),
                active_opportunity_id=_optional_string(value.get("active_opportunity_id")),
                active_candidate_id=_optional_string(value.get("active_candidate_id")),
                context_packet_id=_optional_string(value.get("context_packet_id")),
                candidate_artifact_ref=_optional_string(value.get("candidate_artifact_ref")),
                verification_receipts=tuple(str(item) for item in value.get("verification_receipts", ())),
                decisions=tuple(SearchDecision.from_dict(item) for item in value.get("decisions", ())),
                exit_reason=_optional_string(value.get("exit_reason")),
                final_clean_replay_verified=value.get("final_clean_replay_verified") is True,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed E2E search state", "invalid_e2e_state") from error


@dataclass(frozen=True, slots=True)
class ActionState:
    """Immutable lifecycle record for one side-effecting action."""

    action_id: str
    action_type: str
    status: ActionStatus
    parent_anchor_id: str
    parent_anchor_generation: int
    artifact_refs: tuple[str, ...] = ()
    verification_id: str | None = None
    result_anchor_id: str | None = None
    accepted_patch_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["artifact_refs"] = list(self.artifact_refs)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionState":
        try:
            return cls(
                action_id=str(value["action_id"]),
                action_type=str(value["action_type"]),
                status=ActionStatus(str(value["status"])),
                parent_anchor_id=str(value["parent_anchor_id"]),
                parent_anchor_generation=int(value["parent_anchor_generation"]),
                artifact_refs=tuple(str(item) for item in value.get("artifact_refs", ())),
                verification_id=_optional_string(value.get("verification_id")),
                result_anchor_id=_optional_string(value.get("result_anchor_id")),
                accepted_patch_id=_optional_string(value.get("accepted_patch_id")),
                error=_optional_string(value.get("error")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed action state", "invalid_action_state") from error


@dataclass(frozen=True, slots=True)
class WorkloadState:
    """Complete replayable state; only a reducer may construct successors."""

    schema_version: int
    run_id: str
    phase: RunPhase
    sequence: int
    last_event_id: str | None
    anchor_id: str
    anchor_generation: int
    accepted_patch_ids: tuple[str, ...]
    pending_action: ActionState | None
    action_history: tuple[ActionState, ...]
    stop_reason: str | None = None
    e2e: E2ESearchState | None = None

    @classmethod
    def initial(cls, run_id: str) -> "WorkloadState":
        validate_identifier(run_id, field_name="run_id")
        return cls(2, run_id, RunPhase.NEW, 0, None, "anchor-0", 0, (), None, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "phase": self.phase.value,
            "sequence": self.sequence,
            "last_event_id": self.last_event_id,
            "anchor_id": self.anchor_id,
            "anchor_generation": self.anchor_generation,
            "accepted_patch_ids": list(self.accepted_patch_ids),
            "pending_action": self.pending_action.to_dict() if self.pending_action else None,
            "action_history": [action.to_dict() for action in self.action_history],
            "stop_reason": self.stop_reason,
            "e2e": self.e2e.to_dict() if self.e2e else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkloadState":
        try:
            pending = value.get("pending_action")
            state = cls(
                schema_version=int(value["schema_version"]),
                run_id=str(value["run_id"]),
                phase=RunPhase(str(value["phase"])),
                sequence=int(value["sequence"]),
                last_event_id=_optional_string(value.get("last_event_id")),
                anchor_id=str(value["anchor_id"]),
                anchor_generation=int(value["anchor_generation"]),
                accepted_patch_ids=tuple(str(item) for item in value.get("accepted_patch_ids", ())),
                pending_action=ActionState.from_dict(pending) if pending is not None else None,
                action_history=tuple(ActionState.from_dict(item) for item in value.get("action_history", ())),
                stop_reason=_optional_string(value.get("stop_reason")),
                e2e=E2ESearchState.from_dict(value["e2e"]) if value.get("e2e") is not None else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Malformed workload state", "invalid_workload_state") from error
        if state.schema_version != 2 or state.sequence < 0 or state.anchor_generation < 0:
            raise ContractError("Unsupported workload state", "invalid_workload_state")
        validate_identifier(state.run_id, field_name="run_id")
        return state


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ActionState",
    "ActionStatus",
    "E2ESearchState",
    "RunPhase",
    "SearchBudget",
    "SearchDecision",
    "SearchStage",
    "WorkloadState",
]
