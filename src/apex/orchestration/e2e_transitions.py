"""Pure hierarchical transitions for the E2E kernel-search state machine."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Mapping

from apex.core import StateTransitionError, validate_identifier

from .state import (
    E2ESearchState,
    SearchBudget,
    SearchDecision,
    SearchStage,
    WorkloadState,
)


E2EHandler = Callable[[WorkloadState, Mapping[str, Any], str], WorkloadState]


def _initialized(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    _require_running(state)
    if state.e2e is not None:
        _reject("E2E workload is already initialized", "e2e_already_initialized")
    workload_id = _required_string(payload, "workload_id")
    validate_identifier(workload_id, field_name="workload_id")
    search = E2ESearchState(
        workload_id=workload_id,
        stage=SearchStage.BASELINING,
        state_generation=1,
        cycle=0,
        provenance_hash=_required_string(payload, "provenance_hash"),
        objective_policy_hash=_required_string(payload, "objective_policy_hash"),
        accuracy_contract_hash=_required_string(payload, "accuracy_contract_hash"),
        measurement_protocol_hash=_required_string(payload, "measurement_protocol_hash"),
        budget=SearchBudget(
            candidate_limit=_required_int(payload, "candidate_limit"),
            cycle_limit=_required_int(payload, "cycle_limit"),
        ),
    )
    return replace(state, e2e=search)


def _baseline_committed(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.BASELINING)
    receipt = _required_string(payload, "receipt")
    metrics = _metric_pairs(payload.get("metrics"))
    quality_passed = payload.get("quality_passed")
    if not isinstance(quality_passed, bool):
        _reject("quality_passed must be boolean", "event_field_invalid")
    stage = SearchStage.DIAGNOSING if quality_passed else SearchStage.FINALIZING
    return replace(
        state,
        e2e=_advance(
            search,
            stage=stage,
            baseline_receipt=receipt,
            baseline_metrics=metrics,
            exit_reason=None if quality_passed else "baseline_invalid",
        ),
    )


def _diagnostics_committed(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.DIAGNOSING)
    receipt = _required_string(payload, "receipt")
    opportunities = _opportunity_ids(payload)
    stage = SearchStage.PLANNING if opportunities else SearchStage.FINALIZING
    return replace(
        state,
        e2e=_advance(
            search,
            stage=stage,
            diagnostic_receipt=receipt,
            opportunity_queue=opportunities,
            bottleneck_generation=search.bottleneck_generation + 1,
            exit_reason=None if opportunities else "no_opportunities",
        ),
    )


def _opportunity_selected(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.PLANNING)
    _require_current_generation(state, search, payload)
    attempt = _required_string(payload, "attempt_id")
    validate_identifier(attempt, field_name="attempt_id")
    if attempt in {item.attempt_id for item in search.decisions}:
        _reject("Attempt ID was already used", "attempt_id_reused")
    opportunity = _required_string(payload, "opportunity_id")
    context = _required_string(payload, "context_packet_id")
    if opportunity not in search.opportunity_queue:
        _reject("Opportunity is not in the current queue", "unknown_opportunity")
    if search.budget.candidates_used >= search.budget.candidate_limit:
        _reject("Candidate budget is exhausted", "candidate_budget_exhausted")
    budget = replace(search.budget, candidates_used=search.budget.candidates_used + 1)
    attempts = dict(search.opportunity_attempts)
    attempts[opportunity] = attempts.get(opportunity, 0) + 1
    return replace(
        state,
        e2e=_advance(
            search,
            stage=SearchStage.EXECUTING,
            budget=budget,
            opportunity_attempts=tuple(sorted(attempts.items())),
            active_attempt_id=attempt,
            active_opportunity_id=opportunity,
            context_packet_id=context,
            active_candidate_id=None,
            candidate_artifact_ref=None,
            verification_receipts=(),
            exit_reason=None,
        ),
    )


def _execution_rejected(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.EXECUTING)
    _require_active_attempt(search, payload)
    candidate = _optional_candidate(payload)
    receipt = _required_string(payload, "receipt")
    reason = _required_string(payload, "reason")
    return replace(
        state,
        e2e=_advance(
            search,
            stage=SearchStage.DECIDING,
            active_candidate_id=candidate,
            candidate_artifact_ref=receipt,
            verification_receipts=(receipt,),
            exit_reason=reason,
        ),
    )


def _candidate_frozen(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.EXECUTING)
    _require_active_attempt(search, payload)
    candidate = _required_string(payload, "candidate_id")
    validate_identifier(candidate, field_name="candidate_id")
    return replace(
        state,
        e2e=_advance(
            search,
            stage=SearchStage.MICRO_VERIFYING,
            active_candidate_id=candidate,
            candidate_artifact_ref=_required_string(payload, "artifact_ref"),
        ),
    )


def _micro_verified(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.MICRO_VERIFYING)
    _require_active_attempt(search, payload)
    candidate = _require_active_candidate(search, payload)
    receipt = _required_string(payload, "receipt")
    qualified = payload.get("qualified")
    if not isinstance(qualified, bool):
        _reject("qualified must be boolean", "event_field_invalid")
    if qualified:
        successor = _advance(
            search,
            stage=SearchStage.SAFETY_VERIFYING,
            verification_receipts=(*search.verification_receipts, receipt),
        )
    else:
        reason = _required_string(payload, "reason")
        successor = _advance(
            search,
            stage=SearchStage.DECIDING,
            verification_receipts=(*search.verification_receipts, receipt),
            exit_reason=reason,
        )
    return replace(state, e2e=successor)


def _safety_verified(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.SAFETY_VERIFYING)
    _require_active_attempt(search, payload)
    candidate = _require_active_candidate(search, payload)
    receipt = _required_string(payload, "receipt")
    finding = payload.get("finding")
    allowed = payload.get("allowed_to_measure", True)
    promotion = payload.get("promotion_eligible", True)
    if not all(isinstance(item, bool) for item in (finding, allowed, promotion)):
        _reject("Safety decision fields must be boolean", "event_field_invalid")
    if finding or not allowed or not promotion:
        reason = _required_string(payload, "reason")
        successor = _advance(
            search,
            stage=SearchStage.DECIDING,
            verification_receipts=(*search.verification_receipts, receipt),
            exit_reason=reason,
        )
    else:
        successor = _advance(
            search,
            stage=SearchStage.DELIVERY_VERIFYING,
            verification_receipts=(*search.verification_receipts, receipt),
        )
    return replace(state, e2e=successor)


def _delivery_verified(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.DELIVERY_VERIFYING)
    _require_active_attempt(search, payload)
    candidate = _require_active_candidate(search, payload)
    receipt = _required_string(payload, "receipt")
    verified = payload.get("verified", True)
    if not isinstance(verified, bool):
        _reject("verified must be boolean", "event_field_invalid")
    if verified:
        return replace(
            state,
            e2e=_advance(
                search,
                stage=SearchStage.E2E_VERIFYING,
                verification_receipts=(*search.verification_receipts, receipt),
            ),
        )
    reason = _required_string(payload, "reason")
    return replace(
        state,
        e2e=_advance(
            search,
            stage=SearchStage.DECIDING,
            verification_receipts=(*search.verification_receipts, receipt),
            exit_reason=reason,
        ),
    )


def _candidate_decided(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stages(
        state, {SearchStage.DECIDING, SearchStage.E2E_VERIFYING}
    )
    _require_active_attempt(search, payload)
    _require_current_generation(state, search, payload)
    candidate = _optional_candidate(payload)
    if candidate != search.active_candidate_id:
        _reject("Event targets another E2E candidate", "candidate_id_mismatch")
    receipt = _required_string(payload, "receipt")
    verdict = _required_string(payload, "verdict")
    reason = _required_string(payload, "reason")
    if verdict not in {"keep", "revert", "reject", "needs_more_measurement"}:
        _reject("Invalid E2E candidate verdict", "invalid_search_verdict")
    if search.stage is SearchStage.DECIDING and verdict != "reject":
        _reject("Failed pre-measurement gates require REJECT", "invalid_search_verdict")
    if candidate is None and verdict != "reject":
        _reject("A source-free attempt can only be rejected", "invalid_search_verdict")
    decision = _decision(state, search, candidate, verdict, reason, receipt)
    successor = _advance(
        search,
        stage=SearchStage.REPROFILING if verdict == "keep" else SearchStage.UPDATING,
        verification_receipts=(*search.verification_receipts, receipt),
        decisions=(*search.decisions, decision),
        exit_reason=reason,
    )
    if verdict != "keep":
        return replace(state, e2e=successor)
    anchor = _required_string(payload, "new_anchor_id")
    patch = _required_string(payload, "accepted_patch_id")
    validate_identifier(anchor, field_name="new_anchor_id")
    validate_identifier(patch, field_name="accepted_patch_id")
    return replace(
        state,
        anchor_id=anchor,
        anchor_generation=state.anchor_generation + 1,
        accepted_patch_ids=(*state.accepted_patch_ids, patch),
        e2e=successor,
    )


def _reprofiled(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.REPROFILING)
    opportunities = _opportunity_ids(payload, unique=False)
    return replace(
        state,
        e2e=_advance(
            search,
            stage=SearchStage.UPDATING,
            diagnostic_receipt=_required_string(payload, "receipt"),
            opportunity_queue=opportunities,
            opportunity_attempts=(),
            bottleneck_generation=search.bottleneck_generation + 1,
            active_attempt_id=None,
            active_opportunity_id=None,
            active_candidate_id=None,
            context_packet_id=None,
            candidate_artifact_ref=None,
        ),
    )


def _updated(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.UPDATING)
    stop = payload.get("stop")
    if not isinstance(stop, bool):
        _reject("stop must be boolean", "event_field_invalid")
    kept = bool(search.decisions and search.decisions[-1].verdict == "keep")
    next_cycle = search.cycle + (1 if kept else 0)
    budget = replace(
        search.budget,
        cycles_used=min(next_cycle, search.budget.cycle_limit),
    )
    queue = _retry_queue(search)
    exhausted = (
        next_cycle >= search.budget.cycle_limit
        or budget.candidates_used >= budget.candidate_limit
    )
    stage = SearchStage.FINALIZING if stop or exhausted or not queue else SearchStage.PLANNING
    reason = _required_string(payload, "reason") if stage is SearchStage.FINALIZING else None
    return replace(
        state,
        e2e=_advance(
            search,
            stage=stage,
            cycle=next_cycle,
            budget=budget,
            opportunity_queue=queue,
            active_attempt_id=None,
            active_opportunity_id=None,
            active_candidate_id=None,
            context_packet_id=None,
            candidate_artifact_ref=None,
            verification_receipts=(),
            exit_reason=reason,
        ),
    )


def _finalization_requested(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.PLANNING)
    reason = _required_string(payload, "reason")
    return replace(
        state,
        e2e=_advance(search, stage=SearchStage.FINALIZING, exit_reason=reason),
    )


def _final_committed(
    state: WorkloadState, payload: Mapping[str, Any], _event_type: str
) -> WorkloadState:
    search = _require_stage(state, SearchStage.FINALIZING)
    clean = payload.get("clean_replay_verified", False)
    if not isinstance(clean, bool):
        _reject("clean_replay_verified must be boolean", "event_field_invalid")
    return replace(
        state,
        e2e=_advance(
            search,
            stage=SearchStage.COMPLETED,
            verification_receipts=(
                *search.verification_receipts,
                _required_string(payload, "receipt"),
            ),
            final_clean_replay_verified=clean,
        ),
    )


def _opportunity_ids(
    payload: Mapping[str, Any], *, unique: bool = True
) -> tuple[str, ...]:
    raw = payload.get("opportunity_ids")
    if not isinstance(raw, (list, tuple)):
        _reject("opportunity_ids must be a list", "event_field_invalid")
    values = tuple(str(item) for item in raw)
    if unique and len(set(values)) != len(values):
        _reject("Opportunity IDs must be unique", "duplicate_opportunity")
    for item in values:
        validate_identifier(item, field_name="opportunity_id")
    return values


def _decision(
    state: WorkloadState,
    search: E2ESearchState,
    candidate: str | None,
    verdict: str,
    reason: str,
    receipt: str,
) -> SearchDecision:
    opportunity = search.active_opportunity_id
    if opportunity is None:
        _reject("No E2E opportunity is active", "opportunity_not_active")
    attempt = search.active_attempt_id
    context = search.context_packet_id
    candidate_artifact = search.candidate_artifact_ref
    if attempt is None or context is None or candidate_artifact is None:
        _reject("E2E attempt lineage is incomplete", "attempt_lineage_missing")
    return SearchDecision(
        attempt,
        opportunity,
        candidate,
        verdict,
        reason,
        receipt,
        state.anchor_generation,
        candidate_artifact,
        context,
    )


def _retry_queue(search: E2ESearchState) -> tuple[str, ...]:
    queue = search.opportunity_queue
    active = search.active_opportunity_id
    if active is None:
        return queue
    attempts = dict(search.opportunity_attempts).get(active, 0)
    remaining = tuple(item for item in queue if item != active)
    return (*remaining, active) if attempts < search.budget.cycle_limit else remaining


def _require_stage(state: WorkloadState, stage: SearchStage) -> E2ESearchState:
    _require_running(state)
    if state.pending_action is not None:
        _reject("A side-effecting action is still pending", "pending_action_at_transition")
    search = state.e2e
    if search is None:
        _reject("E2E workload is not initialized", "e2e_not_initialized")
    if search.stage is not stage:
        _reject("E2E search stage transition is illegal", "illegal_e2e_transition")
    return search


def _require_stages(
    state: WorkloadState, stages: set[SearchStage]
) -> E2ESearchState:
    _require_running(state)
    if state.pending_action is not None:
        _reject("A side-effecting action is still pending", "pending_action_at_transition")
    search = state.e2e
    if search is None:
        _reject("E2E workload is not initialized", "e2e_not_initialized")
    if search.stage not in stages:
        _reject("E2E search stage transition is illegal", "illegal_e2e_transition")
    return search


def _advance(search: E2ESearchState, **changes: Any) -> E2ESearchState:
    return replace(search, state_generation=search.state_generation + 1, **changes)


def _require_current_generation(
    state: WorkloadState,
    search: E2ESearchState,
    payload: Mapping[str, Any],
) -> None:
    if _required_string(payload, "parent_anchor_id") != state.anchor_id:
        _reject("Candidate was based on a stale anchor", "stale_anchor")
    if _required_int(payload, "parent_anchor_generation") != state.anchor_generation:
        _reject("Candidate anchor generation is stale", "stale_anchor")
    if _required_int(payload, "state_generation") != search.state_generation:
        _reject("Candidate state generation is stale", "stale_state_generation")


def _require_active_candidate(
    search: E2ESearchState, payload: Mapping[str, Any]
) -> str:
    candidate = _required_string(payload, "candidate_id")
    if candidate != search.active_candidate_id:
        _reject("Event targets another E2E candidate", "candidate_id_mismatch")
    return candidate


def _require_active_attempt(
    search: E2ESearchState, payload: Mapping[str, Any]
) -> str:
    attempt = _required_string(payload, "attempt_id")
    if attempt != search.active_attempt_id:
        _reject("Event targets another E2E attempt", "attempt_id_mismatch")
    return attempt


def _optional_candidate(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("candidate_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        _reject("candidate_id must be a non-empty string", "event_field_invalid")
    validate_identifier(value, field_name="candidate_id")
    return value


def _metric_pairs(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, Mapping) or not value:
        _reject("Baseline metrics must be a non-empty object", "event_field_invalid")
    pairs: list[tuple[str, float]] = []
    for key, raw in value.items():
        try:
            metric = float(raw)
        except (TypeError, ValueError):
            _reject("Baseline metric is not numeric", "event_field_invalid")
        if not math.isfinite(metric):
            _reject("Baseline metric is not finite", "event_field_invalid")
        pairs.append((str(key), metric))
    return tuple(sorted(pairs))


def _require_running(state: WorkloadState) -> None:
    if state.phase.value != "running":
        _reject("Run is not active", "run_not_active")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        _reject(f"Event field {key!r} is required", "event_field_missing")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        _reject(f"Event field {key!r} must be an integer", "event_field_invalid")
    return value


def _reject(message: str, reason_code: str) -> None:
    raise StateTransitionError(message, reason_code=reason_code)


E2E_EVENT_HANDLERS: Mapping[str, E2EHandler] = {
    "e2e.initialized": _initialized,
    "e2e.baseline_committed": _baseline_committed,
    "e2e.diagnostics_committed": _diagnostics_committed,
    "e2e.opportunity_selected": _opportunity_selected,
    "e2e.candidate_frozen": _candidate_frozen,
    "e2e.execution_rejected": _execution_rejected,
    "e2e.micro_verified": _micro_verified,
    "e2e.safety_verified": _safety_verified,
    "e2e.delivery_verified": _delivery_verified,
    "e2e.candidate_decided": _candidate_decided,
    "e2e.reprofiled": _reprofiled,
    "e2e.updated": _updated,
    "e2e.finalization_requested": _finalization_requested,
    "e2e.final_committed": _final_committed,
}


__all__ = ["E2E_EVENT_HANDLERS"]
