"""Offline replay of measured E2E acceptance from exact CAS evidence."""

from __future__ import annotations

import re
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError
from apex.evaluation import E2EAcceptancePolicy, E2ERewardContract, grade_e2e_outcome
from apex.storage import ArtifactStore

from .e2e_benchmark_validation import (
    event_has_role,
    load_delivery,
    read_json,
    reject,
    single_event_receipt,
)
from .e2e_promotion_validation import replay_matched_promotion
from .models import CandidateEpisode, EpisodeGraph


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def validate_measured_e2e_evidence(
    graph: EpisodeGraph,
    child: CandidateEpisode,
    artifacts: ArtifactStore,
    decision: Mapping[str, Any],
) -> None:
    """Recompute a v2 matched KEEP/REVERT and reward from raw CAS evidence."""

    protocol_hash = _protocol_hash(graph)
    acceptance_policy = _acceptance_policy(graph, artifacts)
    delivery = load_delivery(child, artifacts)
    promotion = replay_matched_promotion(
        run_id=graph.run_id,
        child=child,
        artifacts=artifacts,
        protocol_hash=protocol_hash,
        acceptance_policy=acceptance_policy,
        delivery=delivery,
        decision=decision,
    )
    verdict = promotion.verdict
    expected_verdict = "keep" if verdict.keep else "revert"
    grade = grade_e2e_outcome(
        verdict=expected_verdict,
        reason_code=verdict.reason_code,
        candidate_present=True,
        measurement_verdict=verdict,
    )
    if (
        child.verdict != expected_verdict
        or decision.get("verdict") != expected_verdict
        or decision.get("reason") != verdict.reason_code
        or child.reward_vector != grade.to_dict()
        or child.scalar_reward != grade.scalar_reward
    ):
        reject("Measured E2E decision or reward differs from CAS replay")


def _protocol_hash(graph: EpisodeGraph) -> str:
    events = tuple(
        event
        for event in graph.parent.events
        if event.event_type.replace(".", "_") == "e2e_initialized"
    )
    if len(events) != 1:
        reject("Measured E2E episode has no unique protocol declaration")
    value = events[0].payload.get("measurement_protocol_hash")
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        reject("Measured E2E protocol declaration is invalid")
    return value


def _acceptance_policy(
    graph: EpisodeGraph,
    artifacts: ArtifactStore,
) -> E2EAcceptancePolicy:
    return load_e2e_acceptance_policy(graph.parent.events, graph.run_id, artifacts)


def load_e2e_acceptance_policy(
    parent_events: tuple[Any, ...],
    run_id: str,
    artifacts: ArtifactStore,
) -> E2EAcceptancePolicy:
    """Rebuild the frozen E2E acceptance policy from parent evidence."""

    events = tuple(
        event
        for event in parent_events
        if event.event_type.replace(".", "_") == "dependency_verified"
        and event.payload.get("kind") == "resolved_e2e_run_request"
        and event_has_role(event, "e2e_reward_contract")
    )
    initialized = tuple(
        event
        for event in parent_events
        if event.event_type.replace(".", "_") == "e2e_initialized"
    )
    if len(events) != 1 or len(initialized) != 1:
        reject("Measured E2E episode has no unique frozen reward contract")
    receipt = single_event_receipt(events[0], "e2e_reward_contract")
    document = read_json(artifacts, receipt, canonical=True)
    try:
        contract = E2ERewardContract.from_mapping(document)
    except (ContractError, TypeError, ValueError) as error:
        raise IntegrityError(
            "Frozen E2E reward contract is invalid",
            "e2e_measurement_evidence_mismatch",
        ) from error
    initialized_payload = initialized[0].payload
    if (
        contract.run_id != run_id
        or initialized_payload.get("objective_policy_hash")
        != contract.objective_policy_hash
        or initialized_payload.get("measurement_protocol_hash")
        != contract.measurement_protocol_hash
    ):
        reject("Frozen E2E reward contract does not match initialization")
    return contract.acceptance_policy


__all__ = ["load_e2e_acceptance_policy", "validate_measured_e2e_evidence"]
