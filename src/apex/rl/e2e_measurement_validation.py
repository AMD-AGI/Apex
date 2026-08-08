"""Offline replay of measured E2E acceptance from exact CAS evidence."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, sha256_json
from apex.evaluation import E2EAcceptancePolicy, grade_e2e_outcome
from apex.intake import RegressionGates
from apex.storage import ArtifactStore

from .e2e_benchmark_validation import (
    event_has_role,
    load_delivery,
    mapping,
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
    events = tuple(
        event
        for event in graph.parent.events
        if event.event_type.replace(".", "_") == "dependency_verified"
        and event.payload.get("kind") == "resolved_e2e_run_request"
        and event_has_role(event, "run_request")
    )
    initialized = tuple(
        event
        for event in graph.parent.events
        if event.event_type.replace(".", "_") == "e2e_initialized"
    )
    if len(events) != 1 or len(initialized) != 1:
        reject("Measured E2E episode has no unique frozen run request")
    receipt = single_event_receipt(events[0], "run_request")
    request = read_json(artifacts, receipt, canonical=True)
    spec = mapping(request.get("spec"), "run request spec")
    goal = mapping(spec.get("goal"), "optimization goal")
    gates = mapping(goal.get("gates"), "regression gates")
    if (
        request.get("schema") != "apex.e2e-run-request/v1"
        or request.get("run_id") != graph.run_id
        or goal.get("primary") != "throughput"
        or goal.get("direction") != "maximize"
        or set(gates)
        != {
            "accuracy_regression_pct",
            "ttft_p99_regression_pct",
            "tpot_p99_regression_pct",
        }
        or initialized[0].payload.get("objective_policy_hash") != sha256_json(goal)
    ):
        reject("Frozen E2E objective policy does not match initialization")
    try:
        frozen_gates = RegressionGates(**{key: float(value) for key, value in gates.items()})
        if asdict(frozen_gates) != dict(gates):
            reject("Frozen E2E regression gates are not canonical numeric values")
        return E2EAcceptancePolicy(frozen_gates)
    except (ContractError, TypeError, ValueError) as error:
        raise IntegrityError(
            "Frozen E2E acceptance policy is invalid",
            "e2e_measurement_evidence_mismatch",
        ) from error


__all__ = ["validate_measured_e2e_evidence"]
