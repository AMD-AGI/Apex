"""Offline replay for trusted E2E quality-gate reward outcomes."""

from __future__ import annotations

from typing import Any, Mapping

from apex.evaluation import grade_e2e_outcome
from apex.storage import ArtifactStore

from .e2e_benchmark_validation import (
    load_delivery,
    load_quality_gate_failure_bundle,
    reject,
    validate_failed_candidate_runtime,
)
from .models import CandidateEpisode, EpisodeEvent


def validate_quality_gate_e2e_evidence(
    child: CandidateEpisode,
    artifacts: ArtifactStore,
    decision: Mapping[str, Any],
) -> None:
    """Recompute a quality-stopped REVERT and its runtime-only reward."""

    normalized_digest = decision.get("benchmark_receipt")
    event = _quality_failure_event(child, normalized_digest)
    bundle = load_quality_gate_failure_bundle(event, artifacts)
    delivery = load_delivery(child, artifacts)
    validate_failed_candidate_runtime(bundle, delivery)
    vector = child.reward_vector
    safety_certified = (
        vector.get("safety_certified") if isinstance(vector, Mapping) else None
    )
    if not isinstance(safety_certified, bool):
        reject("Quality-gate reward has invalid safety semantics")
    grade = grade_e2e_outcome(
        verdict="revert",
        reason_code="quality_gate_failed",
        candidate_present=True,
        safety_certified=safety_certified,
        performance_skipped="quality_gate",
    )
    expected_lineage = {
        "attempt_id": child.attempt_id,
        "candidate_id": child.candidate_id,
        "opportunity_id": child.opportunity_id,
        "anchor_generation": child.anchor_generation,
    }
    if (
        any(event.payload.get(key) != value for key, value in expected_lineage.items())
        or decision.get("performance_skipped") != "quality_gate"
        or decision.get("verdict") != "revert"
        or decision.get("reason") != "quality_gate_failed"
        or child.verdict != "revert"
        or vector != grade.to_dict()
        or child.scalar_reward != grade.scalar_reward
    ):
        reject("Quality-gate decision or reward differs from raw CAS replay")


def _quality_failure_event(
    child: CandidateEpisode,
    normalized_digest: object,
) -> EpisodeEvent:
    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "measurement_result"
        and any(
            artifact.role == "normalized_benchmark"
            and artifact.receipt.digest == normalized_digest
            for artifact in event.artifacts
        )
    )
    if not isinstance(normalized_digest, str) or len(events) != 1:
        reject("Quality-gate decision lacks one bound failed measurement")
    return events[0]


__all__ = ["validate_quality_gate_e2e_evidence"]
