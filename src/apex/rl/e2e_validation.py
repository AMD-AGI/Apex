"""Shared fail-closed validation for materialized E2E RL attempts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    validate_identifier,
)
from apex.evaluation import E2ERewardPolicy, replay_e2e_reward
from apex.storage import ArtifactReceipt, ArtifactStore, EventRecord

from .e2e_measurement_validation import validate_measured_e2e_evidence
from .e2e_hard_gate_validation import validate_quality_gate_e2e_evidence
from .models import CandidateEpisode, EpisodeArtifact, EpisodeEvent, EpisodeGraph


E2E_REWARD_POLICY_ID = "e2e_throughput_qos_v1"
MEASURED_E2E_ARTIFACT_ROLES = frozenset(
    {
        "benchmark_config",
        "normalized_benchmark",
        "benchmark_report",
        "quality_evidence",
        "quality_result",
        "primary_delivery",
        "delivery_measurement_config",
        "delivery_diagnostic_config",
        "delivery_replay_config",
    }
)
ATTEMPT_BOUND_E2E_EVENTS = frozenset(
    {
        "e2e_opportunity_selected",
        "e2e_candidate_frozen",
        "e2e_execution_rejected",
        "e2e_micro_verified",
        "e2e_safety_verified",
        "e2e_delivery_verified",
        "e2e_candidate_decided",
    }
)


def artifact_roles(events: Sequence[EpisodeEvent]) -> set[str]:
    """Return all verified artifact roles bound to one child lineage."""

    return {
        artifact.role
        for event in events
        for artifact in event.artifacts
    }


def explicit_attempt_id(record: EventRecord) -> str | None:
    """Read explicit attempt lineage without candidate/action compatibility aliases."""

    value = record.payload.get("attempt_id")
    if value is None and "attempt_id" not in record.payload:
        normalized = record.event_type.replace(".", "_")
        if normalized == "reward_committed" and record.payload.get("scope") == "task_terminal":
            return None
        required = (
            "candidate_id" in record.payload
            or normalized == "reward_committed"
            or normalized in ATTEMPT_BOUND_E2E_EVENTS
        )
        if required:
            raise IntegrityError(
                "Attempt-scoped event has no explicit attempt ID",
                "attempt_lineage_missing",
            )
        return None
    if not isinstance(value, str) or not value:
        raise IntegrityError("Attempt ID is invalid", "attempt_lineage_invalid")
    try:
        return validate_identifier(value, field_name="attempt_id")
    except ContractError as error:
        raise IntegrityError("Attempt ID is invalid", "attempt_lineage_invalid") from error


def optional_identifier(payload: Mapping[str, Any], key: str) -> str | None:
    """Validate an optional identity-bearing payload field."""

    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{key} is invalid", f"{key}_invalid")
    try:
        return validate_identifier(value, field_name=key)
    except ContractError as error:
        raise IntegrityError(f"{key} is invalid", f"{key}_invalid") from error


def e2e_completion_reasons(
    *,
    infrastructure_failure: bool,
    terminal: bool,
    e2e_decision_count: int,
    reward_count: int,
    roles: set[str],
    vector: Mapping[str, Any] | None,
    candidate_id: str | None,
    opportunity_id: str | None,
    verdict: str | None,
    decision_reason: str | None,
) -> set[str]:
    """Return completeness failures for one terminal, non-infra E2E attempt."""

    reasons: set[str] = set()
    if infrastructure_failure or not terminal:
        return reasons
    if e2e_decision_count != 1:
        reasons.add(
            "decision_missing"
            if e2e_decision_count == 0
            else "multiple_decision_events"
        )
    if verdict == "needs_more_measurement":
        if reward_count != 0:
            reasons.add("unexpected_reward_for_untrainable_attempt")
        if "candidate_manifest" not in roles:
            reasons.add("candidate_manifest_receipt_missing")
        if opportunity_id is None:
            reasons.add("opportunity_id_missing")
        reasons.add("reward_null")
        return reasons
    if reward_count != 1:
        reasons.add("reward_missing" if reward_count == 0 else "multiple_reward_events")
    if "candidate_manifest" not in roles:
        reasons.add("candidate_manifest_receipt_missing")
    if "raw_measurement" in roles:
        reasons.add("legacy_raw_measurement_role")
    if opportunity_id is None:
        reasons.add("opportunity_id_missing")
    if not isinstance(vector, Mapping):
        reasons.add("reward_vector_missing")
        return reasons
    candidate_present = vector.get("candidate_present")
    if not isinstance(candidate_present, bool):
        reasons.add("candidate_presence_invalid")
    elif candidate_present:
        if candidate_id is None:
            reasons.add("candidate_id_missing")
        if "candidate_source" not in roles:
            reasons.add("candidate_artifact_missing")
    elif candidate_id is not None:
        reasons.add("candidate_presence_mismatch")
    if vector.get("verdict") != verdict:
        reasons.add("reward_decision_mismatch")
    if decision_reason is not None and vector.get("reason_code") != decision_reason:
        reasons.add("reward_decision_mismatch")
    return reasons


def e2e_outcome_transaction_reasons(
    events: Sequence[EpisodeEvent],
    transaction_members: Mapping[str, Sequence[str]],
) -> set[str]:
    """Prove the E2E decision and reward are one exact SQLite transaction."""

    decisions = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "e2e_candidate_decided"
    )
    rewards = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "reward_committed"
    )
    if len(decisions) != 1 or len(rewards) != 1:
        return set()
    decision = decisions[0]
    reward = rewards[0]
    reasons: set[str] = set()
    if decision.transaction_id != reward.transaction_id:
        reasons.add("e2e_outcome_transaction_mismatch")
    expected_ids = {decision.event_id, reward.event_id}
    for transaction_id in {decision.transaction_id, reward.transaction_id}:
        actual_ids = tuple(transaction_members.get(transaction_id, ()))
        if len(actual_ids) != 2 or set(actual_ids) != expected_ids:
            reasons.add("e2e_outcome_transaction_shape_invalid")
    return reasons


def transaction_members(
    records: Sequence[EventRecord],
) -> Mapping[str, tuple[str, ...]]:
    """Index exact event membership of every verified SQLite transaction."""

    members: dict[str, list[str]] = {}
    for record in records:
        members.setdefault(record.transaction_id, []).append(record.event_id)
    return {key: tuple(value) for key, value in members.items()}


def allows_source_free_e2e(child: CandidateEpisode) -> bool:
    """A rejected attempt with no frozen source is still a valid RL outcome."""

    vector = child.reward_vector
    return (
        E2E_REWARD_POLICY_ID in child.policy_ids
        and child.candidate_id is None
        and isinstance(vector, Mapping)
        and vector.get("candidate_present") is False
        and vector.get("verdict") == "reject"
    )


def validate_e2e_export_reward(
    graph: EpisodeGraph,
    child: CandidateEpisode,
    artifacts: ArtifactStore,
) -> None:
    """Replay scalarization and validate canonical grade/decision/source proof."""

    vector = child.reward_vector
    if vector is None:
        raise IntegrityError("E2E reward vector is missing", "reward_vector_missing")
    try:
        expected = replay_e2e_reward(vector)
        embedded = float(vector["scalar_reward"])
        scalar = float(child.scalar_reward)
    except (ContractError, KeyError, TypeError, ValueError) as error:
        raise IntegrityError(
            "Stored E2E reward cannot be exactly replayed",
            "reward_replay_mismatch",
        ) from error
    if abs(expected - embedded) > 1e-9 or abs(expected - scalar) > 1e-9:
        raise IntegrityError(
            "Stored E2E reward cannot be exactly replayed",
            "reward_replay_mismatch",
        )
    _, policy_document = _read_single_json(child, artifacts, "reward_policy")
    _, grade_document = _read_single_json(child, artifacts, "e2e_reward_vector")
    decision_receipt, decision_document = _read_single_json(
        child, artifacts, "decision_evidence"
    )
    manifest_receipt, manifest_document = _read_single_json(
        child, artifacts, "candidate_manifest"
    )
    if policy_document != E2ERewardPolicy().to_dict() or grade_document != dict(vector):
        raise IntegrityError(
            "E2E reward policy or grade artifact differs from the reward event",
            "e2e_reward_artifact_mismatch",
        )
    _validate_lineage(
        child,
        vector,
        decision_receipt,
        decision_document,
        manifest_receipt,
        manifest_document,
    )
    if vector.get("performance_skipped") == "quality_gate":
        validate_quality_gate_e2e_evidence(
            child,
            artifacts,
            decision_document,
        )
    elif vector.get("verdict") in {"keep", "revert"}:
        validate_measured_e2e_evidence(
            graph,
            child,
            artifacts,
            decision_document,
        )


def _read_single_json(
    child: CandidateEpisode,
    artifacts: ArtifactStore,
    role: str,
) -> tuple[ArtifactReceipt, Mapping[str, Any]]:
    bindings = _unique_role_artifacts(child, role)
    if len(bindings) != 1:
        raise IntegrityError(
            f"E2E reward requires exactly one {role} artifact",
            "e2e_reward_artifact_mismatch",
        )
    receipt = bindings[0].receipt
    raw = artifacts.read_bytes(receipt)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise IntegrityError(
            f"E2E {role} artifact is not JSON",
            "e2e_reward_artifact_mismatch",
        ) from error
    if not isinstance(document, Mapping) or canonical_json_bytes(document) != raw:
        raise IntegrityError(
            f"E2E {role} artifact is not canonical",
            "e2e_reward_artifact_mismatch",
        )
    return receipt, document


def _validate_lineage(
    child: CandidateEpisode,
    vector: Mapping[str, Any],
    decision_receipt: ArtifactReceipt,
    decision: Mapping[str, Any],
    manifest_receipt: ArtifactReceipt,
    manifest: Mapping[str, Any],
) -> None:
    candidate_present = vector.get("candidate_present")
    if not isinstance(candidate_present, bool):
        raise IntegrityError(
            "E2E reward candidate presence is invalid",
            "e2e_reward_artifact_mismatch",
        )
    expected = (child.attempt_id, child.candidate_id)
    if any(
        (document.get("attempt_id"), document.get("candidate_id")) != expected
        for document in (decision, manifest)
    ):
        raise IntegrityError(
            "E2E decision or manifest targets another candidate",
            "e2e_reward_artifact_mismatch",
        )
    _validate_outcome_lineage(child, decision_receipt, decision)
    if decision.get("candidate_manifest_receipt") != manifest_receipt.digest:
        raise IntegrityError(
            "E2E decision targets another candidate manifest",
            "e2e_reward_artifact_mismatch",
        )
    _validate_decision_values(child, vector, decision)
    _validate_candidate_sources(child, candidate_present, manifest)
    if vector.get("verdict") in {"keep", "revert"} and not _unique_role_artifacts(
        child, "normalized_benchmark"
    ):
        raise IntegrityError(
            "Measured E2E verdict has no normalized benchmark",
            "e2e_reward_artifact_mismatch",
        )


def _validate_outcome_lineage(
    child: CandidateEpisode,
    decision_receipt: ArtifactReceipt,
    decision: Mapping[str, Any],
) -> None:
    decision_events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "e2e_candidate_decided"
    )
    reward_events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "reward_committed"
    )
    if len(decision_events) != 1 or len(reward_events) != 1:
        raise IntegrityError(
            "E2E outcome event lineage is ambiguous",
            "e2e_reward_artifact_mismatch",
        )
    expected = child.opportunity_id
    decision_event = decision_events[0]
    reward_event = reward_events[0]
    if (
        expected is None
        or decision.get("opportunity_id") != expected
        or decision_event.payload.get("opportunity_id") != expected
        or reward_event.payload.get("opportunity_id") != expected
        or decision_event.payload.get("receipt") != decision_receipt.digest
        or not any(
            artifact.role == "decision_evidence"
            and artifact.receipt == decision_receipt
            for artifact in decision_event.artifacts
        )
    ):
        raise IntegrityError(
            "Context, decision, and reward opportunity lineage differs",
            "e2e_reward_artifact_mismatch",
        )


def _unique_role_artifacts(
    child: CandidateEpisode, role: str
) -> tuple[EpisodeArtifact, ...]:
    unique: dict[str, EpisodeArtifact] = {}
    for event in child.events:
        for artifact in event.artifacts:
            if artifact.role == role:
                unique.setdefault(artifact.receipt.digest, artifact)
    return tuple(unique[digest] for digest in sorted(unique))


def _validate_decision_values(
    child: CandidateEpisode,
    vector: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    verdict = vector.get("verdict")
    reason = vector.get("reason_code")
    if child.verdict != verdict:
        raise IntegrityError(
            "E2E decision and reward verdict differ",
            "e2e_reward_artifact_mismatch",
        )
    if "verdict" in decision and decision.get("verdict") != verdict:
        raise IntegrityError(
            "E2E decision artifact and reward verdict differ",
            "e2e_reward_artifact_mismatch",
        )
    if "reason_code" in decision and decision.get("reason_code") != reason:
        raise IntegrityError(
            "E2E decision artifact and reward reason differ",
            "e2e_reward_artifact_mismatch",
        )
    if "reason" in decision and decision.get("reason") != reason:
        raise IntegrityError(
            "E2E decision artifact and reward reason differ",
            "e2e_reward_artifact_mismatch",
        )
    if vector.get("performance_skipped") == "quality_gate":
        if (
            verdict != "revert"
            or reason != "quality_gate_failed"
            or decision.get("performance_skipped") != "quality_gate"
            or "measurement_verdict" in decision
        ):
            raise IntegrityError(
                "Quality-gate decision semantics differ from the reward grade",
                "e2e_reward_artifact_mismatch",
            )
        return
    measured = decision.get("measurement_verdict")
    if verdict not in {"keep", "revert"}:
        return
    if not isinstance(measured, Mapping) or measured.get("reason_code") != reason:
        raise IntegrityError(
            "Measured E2E decision evidence differs from the reward grade",
            "e2e_reward_artifact_mismatch",
        )
    if measured.get("keep") is not (verdict == "keep"):
        raise IntegrityError(
            "Measured E2E decision evidence differs from the reward grade",
            "e2e_reward_artifact_mismatch",
        )
    metrics = vector.get("metrics")
    measured_metrics = measured.get("metrics")
    measured_values = (
        {
            **dict(measured_metrics),
            "anchor_measurement_id": measured.get("anchor_measurement_id"),
            "candidate_measurement_id": measured.get("candidate_measurement_id"),
        }
        if isinstance(measured_metrics, Mapping)
        else None
    )
    if not isinstance(metrics, Mapping) or measured_values != dict(metrics):
        raise IntegrityError(
            "Measured E2E decision metrics differ from the reward grade",
            "e2e_reward_artifact_mismatch",
        )
    if measured.get("ratios") != vector.get("ratios"):
        raise IntegrityError(
            "Measured E2E decision ratios differ from the reward grade",
            "e2e_reward_artifact_mismatch",
        )


def _validate_candidate_sources(
    child: CandidateEpisode,
    candidate_present: bool,
    manifest: Mapping[str, Any],
) -> None:
    raw_receipts = manifest.get("source_receipts")
    if not isinstance(raw_receipts, list):
        raise IntegrityError(
            "E2E candidate manifest has invalid source receipts",
            "e2e_reward_artifact_mismatch",
        )
    try:
        declared = tuple(
            ArtifactReceipt.from_dict(dict(item)) for item in raw_receipts
        )
    except (ContractError, TypeError, ValueError) as error:
        raise IntegrityError(
            "E2E candidate manifest has invalid source receipts",
            "e2e_reward_artifact_mismatch",
        ) from error
    observed = tuple(
        artifact.receipt
        for artifact in _unique_role_artifacts(child, "candidate_source")
    )
    declared_documents = tuple(
        receipt.to_dict() for receipt in sorted(declared, key=lambda item: item.digest)
    )
    observed_documents = tuple(receipt.to_dict() for receipt in observed)
    valid_presence = (
        manifest.get("succeeded") is candidate_present
        and (child.candidate_id is not None) is candidate_present
    )
    if (
        len({receipt.digest for receipt in declared}) != len(declared)
        or declared_documents != observed_documents
        or not valid_presence
        or (candidate_present and not observed_documents)
        or (not candidate_present and bool(observed_documents))
    ):
        raise IntegrityError(
            "E2E candidate source lineage differs from its manifest",
            "e2e_reward_artifact_mismatch",
        )


__all__ = [
    "ATTEMPT_BOUND_E2E_EVENTS",
    "E2E_REWARD_POLICY_ID",
    "MEASURED_E2E_ARTIFACT_ROLES",
    "allows_source_free_e2e",
    "artifact_roles",
    "e2e_completion_reasons",
    "e2e_outcome_transaction_reasons",
    "explicit_attempt_id",
    "optional_identifier",
    "transaction_members",
    "validate_e2e_export_reward",
]
