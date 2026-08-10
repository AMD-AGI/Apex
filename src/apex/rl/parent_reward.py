"""Replay task-terminal E2E reward truth into the parent RL episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError
from apex.evaluation import (
    E2ERewardPolicy,
    evaluate_paired_current_anchor,
    grade_e2e_outcome,
    load_e2e_paired_measurement,
)
from apex.storage import ArtifactReceipt, ArtifactStore

from .e2e_benchmark_validation import read_json, reject
from .e2e_measurement_validation import load_e2e_acceptance_policy
from .models import EpisodeEvent, EvidenceClass
from .kernel_parent_reward import project_kernel_parent_reward
from .terminal_raw_validation import validate_terminal_raw_evidence


@dataclass(frozen=True, slots=True)
class ParentRewardProjection:
    task_reward: float | None
    reward_vector: Mapping[str, Any] | None
    reward_policy_id: str | None
    reward_policy_digest: str | None
    reward_source_receipt: str | None
    raw_measurement_receipts: tuple[str, ...]
    trainability: str
    untrainable_reason: str | None
    policy_ids: tuple[str, ...]


def project_parent_reward(
    run_id: str,
    events: tuple[EpisodeEvent, ...],
    artifacts: ArtifactStore,
) -> ParentRewardProjection:
    """Project terminal result/reward, preserving explicit reward-null semantics."""

    terminal = _terminal_result(events, artifacts)
    if terminal is None:
        return ParentRewardProjection(None, None, None, None, None, (), "unscored", None, ())
    kind, terminal_result = terminal
    if kind == "kernel_terminal_result":
        projected = project_kernel_parent_reward(
            run_id, events, artifacts, terminal_result
        )
        return ParentRewardProjection(
            projected.task_reward,
            projected.reward_vector,
            projected.reward_policy_id,
            projected.reward_policy_digest,
            projected.reward_source_receipt,
            projected.raw_measurement_receipts,
            projected.trainability,
            projected.untrainable_reason,
            (projected.reward_policy_id,),
        )
    policy = E2ERewardPolicy()
    if (
        terminal_result.get("task_kind") != "e2e_kernel_only"
        or terminal_result.get("reward_policy_id") != policy.policy_id
        or terminal_result.get("reward_policy_digest") != policy.digest
    ):
        reject("Terminal result reward policy is invalid")
    if terminal_result.get("trainability") == "untrainable":
        _validate_reward_null(events, terminal_result)
        return ParentRewardProjection(
            None,
            None,
            policy.policy_id,
            policy.digest,
            None,
            (),
            "untrainable",
            str(terminal_result.get("untrainable_reason")),
            (policy.policy_id,),
        )
    return _project_scored(run_id, events, artifacts, terminal_result, policy)


def _project_scored(
    run_id: str,
    events: tuple[EpisodeEvent, ...],
    artifacts: ArtifactStore,
    result: Mapping[str, Any],
    policy: E2ERewardPolicy,
) -> ParentRewardProjection:
    reward = _terminal_reward_event(events)
    vector_receipt = _single_role(reward, "e2e_reward_vector")
    policy_receipt = _single_role(reward, "reward_policy")
    paired_receipt = _single_role(reward, "terminal_paired_measurement")
    source_receipt = _single_role(reward, "terminal_reward_source")
    vector = read_json(artifacts, vector_receipt, canonical=True)
    stored_policy = read_json(artifacts, policy_receipt, canonical=True)
    measurement_document = read_json(artifacts, paired_receipt, canonical=True)
    source_document = read_json(artifacts, source_receipt, canonical=True)
    try:
        measurement = load_e2e_paired_measurement(measurement_document)
        acceptance = load_e2e_acceptance_policy(events, run_id, artifacts)
        verdict = evaluate_paired_current_anchor(measurement, acceptance)
        grade = grade_e2e_outcome(
            verdict="keep" if verdict.keep else "revert",
            reason_code=verdict.reason_code,
            candidate_present=vector.get("candidate_present") is True,
            measurement_verdict=verdict,
            safety_certified=vector.get("safety_certified") is True,
            scope="task_terminal",
        )
        validate_terminal_raw_evidence(
            reward,
            source_document,
            measurement,
            artifacts,
        )
    except ContractError as error:
        raise IntegrityError(
            "Terminal reward evidence cannot be replayed",
            "e2e_measurement_evidence_mismatch",
        ) from error
    raw = tuple(measurement.raw_measurement_receipts)
    if (
        reward.evidence_class is not EvidenceClass.DERIVED
        or stored_policy != policy.to_dict()
        or vector != grade.to_dict()
        or reward.payload.get("reward_vector") != vector
        or reward.payload.get("scalar_reward") != grade.scalar_reward
        or reward.payload.get("policy_id") != policy.policy_id
        or reward.payload.get("policy_digest") != policy.digest
        or reward.payload.get("reward_source_receipt") != source_receipt.digest
        or reward.payload.get("raw_measurement_receipts") != list(raw)
        or not _result_matches(result, grade, source_receipt, raw)
    ):
        reject("Terminal result, reward, and paired evidence differ")
    return ParentRewardProjection(
        grade.scalar_reward,
        grade.to_dict(),
        policy.policy_id,
        policy.digest,
        source_receipt.digest,
        raw,
        "complete",
        None,
        (policy.policy_id,),
    )


def _terminal_result(
    events: tuple[EpisodeEvent, ...], artifacts: ArtifactStore
) -> tuple[str, Mapping[str, Any]] | None:
    matches = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "delivery_result"
        and event.payload.get("kind")
        in {"e2e_terminal_result", "kernel_terminal_result"}
    )
    if not matches:
        return None
    if len(matches) != 1:
        reject("Parent episode has multiple terminal results")
    kind = str(matches[0].payload["kind"])
    return kind, read_json(
        artifacts,
        _single_role(matches[0], kind),
        canonical=True,
    )


def _terminal_reward_event(events: tuple[EpisodeEvent, ...]) -> EpisodeEvent:
    matches = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "reward_committed"
        and event.payload.get("scope") == "task_terminal"
    )
    if len(matches) != 1:
        reject("Trainable E2E parent lacks one terminal reward")
    return matches[0]


def _validate_reward_null(
    events: tuple[EpisodeEvent, ...], result: Mapping[str, Any]
) -> None:
    reward_events = tuple(
        event
        for event in events
        if event.event_type.replace(".", "_") == "reward_committed"
        and event.payload.get("scope") == "task_terminal"
    )
    if (
        reward_events
        or result.get("task_reward") is not None
        or result.get("reward_vector") is not None
        or result.get("reward_source_receipt") is not None
        or result.get("raw_measurement_receipts") != []
        or not result.get("untrainable_reason")
    ):
        reject("Untrainable E2E parent fabricates terminal reward evidence")


def _result_matches(
    result: Mapping[str, Any],
    grade: Any,
    source: ArtifactReceipt,
    raw: tuple[str, ...],
) -> bool:
    return bool(
        result.get("trainability") == "trainable"
        and result.get("task_reward") == grade.scalar_reward
        and result.get("reward_vector") == grade.to_dict()
        and result.get("reward_source_receipt") == source.digest
        and result.get("raw_measurement_receipts") == list(raw)
        and result.get("untrainable_reason") is None
    )


def _single_role(event: EpisodeEvent, role: str) -> ArtifactReceipt:
    receipts = tuple(item.receipt for item in event.artifacts if item.role == role)
    if len(receipts) != 1:
        reject(f"Terminal reward requires exactly one {role} artifact")
    return receipts[0]


__all__ = ["ParentRewardProjection", "project_parent_reward"]
