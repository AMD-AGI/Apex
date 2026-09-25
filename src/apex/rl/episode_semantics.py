"""Pure event semantics used by episode-graph materialization."""

from __future__ import annotations

from typing import Sequence

from apex.core import ContractError
from apex.storage import EventRecord

from .models import EvidenceClass, SemanticRole


def semantic_role(event_type: str) -> SemanticRole:
    normalized = event_type.replace(".", "_")
    if normalized in {"tool_called", "tool_result"}:
        return SemanticRole.TOOL
    if "reward" in normalized:
        return SemanticRole.REWARD
    if "cost" in normalized or normalized in {"usage_recorded"}:
        return SemanticRole.COST
    if normalized in {"error", "run_failed", "action_failed", "agent_failed"}:
        return SemanticRole.FAILURE
    if normalized in {
        "decision", "e2e_candidate_decided", "action_committed", "action_aborted"
    }:
        return SemanticRole.DECISION
    if normalized in {
        "compile_result",
        "correctness_result",
        "safety_result",
        "measurement_result",
        "e2e_result",
        "delivery_verified",
        "action_verified",
        "e2e_micro_verified",
        "e2e_safety_verified",
        "e2e_delivery_verified",
        "experience_measured",
    }:
        return SemanticRole.OUTCOME
    if normalized in {
        "observation_created",
        "context_packet_created",
        "knowledge_read",
        "knowledge_outcome_linked",
        "experience_deferred",
        "e2e_baseline_committed",
        "e2e_diagnostics_committed",
        "e2e_reprofiled",
    }:
        return SemanticRole.OBSERVATION
    if normalized in {
        "prompt_sent",
        "agent_message",
        "candidate_materialized",
        "candidate_frozen",
        "delivery_materialized",
        "action_queued",
        "action_started",
        "action_artifacts_ready",
        "e2e_candidate_frozen",
    }:
        return SemanticRole.ACTION
    return SemanticRole.CONTROL


def evidence_class(value: object) -> EvidenceClass:
    if value is None:
        return EvidenceClass.UNSPECIFIED
    try:
        return EvidenceClass(str(value))
    except ValueError as error:
        raise ContractError("Unknown evidence class", "invalid_evidence_class") from error


def parent_status(records: Sequence[EventRecord]) -> str:
    terminal = records[-1].event_type.replace(".", "_")
    if terminal in {"run_succeeded", "run_finished"}:
        return str(records[-1].payload.get("status", "succeeded"))
    if terminal == "run_failed":
        return "failed"
    if terminal == "run_cancelled":
        return "cancelled"
    return "incomplete"


def decision_from_type(event_type: str) -> str | None:
    normalized = event_type.replace(".", "_")
    if normalized == "action_committed":
        return "keep"
    if normalized == "action_aborted":
        return "revert"
    return None


def text(value: object) -> str | None:
    return None if value is None or not str(value).strip() else str(value)


__all__ = [
    "decision_from_type",
    "evidence_class",
    "parent_status",
    "semantic_role",
    "text",
]
