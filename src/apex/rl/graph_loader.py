"""Strict typed loading for an exported :class:`EpisodeGraph`."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

from apex.core import IntegrityError
from apex.storage import ArtifactReceipt

from .models import (
    CandidateEpisode,
    EpisodeArtifact,
    EpisodeEvent,
    EpisodeGraph,
    EvidenceClass,
    ParentEpisode,
    SemanticRole,
)


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GRAPH_KEYS = {
    "schema_name", "schema_version", "run_id", "high_water_mark",
    "journal_head_event_id", "workload_state_hash", "parent", "children",
    "provenance", "policy_ids",
}
_PARENT_KEYS = {
    "episode_id", "kind", "run_id", "workload_id", "task_id", "events",
    "child_episode_ids", "terminal_status", "task_reward", "reward_vector",
    "reward_policy_id", "reward_policy_digest", "reward_source_receipt",
    "raw_measurement_receipts", "trainability", "untrainable_reason",
}
_CHILD_KEYS = {
    "episode_id", "parent_episode_id", "attempt_id", "candidate_id",
    "opportunity_id", "task_id", "kernel_id", "state_generation",
    "anchor_generation", "context_packet_id", "context_packet_receipt", "events",
    "status", "verdict", "scalar_reward", "reward_vector", "policy_ids", "split",
    "visibility", "trainability", "validation_reasons",
}
_EVENT_KEYS = {
    "sequence", "event_id", "transaction_id", "parent_event_id", "event_type",
    "semantic_role", "evidence_class", "causation_id", "correlation_id",
    "agent_run_id", "payload", "artifacts",
}
_ARTIFACT_KEYS = {"role", "event_id", "receipt"}
_RECEIPT_KEYS = {"digest", "size", "media_type", "relative_path"}


def load_episode_graph(value: Mapping[str, Any]) -> EpisodeGraph:
    """Decode only the exact canonical graph schema used by Apex projections."""

    document = _mapping(value, "episode graph")
    _exact(document, _GRAPH_KEYS, "episode graph")
    if document["schema_name"] != "apex.episode_graph":
        _reject("Episode graph schema is invalid")
    parent = _load_parent(_mapping(document["parent"], "parent episode"))
    children = tuple(
        _load_child(_mapping(item, "child episode"))
        for item in _list(document["children"], "children")
    )
    try:
        return EpisodeGraph(
            schema_version=_integer(document["schema_version"], "schema_version", 1),
            run_id=_text(document["run_id"], "run_id"),
            high_water_mark=_integer(document["high_water_mark"], "high_water_mark", 1),
            journal_head_event_id=_text(
                document["journal_head_event_id"], "journal_head_event_id"
            ),
            workload_state_hash=_optional_digest(
                document["workload_state_hash"], "workload_state_hash"
            ),
            parent=parent,
            children=children,
            provenance=dict(_mapping(document["provenance"], "provenance")),
            policy_ids=_strings(document["policy_ids"], "policy_ids"),
        )
    except Exception as error:
        if isinstance(error, IntegrityError):
            raise
        raise IntegrityError(
            "Episode graph cannot be decoded", "showcase_episode_invalid"
        ) from error


def _load_parent(value: Mapping[str, Any]) -> ParentEpisode:
    _exact(value, _PARENT_KEYS, "parent episode")
    return ParentEpisode(
        episode_id=_text(value["episode_id"], "parent episode_id"),
        kind=_text(value["kind"], "parent kind"),
        run_id=_text(value["run_id"], "parent run_id"),
        workload_id=_optional_text(value["workload_id"], "workload_id"),
        task_id=_optional_text(value["task_id"], "task_id"),
        events=_events(value["events"]),
        child_episode_ids=_strings(value["child_episode_ids"], "child_episode_ids"),
        terminal_status=_text(value["terminal_status"], "terminal_status"),
        task_reward=_optional_number(value["task_reward"], "task_reward"),
        reward_vector=_optional_mapping(value["reward_vector"], "reward_vector"),
        reward_policy_id=_optional_text(value["reward_policy_id"], "reward_policy_id"),
        reward_policy_digest=_optional_digest(
            value["reward_policy_digest"], "reward_policy_digest"
        ),
        reward_source_receipt=_optional_digest(
            value["reward_source_receipt"], "reward_source_receipt"
        ),
        raw_measurement_receipts=_digests(
            value["raw_measurement_receipts"], "raw_measurement_receipts"
        ),
        trainability=_text(value["trainability"], "parent trainability"),
        untrainable_reason=_optional_text(
            value["untrainable_reason"], "untrainable_reason"
        ),
    )


def _load_child(value: Mapping[str, Any]) -> CandidateEpisode:
    _exact(value, _CHILD_KEYS, "child episode")
    context = value["context_packet_receipt"]
    return CandidateEpisode(
        episode_id=_text(value["episode_id"], "child episode_id"),
        parent_episode_id=_text(value["parent_episode_id"], "parent_episode_id"),
        attempt_id=_text(value["attempt_id"], "attempt_id"),
        candidate_id=_optional_text(value["candidate_id"], "candidate_id"),
        opportunity_id=_optional_text(value["opportunity_id"], "opportunity_id"),
        task_id=_optional_text(value["task_id"], "child task_id"),
        kernel_id=_optional_text(value["kernel_id"], "kernel_id"),
        state_generation=_optional_integer(value["state_generation"], "state_generation"),
        anchor_generation=_optional_integer(
            value["anchor_generation"], "anchor_generation"
        ),
        context_packet_id=_optional_text(
            value["context_packet_id"], "context_packet_id"
        ),
        context_packet_receipt=(
            None if context is None else _receipt(_mapping(context, "context receipt"))
        ),
        events=_events(value["events"]),
        status=_text(value["status"], "child status"),
        verdict=_optional_text(value["verdict"], "child verdict"),
        scalar_reward=_optional_number(value["scalar_reward"], "scalar_reward"),
        reward_vector=_optional_mapping(value["reward_vector"], "child reward_vector"),
        policy_ids=_strings(value["policy_ids"], "child policy_ids"),
        split=_text(value["split"], "split"),
        visibility=_text(value["visibility"], "visibility"),
        trainability=_text(value["trainability"], "child trainability"),
        validation_reasons=_strings(value["validation_reasons"], "validation_reasons"),
    )


def _events(value: object) -> tuple[EpisodeEvent, ...]:
    return tuple(_event(_mapping(item, "episode event")) for item in _list(value, "events"))


def _event(value: Mapping[str, Any]) -> EpisodeEvent:
    _exact(value, _EVENT_KEYS, "episode event")
    try:
        semantic = SemanticRole(_text(value["semantic_role"], "semantic_role"))
        evidence = EvidenceClass(_text(value["evidence_class"], "evidence_class"))
    except ValueError as error:
        raise IntegrityError(
            "Episode event enum is invalid", "showcase_episode_invalid"
        ) from error
    event_id = _text(value["event_id"], "event_id")
    artifacts = tuple(
        _artifact(_mapping(item, "episode artifact"), event_id)
        for item in _list(value["artifacts"], "event artifacts")
    )
    return EpisodeEvent(
        sequence=_integer(value["sequence"], "event sequence", 1),
        event_id=event_id,
        transaction_id=_text(value["transaction_id"], "transaction_id"),
        parent_event_id=_optional_text(value["parent_event_id"], "parent_event_id"),
        event_type=_text(value["event_type"], "event_type"),
        semantic_role=semantic,
        evidence_class=evidence,
        payload=dict(_mapping(value["payload"], "event payload")),
        artifacts=artifacts,
        causation_id=_optional_text(value["causation_id"], "causation_id"),
        correlation_id=_optional_text(value["correlation_id"], "correlation_id"),
        agent_run_id=_optional_text(value["agent_run_id"], "agent_run_id"),
    )


def _artifact(value: Mapping[str, Any], event_id: str) -> EpisodeArtifact:
    _exact(value, _ARTIFACT_KEYS, "episode artifact")
    if value["event_id"] != event_id:
        _reject("Episode artifact targets another event")
    return EpisodeArtifact(
        _text(value["role"], "artifact role"),
        _receipt(_mapping(value["receipt"], "artifact receipt")),
        event_id,
    )


def _receipt(value: Mapping[str, Any]) -> ArtifactReceipt:
    _exact(value, _RECEIPT_KEYS, "artifact receipt")
    digest = _digest(value["digest"], "artifact digest")
    relative = _text(value["relative_path"], "artifact relative_path")
    if relative != f"sha256/{digest[:2]}/{digest}":
        _reject("Artifact receipt path is not canonical")
    return ArtifactReceipt(
        digest,
        _integer(value["size"], "artifact size", 0),
        _text(value["media_type"], "artifact media_type"),
        relative,
    )


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        _reject(f"{label.title()} fields are invalid")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _reject(f"{label.title()} must be an object")
    return value


def _optional_mapping(value: object, label: str) -> Mapping[str, Any] | None:
    return None if value is None else dict(_mapping(value, label))


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        _reject(f"{label.title()} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _reject(f"{label.title()} is invalid")
    return value


def _optional_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _integer(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _reject(f"{label.title()} is invalid")
    return value


def _optional_integer(value: object, label: str) -> int | None:
    return None if value is None else _integer(value, label, 0)


def _optional_number(value: object, label: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(f"{label.title()} is invalid")
    if not math.isfinite(float(value)):
        _reject(f"{label.title()} is invalid")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    return tuple(_text(item, label) for item in _list(value, label))


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if not _DIGEST.fullmatch(text):
        _reject(f"{label.title()} is invalid")
    return text


def _optional_digest(value: object, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _digests(value: object, label: str) -> tuple[str, ...]:
    return tuple(_digest(item, label) for item in _list(value, label))


def _reject(message: str) -> None:
    raise IntegrityError(message, "showcase_episode_invalid")


__all__ = ["load_episode_graph"]
