"""Independent structural and terminal-reward replay for exported graphs."""

from __future__ import annotations

from dataclasses import dataclass

from apex.core import IntegrityError
from apex.storage import ArtifactStore

from .e2e_validation import explicit_attempt_id
from .episode_semantics import evidence_class, parent_status, semantic_role
from .models import EpisodeGraph, episode_id
from .parent_reward import project_parent_reward
from .e2e_server_lineage_validation import validate_e2e_server_lineage


@dataclass(frozen=True, slots=True)
class EpisodeGraphValidation:
    event_count: int
    artifact_count: int
    reward_replayed: bool


def validate_episode_graph(
    graph: EpisodeGraph,
    artifacts: ArtifactStore | None = None,
    *,
    replay_reward: bool = False,
) -> EpisodeGraphValidation:
    """Validate a typed graph; optionally replay parent reward from CAS bytes."""

    events = _validate_structure(graph)
    validate_e2e_server_lineage(graph.run_id, events, artifacts)
    bindings = tuple(
        artifact for event in events for artifact in event.artifacts
    )
    if replay_reward:
        if artifacts is None:
            _reject("Reward replay requires the exported artifact store")
        _validate_parent_reward(graph, artifacts)
    elif graph.parent.task_reward is not None:
        _reject("A scored parent cannot skip terminal reward replay")
    return EpisodeGraphValidation(len(events), len(bindings), replay_reward)


def _validate_structure(graph: EpisodeGraph) -> tuple[object, ...]:
    parent = graph.parent
    if parent.run_id != graph.run_id:
        _reject("Parent run identity differs from the graph")
    expected_parent_id = episode_id(
        graph.run_id, parent.workload_id or parent.task_id or "root"
    )
    child_ids = tuple(child.episode_id for child in graph.children)
    if (
        parent.episode_id != expected_parent_id
        or parent.child_episode_ids != child_ids
        or len(child_ids) != len(set(child_ids))
    ):
        _reject("Parent/child episode identities differ")
    attempts = tuple(child.attempt_id for child in graph.children)
    if len(attempts) != len(set(attempts)):
        _reject("Attempt identities are duplicated")
    for child in graph.children:
        if (
            child.episode_id != episode_id(graph.run_id, child.attempt_id)
            or child.parent_episode_id != parent.episode_id
        ):
            _reject("Child episode lineage differs")
        _validate_event_partition(child.events, child.attempt_id)
    _validate_event_partition(parent.events, None)
    events = tuple(
        sorted(
            (*parent.events, *(event for child in graph.children for event in child.events)),
            key=lambda item: item.sequence,
        )
    )
    _validate_event_chain(graph, events)
    expected_policies = {
        *(value for value in (parent.reward_policy_id,) if value is not None),
        *(policy for child in graph.children for policy in child.policy_ids),
    }
    if tuple(sorted(expected_policies)) != graph.policy_ids:
        _reject("Graph reward policy projection differs")
    if parent.terminal_status != parent_status(events):
        _reject("Parent terminal status differs from the event chain")
    return events


def _validate_event_partition(events: tuple[object, ...], attempt_id: str | None) -> None:
    sequences = tuple(event.sequence for event in events)
    if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
        _reject("Episode event order is invalid")
    for event in events:
        try:
            observed_attempt = explicit_attempt_id(event)
            expected_evidence = evidence_class(event.payload.get("evidence_class"))
        except Exception as error:
            raise IntegrityError(
                "Episode event semantics cannot be replayed",
                "showcase_trajectory_mismatch",
            ) from error
        if (
            observed_attempt != attempt_id
            or semantic_role(event.event_type) is not event.semantic_role
            or expected_evidence is not event.evidence_class
        ):
            _reject("Episode event semantic projection differs")
        if any(binding.event_id != event.event_id for binding in event.artifacts):
            _reject("Episode artifact event lineage differs")


def _validate_event_chain(graph: EpisodeGraph, events: tuple[object, ...]) -> None:
    if not events or len(events) != graph.high_water_mark:
        _reject("Episode event count differs from the high-water mark")
    if tuple(event.sequence for event in events) != tuple(
        range(1, graph.high_water_mark + 1)
    ):
        _reject("Episode event sequences are not contiguous")
    event_ids = tuple(event.event_id for event in events)
    if len(event_ids) != len(set(event_ids)):
        _reject("Episode event identities are duplicated")
    for index, event in enumerate(events):
        expected_parent = None if index == 0 else events[index - 1].event_id
        if event.parent_event_id != expected_parent:
            _reject("Episode journal parent chain differs")
    if graph.journal_head_event_id != events[-1].event_id:
        _reject("Episode graph head differs from the event chain")


def _validate_parent_reward(graph: EpisodeGraph, artifacts: ArtifactStore) -> None:
    parent = graph.parent
    try:
        replayed = project_parent_reward(graph.run_id, parent.events, artifacts)
    except Exception as error:
        if isinstance(error, IntegrityError):
            raise
        raise IntegrityError(
            "Terminal reward replay failed", "showcase_reward_replay_mismatch"
        ) from error
    observed = (
        parent.task_reward,
        parent.reward_vector,
        parent.reward_policy_id,
        parent.reward_policy_digest,
        parent.reward_source_receipt,
        parent.raw_measurement_receipts,
        parent.trainability,
        parent.untrainable_reason,
    )
    expected = (
        replayed.task_reward,
        replayed.reward_vector,
        replayed.reward_policy_id,
        replayed.reward_policy_digest,
        replayed.reward_source_receipt,
        replayed.raw_measurement_receipts,
        replayed.trainability,
        replayed.untrainable_reason,
    )
    if observed != expected:
        raise IntegrityError(
            "Parent terminal reward differs from raw evidence replay",
            "showcase_reward_replay_mismatch",
        )


def _reject(message: str) -> None:
    raise IntegrityError(message, "showcase_trajectory_mismatch")


__all__ = ["EpisodeGraphValidation", "validate_episode_graph"]
