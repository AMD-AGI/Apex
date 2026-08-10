"""One deterministic qualification policy shared by export and verification."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from apex.core import IntegrityError
from apex.delivery import PortableBundleVerification, verify_portable_bundle
from apex.rl import EpisodeGraph


_BUNDLE_ROLE = "winner_bundle"
_BUNDLE_VERIFY_ROLE = "bundle_verification"


def qualification_blockers(
    graph: EpisodeGraph,
    reproduction: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    bundle_verified: bool,
    inventory_readiness: Mapping[str, bool],
) -> tuple[str, ...]:
    """Return every reason the graph cannot be declared a published winner."""

    parent = graph.parent
    roles = {
        str(role)
        for item in artifacts
        for role in item.get("roles", ())
        if isinstance(role, str)
    }
    blockers: set[str] = set()
    if parent.task_reward is None:
        blockers.add("terminal_reward_missing")
    elif parent.task_reward <= 120:
        blockers.add("terminal_reward_not_above_120")
    if parent.trainability != "complete":
        blockers.add("parent_episode_not_trainable")
    if not any(item.verdict == "keep" for item in graph.children):
        blockers.add("keep_attempt_missing")
    if not parent.reward_policy_id or not parent.reward_policy_digest:
        blockers.add("reward_policy_missing")
    if not parent.raw_measurement_receipts:
        blockers.add("raw_measurement_receipts_missing")
    if _BUNDLE_ROLE not in roles:
        blockers.add("winner_bundle_missing")
    if _BUNDLE_VERIFY_ROLE not in roles:
        blockers.add("bundle_verification_missing")
    elif not bundle_verified:
        blockers.add("bundle_verification_invalid")
    if reproduction.get("reproducible") is not True:
        blockers.add("reproduction_incomplete")
    if any(item.get("portable_path") is None for item in artifacts):
        blockers.add("nonportable_artifacts_present")
    for name, ready in sorted(inventory_readiness.items()):
        if not ready:
            blockers.add(f"{name}_missing")
    return tuple(sorted(blockers))


def verify_graph_bundle(
    graph: EpisodeGraph, artifacts: object
) -> PortableBundleVerification | None:
    """Verify the one delivery event carrying a portable winner bundle."""

    events = (
        *graph.parent.events,
        *(event for child in graph.children for event in child.events),
    )
    candidates = tuple(
        event
        for event in events
        if any(item.role == _BUNDLE_ROLE for item in event.artifacts)
        or any(item.role == _BUNDLE_VERIFY_ROLE for item in event.artifacts)
    )
    if not candidates:
        return None
    if len(candidates) != 1:
        raise IntegrityError(
            "Showcase has ambiguous bundle evidence", "showcase_bundle_ambiguous"
        )
    event = candidates[0]
    evidence = tuple(item for item in event.artifacts if item.role == _BUNDLE_ROLE)
    receipts = tuple(
        item for item in event.artifacts if item.role == _BUNDLE_VERIFY_ROLE
    )
    if len(evidence) != 1 or len(receipts) != 1:
        raise IntegrityError(
            "Showcase bundle evidence is incomplete", "showcase_bundle_invalid"
        )
    verification = verify_portable_bundle(
        artifacts, evidence[0].receipt, receipts[0].receipt  # type: ignore[arg-type]
    )
    file_role_values = tuple(
        item.role for item in event.artifacts if item.role.startswith("winner_bundle_file_")
    )
    file_roles = set(file_role_values)
    expected_roles = {
        f"winner_bundle_file_{index:04d}"
        for index in range(verification.file_count)
    }
    if file_roles != expected_roles or len(file_role_values) != len(expected_roles):
        raise IntegrityError(
            "Showcase bundle file bindings are incomplete", "showcase_bundle_invalid"
        )
    return verification


__all__ = ["qualification_blockers", "verify_graph_bundle"]
