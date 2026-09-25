"""Offline trajectory, CAS-manifest, and reward replay for showcases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from apex.core import IntegrityError, sha256_bytes
from apex.rl import EpisodeGraph, load_episode_graph, validate_episode_graph
from apex.storage import ArtifactReceipt

from .showcase_sanitization import (
    HOST_PATH_REDACTION_POLICY,
    validate_exported_text_artifact,
)
from .replication import build_replication_guide
from .showcase_inventory import (
    SHOWCASE_FILE_DECLARATION,
    SHOWCASE_STATIC_PATHS,
    ShowcaseInventory,
    build_inventory_evidence,
    build_showcase_readme,
)
from .showcase_qualification import qualification_blockers, verify_graph_bundle


_MANIFEST_KEYS = {
    "digest", "size", "media_type", "roles", "event_ids", "portable_path",
    "export_sha256", "export_size", "redaction_policy_id", "locator", "retention",
}
_GRAPH_ID = re.compile(r"^episode-graph-[0-9a-f]{24}$")


@dataclass(frozen=True, slots=True)
class OfflineShowcaseValidation:
    event_count: int
    artifact_count: int
    reward_replayed: bool
    bundle_verified: bool
    qualification_blockers: tuple[str, ...]


def validate_offline_showcase(
    root: Path,
    showcase: Mapping[str, Any],
    episode_document: Mapping[str, Any],
    manifest_document: Mapping[str, Any],
    reward_document: Mapping[str, Any],
    result_document: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> OfflineShowcaseValidation:
    """Validate one sanitized graph and replay scored terminal evidence."""

    graph = load_episode_graph(episode_document)
    _validate_source(showcase, graph, episode_document)
    available, artifacts = _validate_manifest(root, graph, manifest_document)
    _validate_tree_inventory(root, manifest_document)
    bundle = verify_graph_bundle(graph, artifacts)
    inventory = build_inventory_evidence(
        graph, reproduction, manifest_document["artifacts"], artifacts
    )
    blockers = _validate_projections(
        graph,
        showcase,
        manifest_document,
        reward_document,
        result_document,
        reproduction,
        bundle_verified=bundle is not None,
        inventory_readiness=inventory.readiness,
    )
    _validate_inventory_files(root, graph, showcase, blockers, inventory)
    replay = _should_replay(graph, available, showcase)
    validation = validate_episode_graph(
        graph,
        artifacts if replay else None,
        replay_reward=replay,
    )
    return OfflineShowcaseValidation(
        validation.event_count,
        validation.artifact_count,
        validation.reward_replayed,
        bundle is not None,
        blockers,
    )


def _validate_projections(
    graph: EpisodeGraph,
    showcase: Mapping[str, Any],
    manifest: Mapping[str, Any],
    reward: Mapping[str, Any],
    result: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    *,
    bundle_verified: bool,
    inventory_readiness: Mapping[str, bool],
) -> tuple[str, ...]:
    parent = graph.parent
    artifacts = manifest["artifacts"]
    expected_reproduction = dict(build_replication_guide(graph).document)
    expected_reproduction["episode_graph_id"] = showcase["source"]["episode_graph_id"]
    blockers = qualification_blockers(
        graph,
        reproduction,
        artifacts,
        bundle_verified=bundle_verified,
        inventory_readiness=inventory_readiness,
    )
    status = "published" if not blockers else "pending"
    expected_reward = {
        "schema": "apex.showcase-reward/v1",
        "task_kind": parent.kind,
        "task_reward": parent.task_reward,
        "reward_vector": dict(parent.reward_vector) if parent.reward_vector else None,
        "reward_policy_id": parent.reward_policy_id,
        "reward_policy_digest": parent.reward_policy_digest,
        "reward_source_receipt": parent.reward_source_receipt,
        "raw_measurement_receipts": list(parent.raw_measurement_receipts),
        "trainability": parent.trainability,
        "untrainable_reason": parent.untrainable_reason,
    }
    expected_result = {
        "schema": "apex.showcase-result/v1",
        "showcase_id": showcase.get("showcase_id"),
        "showcase_status": status,
        "qualification_blockers": list(blockers),
        "run_id": graph.run_id,
        "terminal_status": parent.terminal_status,
        "task_reward": parent.task_reward,
        "attempt_count": len(graph.children),
        "keep_attempts": [item.attempt_id for item in graph.children if item.verdict == "keep"],
        "revert_attempts": [
            item.attempt_id
            for item in graph.children
            if item.verdict in {"revert", "reject"}
        ],
    }
    if (
        dict(reward) != expected_reward
        or dict(result) != expected_result
        or dict(reproduction) != expected_reproduction
        or showcase.get("status") != status
        or showcase.get("qualification_blockers") != list(blockers)
        or showcase.get("task_kind") != parent.kind
        or showcase.get("terminal_status") != parent.terminal_status
        or showcase.get("task_reward") != parent.task_reward
        or showcase.get("reward_policy_id") != parent.reward_policy_id
    ):
        _reject("Showcase projections or qualification status differ from the graph")
    return blockers


def _validate_inventory_files(
    root: Path,
    graph: EpisodeGraph,
    showcase: Mapping[str, Any],
    blockers: tuple[str, ...],
    inventory: ShowcaseInventory,
) -> None:
    status = "published" if not blockers else "pending"
    payloads = dict(inventory.payloads)
    payloads["README.md"] = build_showcase_readme(
        graph,
        str(showcase.get("showcase_id", "")),
        status,
        blockers,
        inventory.readiness,
    )
    if showcase.get("files") != SHOWCASE_FILE_DECLARATION:
        _reject("Showcase required file declaration is incomplete")
    for relative, expected in payloads.items():
        try:
            observed = (root / relative).read_bytes()
        except OSError:
            _reject("Showcase required evidence file is missing")
        if observed != expected:
            _reject("Showcase required evidence projection differs from the graph")


def _validate_tree_inventory(root: Path, manifest: Mapping[str, Any]) -> None:
    portable = {
        str(item["portable_path"])
        for item in manifest["artifacts"]
        if item["portable_path"] is not None
    }
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != SHOWCASE_STATIC_PATHS | portable:
        raise IntegrityError(
            "Showcase file inventory differs from its typed manifest",
            "showcase_file_inventory_mismatch",
        )


def _validate_source(
    showcase: Mapping[str, Any],
    graph: EpisodeGraph,
    episode_document: Mapping[str, Any],
) -> None:
    source = showcase.get("source")
    original_id = source.get("episode_graph_id") if isinstance(source, Mapping) else None
    expected = {
        "run_id": graph.run_id,
        "episode_graph_id": original_id,
        "exported_episode_graph_id": graph.graph_id,
        "episode_sha256": sha256_bytes(graph.canonical_bytes),
        "journal_head_event_id": graph.journal_head_event_id,
        "high_water_mark": graph.high_water_mark,
    }
    if (
        not isinstance(source, Mapping)
        or not isinstance(original_id, str)
        or not _GRAPH_ID.fullmatch(original_id)
        or set(source) != set(expected)
        or dict(source) != expected
        or graph.to_dict() != dict(episode_document)
    ):
        _reject("Showcase source declaration differs from its episode graph")


def _validate_manifest(
    root: Path,
    graph: EpisodeGraph,
    document: Mapping[str, Any],
) -> tuple[set[str], "_ExportedArtifactStore"]:
    if set(document) != {"schema", "artifacts"} or document.get("schema") != (
        "apex.showcase-artifact-manifest/v1"
    ):
        _reject("Showcase artifact manifest schema is invalid")
    raw = document.get("artifacts")
    if not isinstance(raw, list):
        _reject("Showcase artifact manifest entries are invalid")
    expected = _artifact_index(graph)
    observed: dict[str, Mapping[str, Any]] = {}
    available: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != _MANIFEST_KEYS:
            _reject("Showcase artifact manifest entry is invalid")
        digest = item.get("digest")
        if not isinstance(digest, str) or digest in observed:
            _reject("Showcase artifact manifest digest is invalid")
        observed[digest] = item
        receipt, roles, event_ids = expected.get(digest, (None, None, None))
        if receipt is None:
            _reject("Showcase manifest contains an unbound artifact")
        _validate_entry(item, receipt, roles, event_ids)
        if item["portable_path"] is not None:
            available.add(digest)
    if set(observed) != set(expected) or list(observed) != sorted(observed):
        _reject("Showcase artifact manifest inventory differs from the graph")
    store = _ExportedArtifactStore(root / "trajectory" / "artifacts", observed)
    for digest in sorted(available):
        store.verify(expected[digest][0])
    return available, store


def _artifact_index(
    graph: EpisodeGraph,
) -> dict[str, tuple[ArtifactReceipt, set[str], set[str]]]:
    indexed: dict[str, tuple[ArtifactReceipt, set[str], set[str]]] = {}
    events = (
        *graph.parent.events,
        *(event for child in graph.children for event in child.events),
    )
    for event in events:
        for artifact in event.artifacts:
            existing = indexed.get(artifact.receipt.digest)
            if existing is None:
                indexed[artifact.receipt.digest] = (
                    artifact.receipt,
                    {artifact.role},
                    {event.event_id},
                )
                continue
            if existing[0] != artifact.receipt:
                _reject("Showcase graph has conflicting artifact receipts")
            existing[1].add(artifact.role)
            existing[2].add(event.event_id)
    return indexed


def _validate_entry(
    item: Mapping[str, Any],
    receipt: ArtifactReceipt,
    roles: set[str],
    event_ids: set[str],
) -> None:
    path = item.get("portable_path")
    included_path = f"trajectory/artifacts/{receipt.relative_path}"
    policy = item.get("redaction_policy_id")
    export_identity_valid = (
        item.get("export_sha256") == receipt.digest
        and item.get("export_size") == receipt.size
        if policy is None and path is not None
        else policy == HOST_PATH_REDACTION_POLICY
        and isinstance(item.get("export_sha256"), str)
        and item.get("export_sha256") != receipt.digest
        and isinstance(item.get("export_size"), int)
    )
    if path is None:
        export_identity_valid = all(
            item.get(key) is None
            for key in ("export_sha256", "export_size", "redaction_policy_id")
        )
    if (
        item.get("size") != receipt.size
        or item.get("media_type") != receipt.media_type
        or item.get("roles") != sorted(roles)
        or item.get("event_ids") != sorted(event_ids)
        or item.get("locator") != f"canonical-run-cas:sha256:{receipt.digest}"
        or path not in {None, included_path}
        or not export_identity_valid
        or item.get("retention")
        != ("included" if path is not None else "source_run_required")
    ):
        _reject("Showcase artifact manifest entry differs from graph/CAS")


class _ExportedArtifactStore:
    """Read original or path-redacted bytes under manifest-bound identities."""

    def __init__(self, root: Path, entries: Mapping[str, Mapping[str, Any]]) -> None:
        self.root = root
        self._entries = entries

    def read_bytes(self, receipt: ArtifactReceipt) -> bytes:
        item = self._entries.get(receipt.digest)
        if item is None or item.get("portable_path") is None:
            raise IntegrityError("Artifact is missing", "artifact_missing")
        path = self.root / receipt.relative_path
        try:
            content = path.read_bytes()
        except OSError as error:
            raise IntegrityError("Artifact is missing", "artifact_missing") from error
        if (
            len(content) != item.get("export_size")
            or sha256_bytes(content) != item.get("export_sha256")
        ):
            raise IntegrityError(
                "Exported artifact bytes differ from the manifest",
                "showcase_artifact_manifest_invalid",
            )
        policy = item.get("redaction_policy_id")
        validate_exported_text_artifact(content, policy)
        if policy is None and (
            len(content) != receipt.size or sha256_bytes(content) != receipt.digest
        ):
            raise IntegrityError(
                "Artifact failed original receipt verification",
                "artifact_digest_mismatch",
            )
        return content

    def verify(self, receipt: ArtifactReceipt) -> None:
        self.read_bytes(receipt)


def _should_replay(
    graph: EpisodeGraph,
    available: set[str],
    showcase: Mapping[str, Any],
) -> bool:
    parent_digests = {
        artifact.receipt.digest
        for event in graph.parent.events
        for artifact in event.artifacts
    }
    terminal = any(
        event.event_type.replace(".", "_") == "delivery_result"
        and event.payload.get("kind")
        in {"e2e_terminal_result", "kernel_terminal_result"}
        for event in graph.parent.events
    )
    required = graph.parent.task_reward is not None or showcase.get("status") == "published"
    return required or not terminal or parent_digests <= available


def _reject(message: str) -> None:
    raise IntegrityError(message, "showcase_trajectory_mismatch")


__all__ = ["OfflineShowcaseValidation", "validate_offline_showcase"]
