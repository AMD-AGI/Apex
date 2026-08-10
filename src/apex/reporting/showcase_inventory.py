"""Strict human and provenance inventory for one offline showcase."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from apex.core import canonical_json_bytes, sha256_bytes
from apex.rl import EpisodeGraph
from apex.storage import ArtifactReceipt

from .showcase_sanitization import export_text_artifact, sanitize_projection_text


_CONFIG_ROLES = frozenset(
    {
        "benchmark_config",
        "raw_config",
        "input_config",
        "template_config",
        "evaluation_contract",
        "delivery_measurement_config",
    }
)
_CANDIDATE_ROLES = frozenset(
    {"candidate", "candidate_source", "candidate_patch", "winner_diff", "patch"}
)
_BASE_ROLES = frozenset({"source", "baseline_source"})
SHOWCASE_FILE_DECLARATION = {
    "result": "result.json",
    "reward": "reward.json",
    "episode": "trajectory/episode.json",
    "artifact_manifest": "trajectory/artifact_manifest.json",
    "report": "report.md",
    "reproduction": "reproduce.json",
    "readme": "README.md",
    "raw_config_snapshot": "template/raw_config_snapshot.json",
    "winner_diff": "winner/winner.diff",
    "dependency_receipts": "receipts/dependencies.json",
    "source_receipts": "receipts/sources.json",
    "image_receipts": "receipts/images.json",
    "gpu_receipts": "receipts/gpu.json",
}
SHOWCASE_STATIC_PATHS = frozenset(
    {
        "README.md",
        "result.json",
        "reward.json",
        "showcase.json",
        "checksums.json",
        "report.json",
        "report.md",
        "reproduce.json",
        "trajectory/episode.json",
        "trajectory/artifact_manifest.json",
        "template/raw_config_snapshot.json",
        "winner/winner.diff",
        "receipts/dependencies.json",
        "receipts/sources.json",
        "receipts/images.json",
        "receipts/gpu.json",
    }
)


class ArtifactReader(Protocol):
    def read_bytes(self, receipt: ArtifactReceipt) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ShowcaseInventory:
    payloads: Mapping[str, bytes]
    readiness: Mapping[str, bool]


def build_inventory_evidence(
    graph: EpisodeGraph,
    reproduction: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    reader: ArtifactReader,
) -> ShowcaseInventory:
    """Build fixed config/diff/receipt projections from bound graph evidence."""

    config = _config_snapshot(graph, artifacts, reader)
    winner_diff, diff_ready = _winner_diff(graph, artifacts, reader)
    dependency = _receipt_projection(
        "apex.showcase-dependency-receipts/v1",
        reproduction.get("dependency_receipts"),
    )
    sources = _source_projection(graph, reproduction, artifacts)
    images = _image_projection(graph, reproduction)
    gpu = _gpu_projection(graph, artifacts)
    readiness = {
        "config_snapshot": config[1],
        "winner_diff": diff_ready,
        "dependency_receipts": dependency[1],
        "source_receipts": sources[1],
        "image_receipts": images[1],
        "gpu_receipts": gpu[1],
    }
    return ShowcaseInventory(
        payloads={
            "template/raw_config_snapshot.json": canonical_json_bytes(config[0]),
            "winner/winner.diff": winner_diff,
            "receipts/dependencies.json": canonical_json_bytes(dependency[0]),
            "receipts/sources.json": canonical_json_bytes(sources[0]),
            "receipts/images.json": canonical_json_bytes(images[0]),
            "receipts/gpu.json": canonical_json_bytes(gpu[0]),
        },
        readiness=readiness,
    )


def build_showcase_readme(
    graph: EpisodeGraph,
    showcase_id: str,
    status: str,
    blockers: Sequence[str],
    readiness: Mapping[str, bool],
) -> bytes:
    """Render the required human index without inventing missing evidence."""

    blockers_text = ", ".join(blockers) if blockers else "none"
    children = tuple(graph.children)
    events = (*graph.parent.events, *(event for child in children for event in child.events))
    tool_names = sorted(
        {
            str(event.payload["tool_name"])
            for event in events
            if isinstance(event.payload.get("tool_name"), str)
        }
    )
    knowledge_events = sum("knowledge" in event.event_type for event in events)
    winner = next((child.attempt_id for child in children if child.verdict == "keep"), None)
    lines = [
        f"# {showcase_id}",
        "",
        "This directory is a deterministic projection of one canonical Apex run.",
        "It is not an applied patch or a substitute for evaluator evidence.",
        "",
        f"- Status: `{status}`",
        f"- Task kind: `{graph.parent.kind}`",
        f"- Run ID: `{graph.run_id}`",
        f"- Terminal status: `{graph.parent.terminal_status}`",
        f"- Task reward: `{graph.parent.task_reward}`",
        f"- Qualification blockers: `{blockers_text}`",
        "- Safety: `sanitizer_runtime=not_implemented`, `safety_certified=false`",
        "",
        "## Hardware and runtime",
        "",
        f"- Frozen provenance: `{canonical_json_bytes(_sanitize(graph.provenance)).decode()}`",
        "- Dependency, source, image, and GPU identities are separate receipt files.",
        "",
        "## Input, trajectory, and winner",
        "",
        "- Exact natural-language/config inputs are receipt-bound in the trajectory and template snapshot.",
        f"- Attempts: `{len(children)}`; KEEP: `{sum(item.verdict == 'keep' for item in children)}`; REVERT/REJECT: `{sum(item.verdict in {'revert', 'reject'} for item in children)}`.",
        f"- Winner attempt: `{winner}`",
        f"- Typed tools: `{', '.join(tool_names) if tool_names else 'none recorded'}`",
        f"- Knowledge-related events: `{knowledge_events}`",
        "- Baseline/winner source identities and the human diff are under `receipts/` and `winner/`.",
        "",
        "## Required evidence inventory",
        "",
    ]
    lines.extend(
        f"- {name}: `{'complete' if ready else 'missing'}`"
        for name, ready in sorted(readiness.items())
    )
    lines.extend(
        [
            "",
            "## Limitations and reproduction",
            "",
            f"- Unresolved limitations: `{blockers_text}`",
            "Verify offline with `apex showcase verify --path .`.",
            "Reproduction commands and immutable identities are in `reproduce.json`.",
            "",
        ]
    )
    return sanitize_projection_text("\n".join(lines)).encode("utf-8")


def _config_snapshot(
    graph: EpisodeGraph,
    artifacts: Sequence[Mapping[str, Any]],
    reader: ArtifactReader,
) -> tuple[dict[str, Any], bool]:
    entries: list[dict[str, Any]] = []
    policies = {
        str(item.get("digest")): item.get("redaction_policy_id")
        for item in artifacts
    }
    for role, receipt in _bound_artifacts(graph, artifacts, _CONFIG_ROLES):
        try:
            content, policy = export_text_artifact(
                reader.read_bytes(receipt), digest=receipt.digest
            )
            text = content.decode("utf-8")
        except Exception:  # missing/non-text evidence remains an explicit blocker
            continue
        entries.append(
            {
                "role": role,
                "receipt": receipt.to_dict(),
                "export_sha256": sha256_bytes(content),
                "redaction_policy_id": policies.get(receipt.digest, policy),
                "content": text,
            }
        )
    ready = bool(entries)
    return {
        "schema": "apex.showcase-raw-config-snapshot/v1",
        "status": "complete" if ready else "missing",
        "entries": entries,
        "summary": (
            "Exact raw/template configuration bytes bound to the canonical run."
            if ready
            else "No portable raw/template configuration receipt is bound."
        ),
    }, ready


def _winner_diff(
    graph: EpisodeGraph,
    artifacts: Sequence[Mapping[str, Any]],
    reader: ArtifactReader,
) -> tuple[bytes, bool]:
    kept = tuple(child for child in graph.children if child.verdict == "keep")
    if len(kept) != 1:
        return b"# Winner diff unavailable: exactly one KEEP is required.\n", False
    bundle_diff = _bundle_diff(graph, artifacts, reader)
    if bundle_diff is not None:
        return bundle_diff
    bound = _child_artifacts(kept[0], artifacts)
    direct = tuple(item for item in bound if item[0] in {"candidate_patch", "winner_diff", "patch"})
    if len(direct) == 1:
        return _text_bytes(reader, direct[0][1], "winner diff")
    bases = tuple(item for item in bound if item[0] in _BASE_ROLES)
    candidates = tuple(item for item in bound if item[0] in _CANDIDATE_ROLES)
    if len(bases) != 1 or len(candidates) != 1:
        return b"# Winner diff unavailable: source/candidate pairing is incomplete.\n", False
    before, before_ok = _text_bytes(reader, bases[0][1], "baseline")
    after, after_ok = _text_bytes(reader, candidates[0][1], "winner")
    if not before_ok or not after_ok:
        return b"# Winner diff unavailable: source bytes are not portable UTF-8.\n", False
    diff = "".join(
        difflib.unified_diff(
            before.decode("utf-8").splitlines(keepends=True),
            after.decode("utf-8").splitlines(keepends=True),
            fromfile=f"baseline-{bases[0][1].digest[:12]}",
            tofile=f"winner-{candidates[0][1].digest[:12]}",
        )
    ).encode("utf-8")
    return (diff, True) if diff else (b"# Winner diff is empty.\n", False)


def _bundle_diff(
    graph: EpisodeGraph,
    artifacts: Sequence[Mapping[str, Any]],
    reader: ArtifactReader,
) -> tuple[bytes, bool] | None:
    evidence = _bound_artifacts(graph, artifacts, frozenset({"winner_bundle"}))
    file_receipts = {
        receipt.digest
        for role, receipt in _bound_artifacts(graph, artifacts, None)
        if role.startswith("winner_bundle_file_")
    }
    if len(evidence) != 1 or not file_receipts:
        return None
    try:
        document = json.loads(reader.read_bytes(evidence[0][1]))
        raw_files = document.get("files") if isinstance(document, Mapping) else None
        patches = [
            item
            for item in raw_files
            if isinstance(item, Mapping)
            and str(item.get("path", "")).endswith((".patch", ".diff"))
        ]
        if len(patches) != 1 or not isinstance(patches[0].get("receipt"), Mapping):
            return None
        receipt = ArtifactReceipt.from_dict(dict(patches[0]["receipt"]))
        if receipt.digest not in file_receipts:
            return None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return _text_bytes(reader, receipt, "winner diff")


def _text_bytes(
    reader: ArtifactReader, receipt: ArtifactReceipt, label: str
) -> tuple[bytes, bool]:
    try:
        content, _ = export_text_artifact(reader.read_bytes(receipt), digest=receipt.digest)
        content.decode("utf-8")
    except Exception:
        return f"# {label} is unavailable.\n".encode("utf-8"), False
    return content, bool(content.strip())


def _receipt_projection(
    schema: str, raw: Any
) -> tuple[dict[str, Any], bool]:
    entries = list(raw) if isinstance(raw, list) else []
    ready = bool(entries) and all(isinstance(item, Mapping) for item in entries)
    return {
        "schema": schema,
        "status": "complete" if ready else "missing",
        "entries": _sanitize(entries),
    }, ready


def _source_projection(
    graph: EpisodeGraph,
    reproduction: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    commits = reproduction.get("source_commits")
    commits = list(commits) if isinstance(commits, list) else []
    receipts = [
        {"role": role, "receipt": receipt.to_dict()}
        for role, receipt in _bound_artifacts(graph, artifacts, _BASE_ROLES)
    ]
    ready = bool(commits) and bool(receipts)
    return {
        "schema": "apex.showcase-source-receipts/v1",
        "status": "complete" if ready else "missing",
        "commits": _sanitize(commits),
        "receipts": receipts,
    }, ready


def _image_projection(
    graph: EpisodeGraph,
    reproduction: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    images = {
        name: reproduction.get(name)
        for name in ("parent_image_digest", "derived_image_digest")
    }
    if graph.parent.kind == "single_kernel":
        parent = images["parent_image_digest"]
        ready = (
            (parent is None or _image_digest(parent))
            and images["derived_image_digest"] is None
        )
        applicability = "optional_parent_no_derived_image"
    else:
        ready = all(_image_digest(value) for value in images.values())
        applicability = "parent_and_derived_images_required"
    return {
        "schema": "apex.showcase-image-receipts/v1",
        "status": "complete" if ready else "missing",
        "applicability": applicability,
        "images": images,
    }, ready


def _image_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _gpu_projection(
    graph: EpisodeGraph, artifacts: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], bool]:
    receipts = [
        {"role": role, "receipt": receipt.to_dict()}
        for role, receipt in _bound_artifacts(graph, artifacts, None)
        if "gpu" in role
    ]
    ready = bool(receipts)
    return {
        "schema": "apex.showcase-gpu-receipts/v1",
        "status": "complete" if ready else "missing",
        "provenance": _sanitize(dict(graph.provenance)),
        "receipts": receipts,
    }, ready


def _bound_artifacts(
    graph: EpisodeGraph,
    artifacts: Sequence[Mapping[str, Any]],
    roles: frozenset[str] | None,
) -> tuple[tuple[str, ArtifactReceipt], ...]:
    portable = {
        str(item.get("digest"))
        for item in artifacts
        if item.get("portable_path") is not None
    }
    values: list[tuple[str, ArtifactReceipt]] = []
    events = (*graph.parent.events, *(event for child in graph.children for event in child.events))
    for event in events:
        for artifact in event.artifacts:
            if artifact.receipt.digest not in portable:
                continue
            if roles is None or artifact.role in roles:
                values.append((artifact.role, artifact.receipt))
    return _deduplicate(values)


def _child_artifacts(child: Any, artifacts: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ArtifactReceipt], ...]:
    portable = {
        str(item.get("digest"))
        for item in artifacts
        if item.get("portable_path") is not None
    }
    return _deduplicate(
        [
            (artifact.role, artifact.receipt)
            for event in child.events
            for artifact in event.artifacts
            if artifact.receipt.digest in portable
        ]
    )


def _deduplicate(
    values: Sequence[tuple[str, ArtifactReceipt]],
) -> tuple[tuple[str, ArtifactReceipt], ...]:
    unique = {(role, receipt.digest): (role, receipt) for role, receipt in values}
    return tuple(unique[key] for key in sorted(unique))


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return sanitize_projection_text(value) if isinstance(value, str) else value


__all__ = [
    "SHOWCASE_FILE_DECLARATION",
    "SHOWCASE_STATIC_PATHS",
    "ShowcaseInventory",
    "build_inventory_evidence",
    "build_showcase_readme",
]
