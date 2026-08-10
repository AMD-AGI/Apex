"""Deterministic, sanitized showcase export and offline verification."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_identifier,
)
from apex.rl import EpisodeArtifact, EpisodeGraph
from apex.storage import ArtifactReceipt, ArtifactStore

from .replication import build_replication_guide
from .report import build_report
from .showcase_offline import validate_offline_showcase
from .showcase_inventory import (
    SHOWCASE_FILE_DECLARATION,
    build_inventory_evidence,
    build_showcase_readme,
)
from .showcase_qualification import qualification_blockers, verify_graph_bundle
from .showcase_sanitization import (
    export_text_artifact,
    projection_contains_private_text,
    sanitize_projection_text,
)


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)$", re.I
)
_TEXT_MEDIA = re.compile(r"^(?:text/|application/(?:json|x-ndjson|yaml|xml))")
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ShowcaseExportResult:
    output_dir: Path
    showcase_id: str
    status: str
    blockers: tuple[str, ...]
    checksums_sha256: str


@dataclass(frozen=True, slots=True)
class ShowcaseVerification:
    root: Path
    showcase_id: str
    status: str
    file_count: int
    checksums_sha256: str
    event_count: int
    artifact_count: int
    reward_replayed: bool
    bundle_verified: bool
    reproduction_verified: bool
    episode_sha256: str
    artifact_manifest_sha256: str
    reward_sha256: str
    result_sha256: str
    reproduction_sha256: str
    verification_receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verification_receipt_sha256",
            sha256_json(self.receipt_payload()),
        )

    def receipt_payload(self) -> dict[str, object]:
        return {
            "schema": "apex.showcase-verification/v2",
            "showcase_id": self.showcase_id,
            "status": self.status,
            "file_count": self.file_count,
            "checksums_sha256": self.checksums_sha256,
            "event_count": self.event_count,
            "artifact_count": self.artifact_count,
            "reward_replayed": self.reward_replayed,
            "bundle_verified": self.bundle_verified,
            "reproduction_verified": self.reproduction_verified,
            "episode_sha256": self.episode_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "reward_sha256": self.reward_sha256,
            "result_sha256": self.result_sha256,
            "reproduction_sha256": self.reproduction_sha256,
        }

    def to_receipt(self) -> dict[str, object]:
        return {
            **self.receipt_payload(),
            "verification_receipt_sha256": self.verification_receipt_sha256,
        }


class ShowcaseExporter:
    """Read canonical graph/CAS evidence; never run or re-grade an optimizer."""

    def __init__(self, artifacts: ArtifactStore) -> None:
        self._artifacts = artifacts

    def export(
        self,
        graph: EpisodeGraph,
        output_dir: Path,
        *,
        showcase_id: str,
    ) -> ShowcaseExportResult:
        selected_id = validate_identifier(showcase_id, field_name="showcase_id")
        _require_public(graph)
        report = build_report(graph)
        replication = build_replication_guide(graph)
        artifacts, artifact_payloads = self._artifacts_payload(graph)
        inventory = build_inventory_evidence(
            graph, replication.document, artifacts, self._artifacts
        )
        bundle = verify_graph_bundle(graph, self._artifacts)
        blockers = qualification_blockers(
            graph,
            replication.document,
            artifacts,
            bundle_verified=bundle is not None,
            inventory_readiness=inventory.readiness,
        )
        status = "published" if not blockers else "pending"
        payloads = _projection_payloads(
            graph, selected_id, status, blockers, artifacts,
            report.document, report.markdown, replication.document,
        )
        payloads.update(artifact_payloads)
        payloads.update(inventory.payloads)
        payloads["README.md"] = build_showcase_readme(
            graph, selected_id, status, blockers, inventory.readiness
        )
        _reject_projection_secrets(payloads)
        checksums = _checksums(payloads)
        payloads["checksums.json"] = canonical_json_bytes(checksums)
        root = _write_payloads(output_dir, payloads)
        verification = verify_showcase(root)
        return ShowcaseExportResult(
            root, selected_id, status, blockers, verification.checksums_sha256
        )

    def _artifacts_payload(
        self, graph: EpisodeGraph
    ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
        indexed = _artifact_index(graph)
        manifest: list[dict[str, Any]] = []
        payloads: dict[str, bytes] = {}
        total = 0
        for digest, value in sorted(indexed.items()):
            receipt: ArtifactReceipt = value["receipt"]
            content = self._artifacts.read_bytes(receipt)
            portable = _portable_artifact(receipt, total)
            exported = content
            redaction_policy = None
            if portable:
                exported, redaction_policy = export_text_artifact(
                    content, digest=receipt.digest
                )
            path = f"trajectory/artifacts/{receipt.relative_path}" if portable else None
            if portable:
                payloads[str(path)] = exported
                total += len(exported)
            manifest.append({
                "digest": digest,
                "size": receipt.size,
                "media_type": receipt.media_type,
                "roles": sorted(value["roles"]),
                "event_ids": sorted(value["event_ids"]),
                "portable_path": path,
                "export_sha256": sha256_bytes(exported) if portable else None,
                "export_size": len(exported) if portable else None,
                "redaction_policy_id": redaction_policy,
                "locator": f"canonical-run-cas:sha256:{digest}",
                "retention": "source_run_required" if not portable else "included",
            })
        return manifest, payloads


def verify_showcase(root: Path) -> ShowcaseVerification:
    """Verify one exported tree without trusting stored scores or checksums."""

    selected = root.expanduser()
    if selected.is_symlink():
        raise IntegrityError("Showcase root cannot be a symlink", "unsafe_showcase_path")
    try:
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise ContractError("Showcase does not exist", "showcase_missing") from error
    if not resolved.is_dir():
        raise ContractError("Showcase root is not a directory", "invalid_showcase")
    checksums_path = resolved / "checksums.json"
    checksums = _load_json(checksums_path, "showcase_checksums_invalid")
    if checksums.get("schema") != "apex.showcase-checksums/v1":
        raise IntegrityError("Showcase checksum schema is invalid", "showcase_checksums_invalid")
    expected = checksums.get("files")
    if not isinstance(expected, Mapping) or not expected:
        raise IntegrityError("Showcase checksums are empty", "showcase_checksums_invalid")
    observed_paths = _showcase_paths(resolved)
    if observed_paths != set(expected) | {"checksums.json"}:
        raise IntegrityError("Showcase file inventory changed", "showcase_file_inventory_mismatch")
    for relative, record in expected.items():
        _verify_checksum_record(resolved, str(relative), record)
    showcase = _load_json(resolved / "showcase.json", "invalid_showcase")
    reward = _load_json(resolved / "reward.json", "invalid_showcase")
    result = _load_json(resolved / "result.json", "invalid_showcase")
    reproduction = _load_json(resolved / "reproduce.json", "invalid_showcase")
    episode = _load_json(
        resolved / "trajectory" / "episode.json", "showcase_episode_invalid"
    )
    manifest = _load_json(
        resolved / "trajectory" / "artifact_manifest.json",
        "showcase_artifact_manifest_invalid",
    )
    _verify_showcase_documents(showcase, reward, result, episode)
    offline = validate_offline_showcase(
        resolved, showcase, episode, manifest, reward, result, reproduction
    )
    return ShowcaseVerification(
        resolved,
        validate_identifier(str(showcase.get("showcase_id", "")), field_name="showcase_id"),
        str(showcase["status"]),
        len(expected) + 1,
        sha256_file(checksums_path),
        offline.event_count,
        offline.artifact_count,
        offline.reward_replayed,
        offline.bundle_verified,
        True,
        sha256_file(resolved / "trajectory" / "episode.json"),
        sha256_file(resolved / "trajectory" / "artifact_manifest.json"),
        sha256_file(resolved / "reward.json"),
        sha256_file(resolved / "result.json"),
        sha256_file(resolved / "reproduce.json"),
    )


def _projection_payloads(
    graph: EpisodeGraph,
    showcase_id: str,
    status: str,
    blockers: tuple[str, ...],
    artifacts: list[dict[str, Any]],
    report: Mapping[str, Any],
    report_markdown: str,
    reproduction: Mapping[str, Any],
) -> dict[str, bytes]:
    episode = _sanitize(graph.to_dict())
    episode_bytes = canonical_json_bytes(episode)
    reward = _sanitize(_reward_document(graph))
    result = _sanitize(_result_document(graph, showcase_id, status, blockers))
    showcase = _sanitize({
        "schema": "apex.showcase/v1",
        "showcase_id": showcase_id,
        "status": status,
        "qualification_blockers": list(blockers),
        "source": {
            "run_id": graph.run_id,
            "episode_graph_id": graph.graph_id,
            "exported_episode_graph_id": (
                f"episode-graph-{sha256_bytes(episode_bytes)[:24]}"
            ),
            "episode_sha256": sha256_bytes(episode_bytes),
            "journal_head_event_id": graph.journal_head_event_id,
            "high_water_mark": graph.high_water_mark,
        },
        "task_kind": graph.parent.kind,
        "terminal_status": graph.parent.terminal_status,
        "task_reward": graph.parent.task_reward,
        "reward_policy_id": graph.parent.reward_policy_id,
        "safety": {
            "sanitizer_runtime": "not_implemented",
            "safety_certified": False,
        },
        "files": dict(SHOWCASE_FILE_DECLARATION),
    })
    return {
        "showcase.json": canonical_json_bytes(showcase),
        "result.json": canonical_json_bytes(result),
        "reward.json": canonical_json_bytes(reward),
        "trajectory/episode.json": episode_bytes,
        "trajectory/artifact_manifest.json": canonical_json_bytes({
            "schema": "apex.showcase-artifact-manifest/v1", "artifacts": artifacts
        }),
        "report.json": canonical_json_bytes(_sanitize(report)),
        "report.md": _sanitize(report_markdown).encode("utf-8"),
        "reproduce.json": canonical_json_bytes(_sanitize(reproduction)),
    }


def _reward_document(graph: EpisodeGraph) -> dict[str, Any]:
    parent = graph.parent
    return {
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


def _result_document(
    graph: EpisodeGraph, showcase_id: str, status: str, blockers: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "schema": "apex.showcase-result/v1",
        "showcase_id": showcase_id,
        "showcase_status": status,
        "qualification_blockers": list(blockers),
        "run_id": graph.run_id,
        "terminal_status": graph.parent.terminal_status,
        "task_reward": graph.parent.task_reward,
        "attempt_count": len(graph.children),
        "keep_attempts": [item.attempt_id for item in graph.children if item.verdict == "keep"],
        "revert_attempts": [
            item.attempt_id for item in graph.children if item.verdict in {"revert", "reject"}
        ],
    }


def _artifact_index(graph: EpisodeGraph) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    events = (*graph.parent.events, *(event for child in graph.children for event in child.events))
    for event in events:
        for artifact in event.artifacts:
            item = indexed.setdefault(
                artifact.receipt.digest,
                {"receipt": artifact.receipt, "roles": set(), "event_ids": set()},
            )
            _same_receipt(item["receipt"], artifact)
            item["roles"].add(artifact.role)
            item["event_ids"].add(artifact.event_id)
    return indexed


def _same_receipt(existing: ArtifactReceipt, artifact: EpisodeArtifact) -> None:
    if existing != artifact.receipt:
        raise IntegrityError("Artifact digest has conflicting receipts", "showcase_artifact_conflict")


def _portable_artifact(receipt: ArtifactReceipt, total: int) -> bool:
    return (
        receipt.size <= _MAX_ARTIFACT_BYTES
        and total + receipt.size <= _MAX_TOTAL_ARTIFACT_BYTES
        and bool(_TEXT_MEDIA.match(receipt.media_type))
    )


def _require_public(graph: EpisodeGraph) -> None:
    private = [
        child.attempt_id for child in graph.children
        if child.visibility != "public" or child.split == "heldout"
    ]
    if private:
        raise ContractError(
            "Private or held-out episodes cannot be exported as a showcase",
            "showcase_private_evidence",
            {"attempt_ids": private},
        )


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else _sanitize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return sanitize_projection_text(value)
    return value


def _reject_projection_secrets(payloads: Mapping[str, bytes]) -> None:
    for path, content in payloads.items():
        if not path.startswith("trajectory/artifacts/"):
            text = content.decode("utf-8")
            if projection_contains_private_text(text):
                raise ContractError("Showcase projection was not sanitized", "showcase_secret_detected")


def _checksums(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    return {
        "schema": "apex.showcase-checksums/v1",
        "files": {
            path: {"sha256": sha256_bytes(content), "size": len(content)}
            for path, content in sorted(payloads.items())
        },
    }


def _write_payloads(output_dir: Path, payloads: Mapping[str, bytes]) -> Path:
    selected = output_dir.expanduser()
    if selected.exists() and selected.is_symlink():
        raise ContractError("Showcase output cannot be a symlink", "unsafe_showcase_path")
    root = selected.resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = _showcase_paths(root)
    if existing.difference(payloads):
        raise ContractError("Showcase output contains unknown files", "showcase_output_not_clean")
    for relative, content in sorted(payloads.items()):
        destination = _safe_output(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return root


def _safe_output(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ContractError("Showcase output path is unsafe", "unsafe_showcase_path")
    destination = root.joinpath(*path.parts)
    if destination.is_symlink():
        raise ContractError("Showcase output file is a symlink", "unsafe_showcase_path")
    return destination


def _showcase_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise IntegrityError("Showcase contains a symlink", "unsafe_showcase_path")
        if path.is_file():
            paths.add(path.relative_to(root).as_posix())
    return paths


def _load_json(path: Path, reason: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("Showcase JSON file is missing", reason)
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("Showcase JSON file is invalid", reason) from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != content:
        raise IntegrityError("Showcase JSON root is invalid", reason)
    return value


def _verify_checksum_record(root: Path, relative: str, record: Any) -> None:
    if not isinstance(record, Mapping):
        raise IntegrityError("Showcase checksum record is invalid", "showcase_checksums_invalid")
    path = _safe_output(root, relative)
    if not path.is_file() or path.stat().st_size != int(record.get("size", -1)):
        raise IntegrityError("Showcase file size changed", "showcase_checksum_mismatch")
    if sha256_file(path) != record.get("sha256"):
        raise IntegrityError("Showcase file digest changed", "showcase_checksum_mismatch")


def _verify_showcase_documents(
    showcase: Mapping[str, Any],
    reward: Mapping[str, Any],
    result: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> None:
    if showcase.get("schema") != "apex.showcase/v1" or showcase.get("status") not in {"pending", "published"}:
        raise IntegrityError("Showcase declaration is invalid", "invalid_showcase")
    if (
        reward.get("schema") != "apex.showcase-reward/v1"
        or result.get("schema") != "apex.showcase-result/v1"
        or reward.get("task_reward") != showcase.get("task_reward")
        or result.get("task_reward") != showcase.get("task_reward")
        or result.get("run_id") != showcase.get("source", {}).get("run_id")
        or episode.get("run_id") != result.get("run_id")
    ):
        raise IntegrityError("Showcase projections disagree", "showcase_projection_mismatch")
    if showcase.get("status") == "published" and showcase.get("qualification_blockers"):
        raise IntegrityError("Published showcase declares blockers", "invalid_showcase")


__all__ = [
    "ShowcaseExportResult",
    "ShowcaseExporter",
    "ShowcaseVerification",
    "verify_showcase",
]
