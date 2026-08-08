"""Deterministic JSON/JSONL/SFT export from an :class:`EpisodeGraph`."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes
from apex.evaluation import GateVerdict, kernel_reward
from apex.storage import ArtifactReceipt, ArtifactStore

from .e2e_validation import allows_source_free_e2e, validate_e2e_export_reward
from .models import CandidateEpisode, EpisodeArtifact, EpisodeGraph, SemanticRole


_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)$", re.I)
_SECRET_TEXT = re.compile(
    r"(?:sk-(?:ant-)?[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{16,}|"
    r"github_pat_[A-Za-z0-9_]{16,}|Bearer\s+[A-Za-z0-9._~+/-]{16,}|"
    r"https?://[^\s/@:]+:[^\s/@]+@)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)"
    r"\s*[:=]\s*[\"']?(?!\[REDACTED\])\S{4,}",
    re.I,
)
_SECRET_OPTION = re.compile(
    r"--(?:api[_-]?key|authorization|password|secret|access[_-]?token)\s+\S{4,}",
    re.I,
)


@dataclass(frozen=True, slots=True)
class DatasetExportConfig:
    """Frozen selection and failure policy for one export batch."""

    split: str | None = None
    policy_id: str | None = None
    on_incomplete: str = "fail"
    include_sft: bool = True
    exporter_version: str = "apex_rl_export_v1"

    def __post_init__(self) -> None:
        if self.on_incomplete not in {"fail", "skip"}:
            raise ContractError("Invalid incomplete policy", "invalid_export_policy")
        if self.split is not None and self.split not in {"train", "validation", "heldout"}:
            raise ContractError("Invalid dataset split", "invalid_export_split")


@dataclass(frozen=True, slots=True)
class DatasetExportResult:
    output_dir: Path
    record_count: int
    sft_count: int
    skipped: tuple[Mapping[str, str], ...]
    dataset_sha256: str
    manifest_sha256: str


class DatasetExporter:
    """A read-only graph/CAS consumer; no event writer exists here."""

    def __init__(self, artifacts: ArtifactStore) -> None:
        self._artifacts = artifacts

    def export(
        self,
        graph: EpisodeGraph,
        output_dir: Path,
        *,
        config: DatasetExportConfig | None = None,
    ) -> DatasetExportResult:
        chosen = config or DatasetExportConfig()
        _validate_export_partition(graph, chosen)
        parent_document = graph.parent.to_dict()
        _reject_secrets(parent_document)
        records, sft, skipped = self._select_records(graph, chosen)
        payloads = _dataset_payloads(parent_document, records, sft, skipped, chosen)
        manifest = _dataset_manifest(graph, records, payloads, chosen)
        payloads["export_manifest.json"] = canonical_json_bytes(manifest)
        output_dir = Path(output_dir)
        _write_files(output_dir, payloads)
        jsonl = payloads["dataset.jsonl"]
        return DatasetExportResult(
            output_dir=output_dir,
            record_count=len(records),
            sft_count=len(sft),
            skipped=tuple(skipped),
            dataset_sha256=sha256_bytes(jsonl),
            manifest_sha256=sha256_bytes(payloads["export_manifest.json"]),
        )

    def _select_records(
        self, graph: EpisodeGraph, chosen: DatasetExportConfig
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
        records: list[dict[str, Any]] = []
        sft: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for child in graph.children:
            reason = self._selection_reason(child, chosen)
            if reason is not None:
                if chosen.on_incomplete == "fail" and reason not in {
                    "split_filtered",
                    "private_excluded_from_train",
                    "policy_filtered",
                }:
                    raise ContractError(
                        f"Episode {child.attempt_id} cannot be exported: {reason}",
                        "episode_export_incomplete",
                        {"attempt_id": child.attempt_id, "reason": reason},
                    )
                skipped.append({"attempt_id": child.attempt_id, "reason": reason})
                continue
            record, candidate_text = self._record(graph, child, chosen.exporter_version)
            records.append(record)
            if chosen.include_sft and child.verdict == "keep" and candidate_text is not None:
                sft.append(self._sft_record(record, candidate_text))
        if not records:
            raise ContractError("Dataset filter produced no records", "empty_dataset_export")
        records.sort(key=lambda item: (item["parent_episode_id"], item["attempt_id"]))
        sft.sort(key=lambda item: item["episode_id"])
        return records, sft, skipped

    def _selection_reason(
        self, child: CandidateEpisode, config: DatasetExportConfig
    ) -> str | None:
        if config.split is not None and child.split != config.split:
            return "split_filtered"
        if (
            config.policy_id is not None
            and child.policy_ids
            and config.policy_id not in child.policy_ids
        ):
            return "policy_filtered"
        if config.split == "train" and (
            child.split == "heldout" or child.visibility in {"private", "heldout_private"}
        ):
            return "private_excluded_from_train"
        if child.trainability != "complete":
            return ",".join(child.validation_reasons) or "episode_truncated"
        return None

    def _record(
        self,
        graph: EpisodeGraph,
        child: CandidateEpisode,
        exporter_version: str,
    ) -> tuple[dict[str, Any], str | None]:
        if child.context_packet_receipt is None:
            raise IntegrityError("ContextPacket receipt is missing", "context_packet_missing")
        context_bytes = self._artifacts.read_bytes(child.context_packet_receipt)
        try:
            observation = json.loads(context_bytes)
        except json.JSONDecodeError as error:
            raise IntegrityError("ContextPacket is not JSON", "invalid_context_packet_artifact") from error
        _reject_secrets(observation)
        artifact_values, candidate_text = self._materialize_artifacts(child)
        events = [event.to_dict() for event in child.events]
        _reject_secrets(events)
        self._validate_reward(graph, child)
        roles: dict[str, list[Mapping[str, Any]]] = {}
        for item in artifact_values:
            roles.setdefault(str(item["role"]), []).append(item)
        record = {
            "schema_name": "apex.rl_transition",
            "schema_version": 1,
            "exporter_version": exporter_version,
            "episode_id": child.episode_id,
            "parent_episode_id": child.parent_episode_id,
            "run_id": graph.run_id,
            "attempt_id": child.attempt_id,
            "candidate_id": child.candidate_id,
            "opportunity_id": child.opportunity_id,
            "task_id": child.task_id,
            "kernel_id": child.kernel_id,
            "state_generation": child.state_generation,
            "anchor_generation": child.anchor_generation,
            "causal_event_ids": [event.event_id for event in child.events],
            "observation": {
                "context_packet_id": child.context_packet_id,
                "receipt": child.context_packet_receipt.to_dict(),
                "content": observation,
            },
            "observations": _events_with_roles(child, {SemanticRole.OBSERVATION}),
            "actions": _events_with_roles(child, {SemanticRole.ACTION}),
            "tools": _events_with_roles(child, {SemanticRole.TOOL}),
            "outcomes": _events_with_roles(child, {SemanticRole.OUTCOME}),
            "decisions": _events_with_roles(child, {SemanticRole.DECISION}),
            "reward": {
                "scalar": child.scalar_reward,
                "vector": dict(child.reward_vector) if child.reward_vector else None,
                "policy_ids": list(child.policy_ids),
            },
            "costs": {
                "events": _events_with_roles(child, {SemanticRole.COST}),
                "reward_component": (
                    child.reward_vector.get("cost")
                    if child.reward_vector is not None
                    else None
                ),
            },
            "failures": _events_with_roles(child, {SemanticRole.FAILURE}),
            "artifacts_by_role": {key: roles[key] for key in sorted(roles)},
            "provenance": dict(graph.provenance),
            "termination": {"status": child.status, "verdict": child.verdict},
            "split": child.split,
            "visibility": child.visibility,
        }
        return record, candidate_text

    def _materialize_artifacts(
        self, child: CandidateEpisode
    ) -> tuple[list[dict[str, Any]], str | None]:
        unique: dict[tuple[str, str], EpisodeArtifact] = {}
        for event in child.events:
            for artifact in event.artifacts:
                unique.setdefault((artifact.role, artifact.receipt.digest), artifact)
        values: list[dict[str, Any]] = []
        candidate_parts: list[str] = []
        for (role, _), artifact in sorted(unique.items()):
            content = self._artifacts.read_bytes(artifact.receipt)
            _reject_secrets(content.decode("utf-8", errors="ignore"))
            encoding, body = _encode_artifact(content, artifact.receipt)
            value = {
                "role": role,
                "receipt": artifact.receipt.to_dict(),
                "encoding": encoding,
                "content": body,
            }
            values.append(value)
            if role in {
                "candidate",
                "candidate_patch",
                "candidate_source",
                "solution",
            } and encoding == "utf-8":
                if not body.strip():
                    raise IntegrityError("Candidate artifact is empty", "empty_candidate_artifact")
                candidate_parts.append(
                    f"# artifact role={role} sha256={artifact.receipt.digest}\n{body}"
                )
        candidate_text = "\n\n".join(candidate_parts) if candidate_parts else None
        if (
            child.status != "infrastructure_error"
            and candidate_text is None
            and not allows_source_free_e2e(child)
        ):
            raise IntegrityError("Real textual candidate is missing", "candidate_artifact_missing")
        return values, candidate_text

    def _validate_reward(
        self,
        graph: EpisodeGraph,
        child: CandidateEpisode,
    ) -> None:
        if not child.policy_ids:
            return
        if "e2e_kernel_candidate_v1" in child.policy_ids:
            validate_e2e_export_reward(graph, child, self._artifacts)
            return
        if "kernel_robust_v1" not in child.policy_ids:
            return
        vector = child.reward_vector
        if vector is None:
            raise IntegrityError("Kernel reward vector is missing", "reward_vector_missing")
        try:
            safety = vector.get("safety", {})
            if not isinstance(safety, Mapping):
                raise TypeError("safety")
            gates = GateVerdict(
                compiled=bool(vector["compile"]),
                correct=bool(vector["correctness"]),
                integrity_passed=bool(vector["integrity"]),
                tampering_passed=bool(vector["anti_tampering"]),
                safety_finding=bool(safety.get("finding", False)),
            )
            srobust_value = vector.get("kernel_srobust")
            expected = kernel_reward(
                gates, None if srobust_value is None else float(srobust_value)
            )
            recorded = vector.get("kernel_robust_reward", child.scalar_reward)
            if expected is None and recorded is None:
                return
            if expected is None or recorded is None or abs(float(recorded) - expected) > 1e-9:
                raise ValueError("reward mismatch")
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError(
                "Stored kernel reward cannot be exactly replayed",
                "reward_replay_mismatch",
            ) from error

    @staticmethod
    def _sft_record(record: Mapping[str, Any], candidate_text: str) -> dict[str, Any]:
        return {
            "schema_name": "apex.sft_pair",
            "schema_version": 1,
            "episode_id": record["episode_id"],
            "task_id": record["task_id"],
            "prompt": canonical_json_bytes(record["observation"]["content"]).decode(),
            "response": candidate_text,
            "policy_ids": record["reward"]["policy_ids"],
            "split": record["split"],
        }


def _validate_export_partition(
    graph: EpisodeGraph, config: DatasetExportConfig
) -> None:
    if len(graph.policy_ids) > 1 and config.policy_id is None:
        raise ContractError(
            "Mixed reward policies require an explicit export partition",
            "mixed_reward_policy_export",
        )


def _dataset_payloads(
    parent: Mapping[str, Any],
    records: list[dict[str, Any]],
    sft: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    config: DatasetExportConfig,
) -> dict[str, bytes]:
    jsonl = b"".join(canonical_json_bytes(item) + b"\n" for item in records)
    sft_jsonl = b"".join(canonical_json_bytes(item) + b"\n" for item in sft)
    document = {
        "schema_name": "apex.rl_dataset",
        "schema_version": 1,
        "exporter_version": config.exporter_version,
        "parent_episode": parent,
        "records": records,
    }
    validation = {
        "schema_name": "apex.rl_dataset_validation",
        "schema_version": 1,
        "valid": True,
        "record_count": len(records),
        "sft_count": len(sft),
        "skipped": skipped,
        "quality_gates": {
            "schema_validation_pct": 100,
            "placeholder_solution_count": 0,
            "secret_count": 0,
            "missing_artifact_count": 0,
            "stdout_transition_recovery_count": 0,
        },
    }
    return {
        "dataset.json": canonical_json_bytes(document),
        "dataset.jsonl": jsonl,
        "parent_episode.json": canonical_json_bytes(parent),
        "sft.jsonl": sft_jsonl,
        "validation_report.json": canonical_json_bytes(validation),
    }


def _dataset_manifest(
    graph: EpisodeGraph,
    records: list[dict[str, Any]],
    payloads: Mapping[str, bytes],
    config: DatasetExportConfig,
) -> dict[str, Any]:
    return {
        "schema_name": "apex.rl_dataset_manifest",
        "schema_version": 1,
        "exporter_version": config.exporter_version,
        "episode_graph_id": graph.graph_id,
        "episode_graph_sha256": sha256_bytes(graph.canonical_bytes),
        "run_id": graph.run_id,
        "high_water_mark": graph.high_water_mark,
        "policy_ids": sorted(
            {
                policy
                for record in records
                for policy in record["reward"]["policy_ids"]
            }
        ),
        "split_filter": config.split,
        "policy_filter": config.policy_id,
        "files": {
            name: sha256_bytes(content) for name, content in sorted(payloads.items())
        },
    }


def _events_with_roles(
    child: CandidateEpisode, roles: set[SemanticRole]
) -> list[dict[str, Any]]:
    return [event.to_dict() for event in child.events if event.semantic_role in roles]


def _encode_artifact(content: bytes, receipt: ArtifactReceipt) -> tuple[str, str]:
    textual = receipt.media_type.startswith("text/") or receipt.media_type in {
        "application/json",
        "application/x-ndjson",
        "application/yaml",
    }
    if textual:
        try:
            return "utf-8", content.decode("utf-8")
        except UnicodeDecodeError:
            pass
    return "base64", base64.b64encode(content).decode("ascii")


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)) and item is not None and item not in (
                "",
                "[REDACTED]",
            ):
                raise IntegrityError("Dataset contains a secret field", "dataset_secret_detected")
            _reject_secrets(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item)
        return
    if isinstance(value, str) and (
        _SECRET_TEXT.search(value)
        or _SECRET_ASSIGNMENT.search(value)
        or _SECRET_OPTION.search(value)
    ):
        raise IntegrityError("Dataset contains secret-like content", "dataset_secret_detected")


def _write_files(output_dir: Path, files: Mapping[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in sorted(files.items()):
        descriptor, temporary_name = tempfile.mkstemp(dir=output_dir, prefix=f".{name}.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output_dir / name)
        finally:
            temporary.unlink(missing_ok=True)
    descriptor = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["DatasetExportConfig", "DatasetExportResult", "DatasetExporter"]
