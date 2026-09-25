"""Trainer-neutral, fail-closed views over one exported Apex RL dataset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes

from .export_sanitization import redact_host_paths


_FILES = frozenset(
    {
        "dataset.json",
        "dataset.jsonl",
        "parent_episode.json",
        "sft.jsonl",
        "validation_report.json",
        "export_manifest.json",
    }
)
_POLICY_TASK_KIND = {
    "kernel_robust_v1": "single_kernel",
    "e2e_throughput_qos_v1": "e2e_kernel_only",
}


@dataclass(frozen=True, slots=True)
class ReferenceDataset:
    """Verified JSON-native dataset with three reference training projections."""

    root: Path
    manifest: Mapping[str, Any]
    parent: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]

    def terminal_episode(self) -> dict[str, Any]:
        """Return parent task reward without aggregating child attempt rewards."""

        parent = self.parent
        return {
            "schema_name": "apex.rl_terminal_episode",
            "schema_version": 1,
            "episode_graph_id": self.manifest["episode_graph_id"],
            "episode_id": parent["episode_id"],
            "run_id": parent["run_id"],
            "task_kind": _task_kind(parent, self.manifest),
            "terminal_status": parent["terminal_status"],
            "scalar_reward": parent["task_reward"],
            "reward_vector": parent["reward_vector"],
            "reward_policy_id": parent["reward_policy_id"],
            "reward_policy_digest": parent["reward_policy_digest"],
            "reward_source_receipt": parent["reward_source_receipt"],
            "raw_measurement_receipts": parent["raw_measurement_receipts"],
            "trainability": parent["trainability"],
            "untrainable_reason": parent["untrainable_reason"],
            "child_episode_ids": parent["child_episode_ids"],
            "causal_event_ids": [event["event_id"] for event in parent["events"]],
        }

    def attempt_transitions(
        self, *, advantage_mode: str = "episode_mean"
    ) -> tuple[dict[str, Any], ...]:
        """Return ordered attempts with an explicit, simple reference advantage."""

        if advantage_mode not in {"episode_mean", "zero"}:
            raise ContractError(
                "Unsupported reference advantage mode",
                "invalid_advantage_mode",
            )
        rewards = tuple(
            float(record["reward"]["scalar"])
            for record in self.records
            if record["reward"]["scalar"] is not None
        )
        baseline = sum(rewards) / len(rewards) if rewards and advantage_mode == "episode_mean" else 0.0
        terminal = self.terminal_episode()
        result: list[dict[str, Any]] = []
        for index, record in enumerate(_ordered_records(self.records)):
            scalar = record["reward"]["scalar"]
            result.append(
                {
                    "schema_name": "apex.rl_attempt_transition",
                    "schema_version": 1,
                    "episode_graph_id": self.manifest["episode_graph_id"],
                    "sequence_index": index,
                    "episode_id": record["episode_id"],
                    "parent_episode_id": record["parent_episode_id"],
                    "run_id": record["run_id"],
                    "attempt_id": record["attempt_id"],
                    "candidate_id": record["candidate_id"],
                    "opportunity_id": record["opportunity_id"],
                    "state_generation": record["state_generation"],
                    "anchor_generation": record["anchor_generation"],
                    "observation": record["observation"],
                    "actions": record["actions"],
                    "tools": record["tools"],
                    "outcomes": record["outcomes"],
                    "decisions": record["decisions"],
                    "failures": record["failures"],
                    "termination": record["termination"],
                    "attempt_reward": scalar,
                    "attempt_reward_vector": record["reward"]["vector"],
                    "terminal_task_reward": terminal["scalar_reward"],
                    "advantage_mode": advantage_mode,
                    "advantage_baseline": baseline if scalar is not None else None,
                    "advantage": float(scalar) - baseline if scalar is not None else None,
                    "performance_trainable": scalar is not None,
                    "costs": record["costs"],
                    "causal_event_ids": record["causal_event_ids"],
                    "split": record["split"],
                    "visibility": record["visibility"],
                }
            )
        return tuple(result)

    def supervision_view(self) -> tuple[dict[str, Any], ...]:
        """Return semantic tool/decision targets without prompt/response flattening."""

        values: list[dict[str, Any]] = []
        for record in _ordered_records(self.records):
            tools = _ordered_events(record["tools"])
            decisions = _ordered_events(record["decisions"])
            if not tools and not decisions:
                continue
            target_ids = [event["event_id"] for event in (*tools, *decisions)]
            values.append(
                {
                    "schema_name": "apex.rl_tool_decision_supervision",
                    "schema_version": 1,
                    "episode_graph_id": self.manifest["episode_graph_id"],
                    "episode_id": record["episode_id"],
                    "parent_episode_id": record["parent_episode_id"],
                    "run_id": record["run_id"],
                    "attempt_id": record["attempt_id"],
                    "observation": record["observation"],
                    "action_context": record["actions"],
                    "tool_targets": list(tools),
                    "decision_targets": list(decisions),
                    "target_event_ids": target_ids,
                    "outcome_context": record["outcomes"],
                    "termination": record["termination"],
                    "reward": record["reward"],
                    "split": record["split"],
                    "visibility": record["visibility"],
                }
            )
        return tuple(values)


class ReferenceDatasetLoader:
    """Load only an intact deterministic export produced by ``DatasetExporter``."""

    def load(self, root: Path) -> ReferenceDataset:
        root = Path(root)
        _validate_inventory(root)
        manifest = _json_file(root / "export_manifest.json")
        _validate_manifest(root, manifest)
        dataset = _json_file(root / "dataset.json")
        parent = _json_file(root / "parent_episode.json")
        validation = _json_file(root / "validation_report.json")
        records = _jsonl_file(root / "dataset.jsonl")
        sft = _jsonl_file(root / "sft.jsonl")
        _validate_documents(manifest, dataset, parent, validation, records, sft)
        return ReferenceDataset(root.resolve(), manifest, parent, records)


def _validate_inventory(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ContractError("RL export directory is invalid", "invalid_rl_export")
    names: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("RL export contains unsafe entries", "rl_export_tampered")
        names.add(path.name)
    if names != _FILES:
        raise IntegrityError("RL export inventory differs", "rl_export_tampered")


def _validate_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_name",
        "schema_version",
        "exporter_version",
        "episode_graph_id",
        "episode_graph_sha256",
        "run_id",
        "high_water_mark",
        "policy_ids",
        "split_filter",
        "policy_filter",
        "visibility_policy",
        "redaction_policy",
        "license_policy",
        "retention_policy",
        "summary",
        "files",
    }
    files = manifest.get("files")
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_name") != "apex.rl_dataset_manifest"
        or manifest.get("schema_version") != 2
        or not isinstance(files, Mapping)
        or set(files) != _FILES - {"export_manifest.json"}
    ):
        raise IntegrityError("RL export manifest is invalid", "rl_export_tampered")
    for name, digest in files.items():
        content = (root / str(name)).read_bytes()
        if digest != sha256_bytes(content):
            raise IntegrityError("RL export file digest differs", "rl_export_tampered")


def _validate_documents(
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
    parent: Mapping[str, Any],
    validation: Mapping[str, Any],
    records: tuple[Mapping[str, Any], ...],
    sft: tuple[Mapping[str, Any], ...],
) -> None:
    if (
        dataset.get("schema_name") != "apex.rl_dataset"
        or dataset.get("schema_version") != 2
        or dataset.get("parent_episode") != parent
        or dataset.get("records") != list(records)
        or manifest.get("run_id") != parent.get("run_id")
        or validation.get("schema_name") != "apex.rl_dataset_validation"
        or validation.get("schema_version") != 2
        or validation.get("valid") is not True
        or validation.get("record_count") != len(records)
        or validation.get("sft_count") != len(sft)
    ):
        raise IntegrityError("RL export projections disagree", "rl_export_tampered")
    _validate_export_policies(manifest, records)
    if (
        redact_host_paths(parent) != parent
        or redact_host_paths(records) != list(records)
        or redact_host_paths(sft) != list(sft)
    ):
        raise IntegrityError(
            "RL export contains an unredacted host path", "rl_export_tampered"
        )
    _validate_parent(parent)
    child_ids = set(parent["child_episode_ids"])
    seen: set[str] = set()
    for record in records:
        _validate_record(record, parent, child_ids, seen)


def _validate_export_policies(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    expected_ids = {
        "visibility_policy": "public_episode_only_fail_closed_v1",
        "redaction_policy": "host_absolute_path_redaction_v1",
        "license_policy": "source_terms_preserved_no_relicense_v1",
        "retention_policy": "manifest_bound_export_retention_v1",
    }
    for field, policy_id in expected_ids.items():
        policy = manifest.get(field)
        if (
            not isinstance(policy, Mapping)
            or policy.get("policy_id") != policy_id
            or not isinstance(policy.get("summary"), str)
            or not str(policy["summary"]).strip()
        ):
            raise IntegrityError("RL export policy is invalid", "rl_export_tampered")
    split_counts = _counts(records, "split")
    visibility_counts = _counts(records, "visibility")
    artifacts = tuple(
        artifact
        for record in records
        for values in record.get("artifacts_by_role", {}).values()
        for artifact in values
    )
    expected_summary = {
        "record_count": len(records),
        "split_counts": split_counts,
        "visibility_counts": visibility_counts,
        "artifact_count": len(artifacts),
        "redacted_artifact_count": sum(
            item.get("redaction_policy_id") == "host_absolute_path_redaction_v1"
            for item in artifacts
        ),
    }
    if (
        visibility_counts != {"public": len(records)}
        or manifest.get("summary") != expected_summary
    ):
        raise IntegrityError("RL export summary is invalid", "rl_export_tampered")


def _counts(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        value = str(record.get(field))
        result[value] = result.get(value, 0) + 1
    return {key: result[key] for key in sorted(result)}


def _validate_parent(parent: Mapping[str, Any]) -> None:
    required = {
        "episode_id",
        "run_id",
        "events",
        "child_episode_ids",
        "terminal_status",
        "task_reward",
        "reward_vector",
        "reward_policy_id",
        "reward_policy_digest",
        "reward_source_receipt",
        "raw_measurement_receipts",
        "trainability",
        "untrainable_reason",
    }
    scalar = parent.get("task_reward")
    if not required <= set(parent) or not isinstance(parent.get("events"), list):
        raise IntegrityError("RL parent episode is invalid", "rl_export_tampered")
    _validate_scalar(scalar)
    if parent.get("trainability") == "complete" and scalar is None:
        raise IntegrityError("Trainable parent lacks task reward", "rl_export_tampered")


def _validate_record(
    record: Mapping[str, Any],
    parent: Mapping[str, Any],
    child_ids: set[str],
    seen: set[str],
) -> None:
    required = {
        "schema_name",
        "schema_version",
        "episode_id",
        "parent_episode_id",
        "run_id",
        "attempt_id",
        "causal_event_ids",
        "observation",
        "actions",
        "tools",
        "outcomes",
        "decisions",
        "reward",
        "costs",
        "failures",
        "termination",
        "split",
        "visibility",
    }
    episode_id = record.get("episode_id")
    reward = record.get("reward")
    if (
        not required <= set(record)
        or record.get("schema_name") != "apex.rl_transition"
        or record.get("schema_version") != 1
        or episode_id in seen
        or episode_id not in child_ids
        or record.get("parent_episode_id") != parent["episode_id"]
        or record.get("run_id") != parent["run_id"]
        or not isinstance(reward, Mapping)
    ):
        raise IntegrityError("RL transition is invalid", "rl_export_tampered")
    seen.add(str(episode_id))
    _validate_scalar(reward.get("scalar"))
    causal = record.get("causal_event_ids")
    if not isinstance(causal, list) or len(causal) != len(set(causal)):
        raise IntegrityError("RL transition lineage is invalid", "rl_export_tampered")
    for event in _all_semantic_events(record):
        if event.get("event_id") not in causal:
            raise IntegrityError("RL transition event is unbound", "rl_export_tampered")


def _validate_scalar(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrityError("RL reward scalar is invalid", "rl_export_tampered")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 320.0:
        raise IntegrityError("RL reward scalar is invalid", "rl_export_tampered")


def _task_kind(parent: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    policies = set(manifest.get("policy_ids", ()))
    policy = parent.get("reward_policy_id")
    if isinstance(policy, str):
        policies.add(policy)
    kinds = {_POLICY_TASK_KIND[value] for value in policies if value in _POLICY_TASK_KIND}
    if len(kinds) != 1:
        raise IntegrityError("RL task kind is ambiguous", "rl_export_tampered")
    return kinds.pop()


def _all_semantic_events(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    values: list[Mapping[str, Any]] = []
    for key in ("observations", "actions", "tools", "outcomes", "decisions", "failures"):
        raw = record.get(key, ())
        if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
            raise IntegrityError("RL semantic events are invalid", "rl_export_tampered")
        values.extend(raw)
    return tuple(values)


def _ordered_events(events: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(sorted(events, key=lambda event: (int(event["sequence"]), event["event_id"])))


def _ordered_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    def sequence(record: Mapping[str, Any]) -> tuple[int, str]:
        events = _all_semantic_events(record)
        first = min((int(event["sequence"]) for event in events), default=2**63 - 1)
        return first, str(record["attempt_id"])

    return tuple(sorted(records, key=sequence))


def _json_file(path: Path) -> Mapping[str, Any]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError("RL export JSON is invalid", "rl_export_tampered") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != content:
        raise IntegrityError("RL export JSON is not canonical", "rl_export_tampered")
    return value


def _jsonl_file(path: Path) -> tuple[Mapping[str, Any], ...]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise IntegrityError("RL export JSONL is not canonical", "rl_export_tampered")
    values: list[Mapping[str, Any]] = []
    for line in content.splitlines():
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise IntegrityError("RL export JSONL is invalid", "rl_export_tampered") from error
        if not isinstance(value, Mapping) or canonical_json_bytes(value) != line:
            raise IntegrityError("RL export JSONL is not canonical", "rl_export_tampered")
        values.append(value)
    return tuple(values)


__all__ = ["ReferenceDataset", "ReferenceDatasetLoader"]
