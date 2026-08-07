"""Deterministic human and machine report projections over EpisodeGraph."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import canonical_json_bytes, sha256_bytes
from apex.rl import EpisodeGraph, EvidenceClass, SemanticRole


_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)$", re.I)
_SECRET_TEXT = re.compile(
    r"(?:sk-(?:ant-)?[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|"
    r"github_pat_[A-Za-z0-9_]{12,}|Bearer\s+[A-Za-z0-9._~+/-]{12,}|"
    r"https?://[^\s/@:]+:[^\s/@]+@)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)"
    r"\s*[:=]\s*[\"']?(?!\[REDACTED\])\S{4,}",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ReportProjection:
    """Byte-stable report values rebuilt solely from canonical evidence."""

    document: Mapping[str, Any]
    markdown: str

    @property
    def json_bytes(self) -> bytes:
        return canonical_json_bytes(self.document)

    @property
    def markdown_bytes(self) -> bytes:
        return self.markdown.encode("utf-8")

    @property
    def digest(self) -> str:
        return sha256_bytes(self.json_bytes)


def build_report(graph: EpisodeGraph) -> ReportProjection:
    """Project headline facts without promoting self-reported measurements."""

    attempts: list[dict[str, Any]] = []
    measured_results: list[dict[str, Any]] = []
    artifact_index: dict[str, dict[str, Any]] = {}
    for event in graph.parent.events:
        _index_artifacts(artifact_index, event)
        if _is_headline_measurement(event):
            measured_results.append(_measurement_row(event, attempt_id=None))
    for child in graph.children:
        attempts.append(
            _candidate_row(child, artifact_index, measured_results)
        )
    document = _report_document(
        graph, attempts, measured_results, _artifact_rows(artifact_index)
    )
    return ReportProjection(document=document, markdown=_render_markdown(document))


def _candidate_row(
    child: Any,
    artifact_index: dict[str, dict[str, Any]],
    measured_results: list[dict[str, Any]],
) -> dict[str, Any]:
    costs = []
    failures = []
    for event in child.events:
        _index_artifacts(artifact_index, event)
        if event.semantic_role is SemanticRole.COST:
            costs.append(_safe_payload(event.payload))
        if event.semantic_role is SemanticRole.FAILURE:
            failures.append(_safe_failure(event.payload))
        if _is_headline_measurement(event):
            measured_results.append(_measurement_row(event, child.attempt_id))
    return {
        "attempt_id": child.attempt_id,
        "candidate_id": child.candidate_id,
        "task_id": child.task_id,
        "kernel_id": child.kernel_id,
        "status": child.status,
        "verdict": child.verdict,
        "state_generation": child.state_generation,
        "anchor_generation": child.anchor_generation,
        "context_packet_id": child.context_packet_id,
        "context_packet_sha256": (
            child.context_packet_receipt.digest
            if child.context_packet_receipt is not None
            else None
        ),
        "scalar_reward": child.scalar_reward,
        "reward_vector": _redact(child.reward_vector),
        "reward_evidence_classes": sorted(
            {
                event.evidence_class.value
                for event in child.events
                if event.semantic_role is SemanticRole.REWARD
            }
        ),
        "policy_ids": list(child.policy_ids),
        "costs": costs,
        "failures": failures,
        "trainability": child.trainability,
        "validation_reasons": list(child.validation_reasons),
    }


def _measurement_row(event: Any, attempt_id: str | None) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "event_id": event.event_id,
        "metrics": _safe_metrics(event.payload.get("metrics", event.payload)),
    }


def _artifact_rows(
    artifact_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "digest": digest,
            "receipt": value["receipt"],
            "roles": sorted(set(value["roles"])),
        }
        for digest, value in sorted(artifact_index.items())
    ]


def _report_document(
    graph: EpisodeGraph,
    attempts: list[dict[str, Any]],
    measured_results: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_name": "apex.run_report",
        "schema_version": 1,
        "episode_graph_id": graph.graph_id,
        "run_id": graph.run_id,
        "high_water_mark": graph.high_water_mark,
        "terminal_status": graph.parent.terminal_status,
        "workload_id": graph.parent.workload_id,
        "task_id": graph.parent.task_id,
        "workload_state_hash": graph.workload_state_hash,
        "policy_ids": list(graph.policy_ids),
        "provenance": _redact(graph.provenance),
        "headline_measured_results": measured_results,
        "attempts": attempts,
        "artifacts": artifacts,
        "summary": {
            "attempt_count": len(attempts),
            "kept_count": sum(item["verdict"] == "keep" for item in attempts),
            "reverted_count": sum(
                item["verdict"] in {"revert", "reject"} for item in attempts
            ),
            "failure_count": sum(bool(item["failures"]) for item in attempts),
            "complete_episode_count": sum(
                item["trainability"] == "complete" for item in attempts
            ),
        },
    }


def _index_artifacts(artifact_index: dict[str, dict[str, Any]], event: Any) -> None:
    for artifact in event.artifacts:
        artifact_index.setdefault(
            artifact.receipt.digest,
            {"receipt": artifact.receipt.to_dict(), "roles": []},
        )["roles"].append(artifact.role)


def _is_headline_measurement(event: Any) -> bool:
    return (
        event.evidence_class is EvidenceClass.MEASURED
        and event.semantic_role is SemanticRole.OUTCOME
        and event.event_type.replace(".", "_")
        in {"measurement_result", "e2e_result"}
    )


def _render_markdown(document: Mapping[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Apex optimization report",
        "",
        f"- Run: `{document['run_id']}`",
        f"- Episode graph: `{document['episode_graph_id']}`",
        f"- Journal high-water mark: `{document['high_water_mark']}`",
        f"- Terminal status: `{document['terminal_status']}`",
        f"- Attempts: {summary['attempt_count']} ({summary['kept_count']} kept, "
        f"{summary['reverted_count']} reverted, {summary['failure_count']} failures)",
        "",
        "## Candidate attempts",
        "",
        "| Attempt | Status | Verdict | Reward | Reward evidence | Policy | Context | RL validity |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for item in document["attempts"]:
        reward = "—" if item["scalar_reward"] is None else f"{item['scalar_reward']:.6g}"
        lines.append(
            "| {attempt} | {status} | {verdict} | {reward} | {evidence} | {policy} | {context} | {validity} |".format(
                attempt=_cell(item["attempt_id"]),
                status=_cell(item["status"]),
                verdict=_cell(item["verdict"] or "—"),
                reward=reward,
                evidence=_cell(", ".join(item["reward_evidence_classes"]) or "—"),
                policy=_cell(", ".join(item["policy_ids"]) or "—"),
                context=_cell(item["context_packet_sha256"] or "missing"),
                validity=_cell(item["trainability"]),
            )
        )
    lines.extend(["", "## Measured headline outcomes", ""])
    measured = document["headline_measured_results"]
    if not measured:
        lines.append("No evaluator-owned measured E2E outcome was committed.")
    else:
        for item in measured:
            metrics = json.dumps(item["metrics"], sort_keys=True, ensure_ascii=False)
            lines.append(f"- `{item['attempt_id'] or 'workload'}`: `{metrics}`")
    lines.extend(["", "## Provenance", "", "```json"])
    lines.append(json.dumps(document["provenance"], sort_keys=True, indent=2, ensure_ascii=False))
    lines.extend(["```", ""])
    return "\n".join(lines)


def _safe_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    allowed = {
        key: value
        for key, value in payload.items()
        if key in {"tokens", "gpu_seconds", "wall_seconds", "cost", "currency", "usage"}
    }
    return _redact(allowed)


def _safe_failure(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _redact(
        {
            key: payload[key]
            for key in ("reason_code", "error", "status", "retry", "phase")
            if key in payload
        }
    )


def _safe_metrics(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _redact(item)
        for key, item in sorted(value.items())
        if isinstance(item, (str, int, float, bool, type(None)))
        and not _SECRET_KEY.search(str(key))
    }


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if _SECRET_KEY.search(str(key)) else _redact(item)
            )
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_ASSIGNMENT.sub(
            "[REDACTED]", _SECRET_TEXT.sub("[REDACTED]", value)
        )
    return value


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = ["ReportProjection", "build_report"]
