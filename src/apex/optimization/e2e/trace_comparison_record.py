"""CAS publication and event projection for terminal trace comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apex.core import IntegrityError, canonical_json_bytes
from apex.ports import TraceComparisonResult
from apex.storage import ArtifactReceipt, ArtifactStore


@dataclass(frozen=True, slots=True)
class TraceComparisonRecord:
    receipt: ArtifactReceipt
    event_payload: Mapping[str, object]


def persist_trace_comparison(
    store: ArtifactStore, result: TraceComparisonResult
) -> TraceComparisonRecord:
    outputs = tuple(
        (path, store.put_file(path, media_type=_media_type(path)))
        for path in result.artifacts
        if path.is_file() and not path.is_symlink()
    )
    if len(outputs) != len(result.artifacts):
        raise IntegrityError(
            "Trace comparison output is unsafe", "invalid_trace_comparison_output"
        )
    output_bindings = tuple(
        _binding(
            result.artifact_roles.get(
                str(path.resolve()), "terminal_trace_comparison_output"
            ),
            receipt,
        )
        for path, receipt in outputs
    )
    document = {
        "schema": "apex.tracelens-terminal-comparison-result/v2",
        "status": result.status.value,
        "reason_code": result.reason_code,
        "reward_eligible": result.reward_eligible,
        "summary": dict(result.summary),
        "output_artifacts": list(output_bindings),
    }
    receipt = store.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )
    event_payload = {
        "tool": "tracelens_terminal_comparison",
        "succeeded": result.status.value in {"succeeded", "partial"},
        "status": result.status.value,
        "reason_code": result.reason_code,
        "evidence_class": "diagnostic",
        "reward_eligible": False,
        "artifacts": [
            _binding("terminal_trace_comparison", receipt),
            *output_bindings,
        ],
    }
    return TraceComparisonRecord(receipt, event_payload)


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def _media_type(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(path.suffix.lower(), "application/octet-stream")


__all__ = ["TraceComparisonRecord", "persist_trace_comparison"]
