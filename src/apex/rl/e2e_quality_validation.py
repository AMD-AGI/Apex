"""Offline validation of evaluator-owned quality artifacts for E2E RL."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from apex.benchmark.quality import PRIMARY_METRICS
from apex.core import ContractError, IntegrityError
from apex.storage import ArtifactReceipt, ArtifactStore

from .models import EpisodeEvent


_QUALITY_ROLES = frozenset(
    {"quality_result", "quality_sample", "quality_raw_artifact"}
)


def validate_quality_evidence(
    event: EpisodeEvent,
    artifacts: ArtifactStore,
    normalized: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> float:
    """Bind normalized accuracy to exact raw evaluator result bytes."""

    normalized_quality = _mapping(normalized.get("quality"), "quality")
    expected_keys = {
        "required",
        "kind",
        "passed",
        "hard_failure",
        "metrics",
        "primary_metrics",
        "error",
        "outcome_digest",
        "sample_set_digest",
    }
    if quality.get("schema") != "apex.e2e-quality-evidence/v1" or any(
        quality.get(key) != normalized_quality.get(key) for key in expected_keys
    ):
        _reject("Normalized quality differs from typed quality evidence")
    if (
        quality.get("required") is not True
        or quality.get("kind") != "lm_eval"
        or quality.get("passed") is not True
        or quality.get("hard_failure") is not False
        or quality.get("error") is not None
    ):
        _reject("Scoring benchmark lacks passing required quality evidence")
    result_receipts = _receipt_documents(quality.get("result_receipts"))
    raw_receipts = _receipt_documents(quality.get("raw_artifact_receipts"))
    bound_results = _event_role_receipts(event, {"quality_result"})
    bound_all = _event_role_receipts(event, _QUALITY_ROLES)
    if (
        result_receipts != bound_results
        or _receipt_union(result_receipts, raw_receipts) != bound_all
    ):
        _reject("Quality receipt A differs from bound raw artifact B")
    if len(bound_results) != 1:
        _reject("Scoring quality requires exactly one result artifact")
    raw_metrics = _parse_quality_result(_read_json(artifacts, bound_results[0]))
    if raw_metrics != _metric_documents(quality.get("metrics")):
        _reject("Typed quality metrics differ from the raw evaluator result")
    primary = _primary_metrics(raw_metrics)
    if primary != _metric_documents(quality.get("primary_metrics")):
        _reject("Typed primary quality differs from raw evaluator evidence")
    selected = _first_primary(raw_metrics)
    return float(selected["value"])


def validate_quality_hard_failure(
    event: EpisodeEvent,
    artifacts: ArtifactStore,
    normalized: Mapping[str, Any],
    quality: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
) -> None:
    """Rebuild an explicit quality failure from raw evaluator artifacts."""

    normalized_quality = _mapping(normalized.get("quality"), "quality")
    expected_keys = {
        "required",
        "kind",
        "passed",
        "hard_failure",
        "metrics",
        "primary_metrics",
        "error",
        "outcome_digest",
        "sample_set_digest",
    }
    if quality.get("schema") != "apex.e2e-quality-evidence/v1" or any(
        quality.get(key) != normalized_quality.get(key) for key in expected_keys
    ):
        _reject("Normalized quality failure differs from typed evidence")
    if (
        quality.get("required") is not True
        or quality.get("passed") is not False
        or quality.get("hard_failure") is not True
        or quality.get("error") != "quality_gate_not_passed"
    ):
        _reject("Quality failure is not an evaluator-confirmed hard gate")
    kind = quality.get("kind")
    if kind == "lm_eval":
        _validate_lm_eval_failure(event, artifacts, quality, quality_gate)
    elif kind == "framework_quality_gate":
        _validate_framework_failure(quality, quality_gate)
    else:
        _reject("Quality failure kind is unsupported")


def _validate_lm_eval_failure(
    event: EpisodeEvent,
    artifacts: ArtifactStore,
    quality: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
) -> None:
    result_receipts = _receipt_documents(quality.get("result_receipts"))
    raw_receipts = _receipt_documents(quality.get("raw_artifact_receipts"))
    if (
        result_receipts != _event_role_receipts(event, {"quality_result"})
        or _receipt_union(result_receipts, raw_receipts)
        != _event_role_receipts(event, _QUALITY_ROLES)
        or len(result_receipts) != 1
    ):
        _reject("Failed quality receipts differ from bound raw artifacts")
    raw_metrics = _parse_quality_result(_read_json(artifacts, result_receipts[0]))
    if raw_metrics != _metric_documents(quality.get("metrics")):
        _reject("Failed quality metrics differ from raw evaluator evidence")
    if _primary_metrics(raw_metrics) != _metric_documents(
        quality.get("primary_metrics")
    ):
        _reject("Failed primary quality differs from raw evaluator evidence")
    gate = quality_gate
    if (
        gate.get("requested") is not True
        or gate.get("status") != "failed"
        or gate.get("passed") is not False
        or gate.get("evidence_present") is not True
        or gate.get("errors") != []
        or gate.get("error_count") != 0
        or gate.get("errors_truncated") is not False
        or gate.get("tasks_truncated") is not False
        or gate.get("result_artifacts_truncated") is not False
        or gate.get("outcome_digest") != quality.get("outcome_digest")
        or gate.get("sample_set_digest") != quality.get("sample_set_digest")
        or not _gate_receipts_match(gate.get("result_artifact_receipts"), result_receipts)
        or not _gate_receipts_match(gate.get("sample_artifact_receipts"), raw_receipts)
    ):
        _reject("Raw quality gate does not prove an explicit bound failure")


def _validate_framework_failure(
    quality: Mapping[str, Any], gate: Mapping[str, Any]
) -> None:
    expected = [
        {
            "task": "framework_quality_gate",
            "name": name,
            "value": float(value),
            "higher_is_better": _higher_is_better(name),
        }
        for name, value in sorted(gate.items())
        if name not in {"passed", "skipped"}
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]
    if (
        gate.get("passed") is not False
        or gate.get("skipped") is True
        or not expected
        or expected != _metric_documents(quality.get("metrics"))
    ):
        _reject("Raw framework gate does not prove an explicit failure")


def _gate_receipts_match(
    raw: object, receipts: Sequence[ArtifactReceipt]
) -> bool:
    if not isinstance(raw, list) or len(raw) != len(receipts):
        return False
    expected = sorted((item.digest, item.size) for item in receipts)
    observed = sorted(
        (item.get("sha256"), item.get("size_bytes"))
        for item in raw
        if isinstance(item, Mapping)
    )
    return len(observed) == len(raw) and observed == expected


def metric_documents(value: object) -> list[dict[str, Any]]:
    """Validate and normalize typed quality metric documents."""

    return _metric_documents(value)


def _event_role_receipts(
    event: EpisodeEvent,
    roles: set[str] | frozenset[str],
) -> tuple[ArtifactReceipt, ...]:
    unique = {
        item.receipt.digest: item.receipt
        for item in event.artifacts
        if item.role in roles
    }
    return tuple(unique[key] for key in sorted(unique))


def _receipt_documents(value: object) -> tuple[ArtifactReceipt, ...]:
    if not isinstance(value, list):
        _reject("Quality receipt list is invalid")
    try:
        receipts = tuple(ArtifactReceipt.from_dict(dict(item)) for item in value)
    except (ContractError, TypeError, ValueError) as error:
        raise IntegrityError(
            "Quality receipt list is invalid",
            "e2e_measurement_evidence_mismatch",
        ) from error
    unique = {item.digest: item for item in receipts}
    if len(unique) != len(receipts):
        _reject("Quality receipt list contains duplicates")
    return tuple(unique[key] for key in sorted(unique))


def _receipt_union(
    first: Sequence[ArtifactReceipt],
    second: Sequence[ArtifactReceipt],
) -> tuple[ArtifactReceipt, ...]:
    unique = {item.digest: item for item in (*first, *second)}
    return tuple(unique[key] for key in sorted(unique))


def _read_json(
    artifacts: ArtifactStore,
    receipt: ArtifactReceipt,
) -> Mapping[str, Any]:
    try:
        value = json.loads(artifacts.read_bytes(receipt))
    except json.JSONDecodeError as error:
        raise IntegrityError(
            "Raw quality result is not JSON",
            "e2e_measurement_evidence_mismatch",
        ) from error
    if not isinstance(value, Mapping):
        _reject("Raw quality result is not an object")
    return value


def _parse_quality_result(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = _mapping(value.get("results"), "quality results")
    metrics: list[dict[str, Any]] = []
    for task, raw_values in sorted(results.items()):
        values = _mapping(raw_values, "task quality metrics")
        for name, raw in sorted(values.items()):
            if not isinstance(task, str) or not isinstance(name, str):
                _reject("Raw quality metric identity is invalid")
            if "stderr" in name.split(",", 1)[0].lower():
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            number = float(raw)
            if math.isfinite(number):
                metrics.append(
                    {
                        "task": task,
                        "name": name,
                        "value": number,
                        "higher_is_better": _higher_is_better(name),
                    }
                )
    return metrics


def _primary_metrics(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task in sorted({str(item["task"]) for item in metrics}):
        candidates = tuple(item for item in metrics if item["task"] == task)
        selected.append(dict(_first_primary(candidates)))
    return selected


def _first_primary(metrics: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = tuple(item for item in metrics if item.get("higher_is_better") is True)
    for name in PRIMARY_METRICS:
        match = next((item for item in eligible if item.get("name") == name), None)
        if match is not None:
            return match
    if eligible:
        return eligible[0]
    _reject("Quality evidence has no higher-is-better primary metric")


def _metric_documents(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        _reject("Typed quality metrics are invalid")
    return [dict(item) for item in value]


def _higher_is_better(name: str) -> bool:
    lowered = name.split(",", 1)[0].lower()
    return not any(
        marker in lowered
        for marker in ("loss", "perplexity", "ppl", "error", "wer", "cer")
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _reject(f"E2E {label} is not an object")
    return value


def _reject(message: str) -> None:
    raise IntegrityError(message, "e2e_measurement_evidence_mismatch")


__all__ = [
    "metric_documents",
    "validate_quality_evidence",
    "validate_quality_hard_failure",
]
