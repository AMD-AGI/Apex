"""Independent replay of terminal paired observations from raw CAS bytes."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from apex.core import IntegrityError, sha256_json
from apex.evaluation import E2EObservation, E2EPairedMeasurement
from apex.storage import ArtifactReceipt, ArtifactStore

from .e2e_benchmark_validation import reject
from .models import EpisodeEvent


def validate_terminal_raw_evidence(
    event: EpisodeEvent,
    source: Mapping[str, Any],
    measurement: E2EPairedMeasurement,
    artifacts: ArtifactStore,
) -> None:
    """Recompute every terminal observation from raw report and quality files."""

    metadata = source.get("terminal_raw_artifacts")
    if not isinstance(metadata, list) or not metadata:
        reject("Terminal delivery lacks raw replay artifact metadata")
    receipts = _raw_receipts(event)
    if len(receipts) != len(metadata):
        reject("Terminal reward raw artifact count differs")
    groups: dict[str, list[tuple[Mapping[str, Any], ArtifactReceipt]]] = {}
    for item, receipt in zip(metadata, receipts, strict=True):
        if not isinstance(item, Mapping) or not _metadata_matches(item, receipt):
            reject("Terminal raw artifact metadata differs from CAS")
        groups.setdefault(str(item.get("measurement_receipt")), []).append(
            (item, receipt)
        )
    observations = tuple(
        item for window in measurement.windows for item in window.observations
    )
    if set(groups) != {item.measurement_receipt for item in observations}:
        reject("Terminal raw artifact lineage differs from paired observations")
    for observation in observations:
        _validate_observation(observation, groups[observation.measurement_receipt], artifacts)


def _validate_observation(
    observation: E2EObservation,
    evidence: Sequence[tuple[Mapping[str, Any], ArtifactReceipt]],
    artifacts: ArtifactStore,
) -> None:
    reports = tuple(item for item in evidence if item[0].get("role") == "benchmark_report")
    quality = tuple(item for item in evidence if item[0].get("role") == "quality_result")
    if len(reports) != 1 or not quality:
        reject("Terminal observation lacks raw report or quality evidence")
    report_meta, report_receipt = reports[0]
    report = _json(artifacts, report_receipt)
    expected_measurement = sha256_json(
        {"run_id": report_meta.get("run_id"), "sha256": report_receipt.digest}
    )
    if expected_measurement != observation.measurement_receipt:
        reject("Terminal measurement identity differs from raw report")
    throughput = _mapping(report.get("throughput"), "throughput")
    latency = _mapping(report.get("latency"), "latency")
    total = throughput.get("total_token_throughput")
    selected = total if total is not None else throughput.get("output_throughput")
    if (
        _finite(selected) != observation.throughput
        or _integer(throughput.get("completed_requests")) != observation.completed_requests
        or _p99(latency, "ttft") != observation.ttft_p99_ms
        or _p99(latency, "tpot") != observation.tpot_p99_ms
    ):
        reject("Terminal paired metrics differ from raw benchmark report")
    matching_quality = tuple(
        (meta, receipt)
        for meta, receipt in quality
        if sha256_json({"run_id": meta.get("run_id"), "sha256": receipt.digest})
        == observation.quality_receipt
    )
    if len(matching_quality) != 1:
        reject("Terminal quality identity differs from raw evidence")
    accuracy = _primary_accuracy(_json(artifacts, matching_quality[0][1]))
    if accuracy != observation.accuracy:
        reject("Terminal accuracy differs from raw quality evidence")


def _raw_receipts(event: EpisodeEvent) -> tuple[ArtifactReceipt, ...]:
    values: dict[int, ArtifactReceipt] = {}
    for artifact in event.artifacts:
        if not artifact.role.startswith("terminal_raw_"):
            continue
        suffix = artifact.role.removeprefix("terminal_raw_")
        try:
            index = int(suffix)
        except ValueError:
            continue
        if index in values:
            reject("Terminal raw artifact index is duplicated")
        values[index] = artifact.receipt
    if set(values) != set(range(len(values))):
        reject("Terminal raw artifact indices are incomplete")
    return tuple(values[index] for index in range(len(values)))


def _metadata_matches(value: Mapping[str, Any], receipt: ArtifactReceipt) -> bool:
    return bool(
        value.get("role")
        in {
            "benchmark_report",
            "quality_result",
            "quality_sample",
            "quality_raw_artifact",
        }
        and value.get("sha256") == receipt.digest
        and value.get("size_bytes") == receipt.size
        and value.get("media_type") == receipt.media_type
        and isinstance(value.get("run_id"), str)
        and isinstance(value.get("measurement_receipt"), str)
        and isinstance(value.get("quality_receipt"), str)
    )


def _primary_accuracy(value: Mapping[str, Any]) -> float:
    results = value.get("results")
    if isinstance(results, Mapping):
        metrics = []
        for task in sorted(results):
            values = results[task]
            if not isinstance(values, Mapping):
                continue
            for name, raw in values.items():
                if isinstance(name, str) and "stderr" not in name and _number(raw):
                    metrics.append((name, float(raw)))
        for preferred in (
            "exact_match,strict-match",
            "acc_norm,none",
            "acc,none",
        ):
            match = next((raw for name, raw in metrics if name == preferred), None)
            if match is not None:
                return match
        if metrics:
            return metrics[0][1]
    gate = value.get("quality_gate")
    if isinstance(gate, Mapping):
        values = tuple(
            float(raw)
            for name, raw in sorted(gate.items())
            if name not in {"passed", "skipped"} and _number(raw)
        )
        if values:
            return values[0]
    reject("Raw terminal quality has no primary metric")


def _json(artifacts: ArtifactStore, receipt: ArtifactReceipt) -> Mapping[str, Any]:
    try:
        value = json.loads(artifacts.read_bytes(receipt))
    except json.JSONDecodeError as error:
        raise IntegrityError(
            "Terminal raw evidence is not JSON",
            "e2e_measurement_evidence_mismatch",
        ) from error
    return _mapping(value, "raw evidence")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        reject(f"Terminal {label} is not an object")
    return value


def _p99(latency: Mapping[str, Any], name: str) -> float:
    return _finite(_mapping(latency.get(name), name).get("p99_ms"))


def _finite(value: object) -> float:
    if not _number(value):
        reject("Terminal raw metric is not finite")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        reject("Terminal raw completed request count is invalid")
    return value


def _number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


__all__ = ["validate_terminal_raw_evidence"]
