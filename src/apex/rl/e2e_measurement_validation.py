"""Offline replay of measured E2E acceptance from exact CAS evidence."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_json,
)
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EMeasurement,
    evaluate_current_anchor,
    grade_e2e_outcome,
)
from apex.intake import RegressionGates
from apex.storage import ArtifactReceipt, ArtifactStore

from .models import CandidateEpisode, EpisodeEvent, EpisodeGraph
from .e2e_quality_validation import metric_documents, validate_quality_evidence


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
@dataclass(frozen=True, slots=True)
class _BenchmarkBundle:
    event: EpisodeEvent
    config: ArtifactReceipt
    normalized_receipt: ArtifactReceipt
    quality_receipt: ArtifactReceipt
    normalized: Mapping[str, Any]
    quality: Mapping[str, Any]
    report: Mapping[str, Any]
    measurement: E2EMeasurement


@dataclass(frozen=True, slots=True)
class _DeliveryEvidence:
    receipt: ArtifactReceipt
    measurement_config: ArtifactReceipt
    image_id: str


def validate_measured_e2e_evidence(
    graph: EpisodeGraph,
    child: CandidateEpisode,
    artifacts: ArtifactStore,
    decision: Mapping[str, Any],
) -> None:
    """Recompute a KEEP/REVERT and reward without trusting journal metrics."""

    protocol_hash = _protocol_hash(graph)
    acceptance_policy = _acceptance_policy(graph, artifacts)
    candidate_event = _candidate_benchmark_event(child)
    candidate = _load_bundle(candidate_event, artifacts, protocol_hash)
    delivery = _load_delivery(child, artifacts)
    _validate_candidate_runtime(candidate, delivery)
    _validate_decision_receipts(child, decision, delivery, candidate)
    measured = _mapping(decision.get("measurement_verdict"), "measurement verdict")
    anchor = _find_anchor_measurement(
        graph,
        artifacts,
        protocol_hash,
        str(measured.get("anchor_measurement_id", "")),
    )
    verdict = evaluate_current_anchor(anchor, candidate.measurement, acceptance_policy)
    expected_verdict = "keep" if verdict.keep else "revert"
    grade = grade_e2e_outcome(
        verdict=expected_verdict,
        reason_code=verdict.reason_code,
        candidate_present=True,
        measurement_verdict=verdict,
    )
    if (
        child.verdict != expected_verdict
        or decision.get("verdict") != expected_verdict
        or decision.get("reason") != verdict.reason_code
        or measured != verdict.to_dict()
        or child.reward_vector != grade.to_dict()
        or child.scalar_reward != grade.scalar_reward
    ):
        _reject("Measured E2E decision or reward differs from CAS replay")


def _protocol_hash(graph: EpisodeGraph) -> str:
    events = tuple(
        event
        for event in graph.parent.events
        if event.event_type.replace(".", "_") == "e2e_initialized"
    )
    if len(events) != 1:
        _reject("Measured E2E episode has no unique protocol declaration")
    value = events[0].payload.get("measurement_protocol_hash")
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        _reject("Measured E2E protocol declaration is invalid")
    return value


def _acceptance_policy(
    graph: EpisodeGraph,
    artifacts: ArtifactStore,
) -> E2EAcceptancePolicy:
    events = tuple(
        event
        for event in graph.parent.events
        if event.event_type.replace(".", "_") == "dependency_verified"
        and event.payload.get("kind") == "resolved_e2e_run_request"
        and _event_has_role(event, "run_request")
    )
    initialized = tuple(
        event
        for event in graph.parent.events
        if event.event_type.replace(".", "_") == "e2e_initialized"
    )
    if len(events) != 1 or len(initialized) != 1:
        _reject("Measured E2E episode has no unique frozen run request")
    receipt = _single_event_receipt(events[0], "run_request")
    request = _read_json(artifacts, receipt, canonical=True)
    spec = _mapping(request.get("spec"), "run request spec")
    goal = _mapping(spec.get("goal"), "optimization goal")
    gates = _mapping(goal.get("gates"), "regression gates")
    if (
        request.get("schema") != "apex.e2e-run-request/v1"
        or request.get("run_id") != graph.run_id
        or goal.get("primary") != "throughput"
        or goal.get("direction") != "maximize"
        or set(gates)
        != {
            "accuracy_regression_pct",
            "ttft_p99_regression_pct",
            "tpot_p99_regression_pct",
        }
        or initialized[0].payload.get("objective_policy_hash") != sha256_json(goal)
    ):
        _reject("Frozen E2E objective policy does not match initialization")
    try:
        return E2EAcceptancePolicy(
            RegressionGates(
                accuracy_regression_pct=_finite(
                    gates.get("accuracy_regression_pct")
                ),
                ttft_p99_regression_pct=_finite(
                    gates.get("ttft_p99_regression_pct")
                ),
                tpot_p99_regression_pct=_finite(
                    gates.get("tpot_p99_regression_pct")
                ),
            )
        )
    except ContractError as error:
        raise IntegrityError(
            "Frozen E2E acceptance policy is invalid",
            "e2e_measurement_evidence_mismatch",
        ) from error


def _candidate_benchmark_event(child: CandidateEpisode) -> EpisodeEvent:
    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "measurement_result"
        and _event_has_role(event, "normalized_benchmark")
    )
    if len(events) != 1:
        _reject("E2E candidate has no unique normalized scoring benchmark")
    event = events[0]
    if (
        event.payload.get("attempt_id") != child.attempt_id
        or event.payload.get("candidate_id") != child.candidate_id
        or event.payload.get("opportunity_id") != child.opportunity_id
    ):
        _reject("Scoring benchmark targets another E2E opportunity")
    return event


def _load_bundle(
    event: EpisodeEvent,
    artifacts: ArtifactStore,
    protocol_hash: str,
) -> _BenchmarkBundle:
    config = _single_event_receipt(event, "benchmark_config")
    normalized_receipt = _single_event_receipt(event, "normalized_benchmark")
    quality_receipt = _single_event_receipt(event, "quality_evidence")
    report_receipt = _single_event_receipt(event, "benchmark_report")
    normalized = _read_json(artifacts, normalized_receipt, canonical=True)
    quality = _read_json(artifacts, quality_receipt, canonical=True)
    report = _read_json(artifacts, report_receipt, canonical=False)
    _validate_event_receipts(event, config, normalized_receipt, quality_receipt)
    _validate_normal_lane(event, normalized)
    accuracy = validate_quality_evidence(event, artifacts, normalized, quality)
    _validate_report(report, normalized)
    serving = _mapping(normalized.get("serving_runtime"), "serving runtime")
    if serving.get("input_config_sha256") != config.digest:
        _reject("Benchmark config differs from the executed scoring runtime")
    measurement = _measurement(
        normalized,
        accuracy,
        protocol_hash,
        quality_receipt.digest,
        normalized_receipt.digest,
    )
    return _BenchmarkBundle(
        event,
        config,
        normalized_receipt,
        quality_receipt,
        normalized,
        quality,
        report,
        measurement,
    )


def _validate_event_receipts(
    event: EpisodeEvent,
    config: ArtifactReceipt,
    normalized: ArtifactReceipt,
    quality: ArtifactReceipt,
) -> None:
    payload = event.payload
    expected = {
        "config_sha256": config.digest,
        "normalized_benchmark_receipt": normalized.digest,
        "quality_receipt": quality.digest,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        _reject("Benchmark event receipt fields differ from bound artifacts")


def _validate_normal_lane(
    event: EpisodeEvent,
    normalized: Mapping[str, Any],
) -> None:
    if (
        event.evidence_class.value != "measured"
        or normalized.get("schema_version") != 1
        or normalized.get("pass_type") != "measurement"
        or normalized.get("succeeded") is not True
        or normalized.get("profiling_enabled") is not False
        or normalized.get("run_kind") != "measurement"
        or normalized.get("reward_eligible") is not True
        or normalized.get("errors") != []
        or normalized.get("command_exit_code") != 0
        or normalized.get("timed_out") is not False
    ):
        _reject("Diagnostic or failed benchmark cannot provide E2E reward evidence")
    expected_metrics = _event_metrics(normalized)
    payload = event.payload
    checks = {
        "pass_type": "measurement",
        "succeeded": True,
        "run_kind": "measurement",
        "reward_eligible": True,
        "metrics": expected_metrics,
    }
    if any(payload.get(key) != value for key, value in checks.items()):
        _reject("Benchmark event summary differs from its normalized artifact")


def _event_metrics(normalized: Mapping[str, Any]) -> dict[str, object]:
    throughput = _mapping(normalized.get("throughput"), "throughput")
    latency = _mapping(normalized.get("latency"), "latency")
    quality = _mapping(normalized.get("quality"), "quality")
    values: dict[str, object] = {
        "request_throughput": throughput.get("request_per_second"),
        "output_throughput": throughput.get("output_tokens_per_second"),
        "total_token_throughput": throughput.get("total_tokens_per_second"),
        "completed_requests": throughput.get("completed_requests"),
        "duration_seconds": throughput.get("duration_seconds"),
        "quality_required": int(quality.get("required") is True),
        "quality_passed": int(quality.get("passed") is True),
    }
    for prefix in ("ttft", "tpot"):
        distribution = _mapping(latency.get(prefix), prefix)
        for label in ("mean", "median", "p99"):
            values[f"{prefix}_{label}_ms"] = distribution.get(f"{label}_ms")
    for metric in metric_documents(quality.get("metrics")):
        values[f"quality.{metric['task']}.{metric['name']}"] = metric["value"]
    return {key: value for key, value in values.items() if value is not None}


def _validate_report(
    report: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> None:
    if (
        report.get("success") is not True
        or report.get("errors") != []
        or report.get("run_kind") != "measurement"
        or report.get("reward_eligible") is not True
        or report.get("profiling_enabled") is not False
        or report.get("framework") != normalized.get("framework")
        or report.get("model") != normalized.get("model")
    ):
        _reject("Raw benchmark report does not prove a normal successful lane")
    raw_throughput = _mapping(report.get("throughput"), "raw throughput")
    throughput = _mapping(normalized.get("throughput"), "throughput")
    pairs = {
        "request_throughput": "request_per_second",
        "output_throughput": "output_tokens_per_second",
        "total_token_throughput": "total_tokens_per_second",
        "completed_requests": "completed_requests",
        "duration_seconds": "duration_seconds",
    }
    if any(raw_throughput.get(raw) != throughput.get(norm) for raw, norm in pairs.items()):
        _reject("Normalized throughput differs from the raw benchmark report")
    raw_latency = _mapping(report.get("latency"), "raw latency")
    latency = _mapping(normalized.get("latency"), "latency")
    for name in ("ttft", "tpot"):
        if _mapping(raw_latency.get(name), name).get("p99_ms") != _mapping(
            latency.get(name), name
        ).get("p99_ms"):
            _reject("Normalized tail latency differs from the raw benchmark report")
    _validate_serving_runtime(report, normalized)


def _validate_serving_runtime(
    report: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> None:
    serving = _mapping(normalized.get("serving_runtime"), "serving runtime")
    if serving.get("required") is not True:
        return
    receipt = _mapping(report.get("serving_runtime_receipt"), "runtime receipt")
    expected = {
        "input_config_sha256": serving.get("input_config_sha256"),
        "requested_image": serving.get("requested_image"),
        "resolved_image_id": serving.get("resolved_image_id"),
        "container_name": serving.get("container_name"),
        "docker_argv_sha256": serving.get("docker_argv_sha256"),
        "process_succeeded": serving.get("process_succeeded"),
    }
    if (
        serving.get("passed") is not True
        or serving.get("error") is not None
        or receipt.get("schema") != "magpie.serving-runtime-receipt/v1"
        or receipt.get("execution_mode") != "docker"
        or receipt.get("verified") is not True
        or receipt.get("errors") != []
        or any(receipt.get(key) != value for key, value in expected.items())
    ):
        _reject("Normalized serving runtime differs from the raw runtime receipt")


def _measurement(
    normalized: Mapping[str, Any],
    accuracy: float,
    protocol_hash: str,
    quality_receipt: str,
    measurement_receipt: str,
) -> E2EMeasurement:
    throughput = _mapping(normalized.get("throughput"), "throughput")
    latency = _mapping(normalized.get("latency"), "latency")
    total = throughput.get("total_tokens_per_second")
    selected = total if total is not None else throughput.get("output_tokens_per_second")
    try:
        return E2EMeasurement(
            throughput=_finite(selected),
            ttft_p99_ms=_finite(_mapping(latency.get("ttft"), "ttft").get("p99_ms")),
            tpot_p99_ms=_finite(_mapping(latency.get("tpot"), "tpot").get("p99_ms")),
            accuracy=accuracy,
            completed_requests=_positive_int(throughput.get("completed_requests")),
            protocol_hash=protocol_hash,
            quality_receipt=quality_receipt,
            measurement_receipt=measurement_receipt,
        )
    except (ContractError, KeyError) as error:
        raise IntegrityError(
            "Normalized benchmark cannot reconstruct an E2E measurement",
            "e2e_measurement_evidence_mismatch",
        ) from error


def _load_delivery(
    child: CandidateEpisode,
    artifacts: ArtifactStore,
) -> _DeliveryEvidence:
    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "delivery_result"
        and _event_has_role(event, "primary_delivery")
    )
    if len(events) != 1:
        _reject("Measured E2E attempt has no unique primary delivery")
    event = events[0]
    delivery_receipt = _single_event_receipt(event, "primary_delivery")
    document = _read_json(artifacts, delivery_receipt, canonical=True)
    digests = _digest_mapping(document.get("config_sha256"))
    event_digests = _digest_mapping(event.payload.get("config_sha256"))
    configs = {
        "measurement": _single_event_receipt(event, "delivery_measurement_config"),
        "diagnostic": _single_event_receipt(event, "delivery_diagnostic_config"),
        "replay": _single_event_receipt(event, "delivery_replay_config"),
    }
    if digests != event_digests or any(
        configs[key].digest != digest for key, digest in digests.items()
    ):
        _reject("Delivery config receipt A differs from bound config artifact B")
    evidence = _mapping(document.get("evidence"), "delivery evidence")
    derived = _mapping(evidence.get("derived_image"), "derived image")
    image_id = document.get("deployed_image_id")
    if (
        document.get("candidate_id") != child.candidate_id
        or event.payload.get("candidate_id") != child.candidate_id
        or document.get("deployed") is not True
        or document.get("engagement_verified") is not True
        or document.get("infrastructure_failure") is not False
        or evidence.get("config_sha256") != digests
        or not isinstance(image_id, str)
        or not _IMAGE_ID.fullmatch(image_id)
        or derived.get("image_id") != image_id
    ):
        _reject("Primary delivery does not prove one engaged immutable image")
    return _DeliveryEvidence(delivery_receipt, configs["measurement"], image_id)


def _validate_candidate_runtime(
    benchmark: _BenchmarkBundle,
    delivery: _DeliveryEvidence,
) -> None:
    serving = _mapping(benchmark.normalized.get("serving_runtime"), "serving runtime")
    if (
        benchmark.config.digest != delivery.measurement_config.digest
        or serving.get("input_config_sha256") != benchmark.config.digest
        or serving.get("requested_image") != delivery.image_id
        or serving.get("resolved_image_id") != delivery.image_id
    ):
        _reject("Delivery image/config differs from the executed scoring runtime")


def _validate_decision_receipts(
    child: CandidateEpisode,
    decision: Mapping[str, Any],
    delivery: _DeliveryEvidence,
    benchmark: _BenchmarkBundle,
) -> None:
    expected = {
        "micro_receipt": _single_child_receipt(child, "micro_qualification").digest,
        "safety_receipt": _single_child_receipt(child, "safety_qualification").digest,
        "delivery_receipt": delivery.receipt.digest,
        "benchmark_receipt": benchmark.normalized_receipt.digest,
    }
    if any(decision.get(key) != digest for key, digest in expected.items()):
        _reject("Decision receipt A differs from bound evidence artifact B")


def _find_anchor_measurement(
    graph: EpisodeGraph,
    artifacts: ArtifactStore,
    protocol_hash: str,
    measurement_id: str,
) -> E2EMeasurement:
    if not _DIGEST.fullmatch(measurement_id):
        _reject("Decision anchor measurement identity is invalid")
    events = (
        *graph.parent.events,
        *(event for item in graph.children for event in item.events),
    )
    matches: list[E2EMeasurement] = []
    for event in events:
        if (
            event.event_type.replace(".", "_") != "measurement_result"
            or event.payload.get("pass_type") != "measurement"
            or not _event_has_role(event, "normalized_benchmark")
        ):
            continue
        measurement = _load_bundle(event, artifacts, protocol_hash).measurement
        if measurement.digest == measurement_id:
            matches.append(measurement)
    if len(matches) != 1:
        _reject("Decision anchor cannot be uniquely rebuilt from canonical evidence")
    return matches[0]


def _single_event_receipt(event: EpisodeEvent, role: str) -> ArtifactReceipt:
    found = tuple(item.receipt for item in event.artifacts if item.role == role)
    if len(found) != 1:
        _reject(f"Benchmark event requires exactly one {role} artifact")
    return found[0]


def _single_child_receipt(child: CandidateEpisode, role: str) -> ArtifactReceipt:
    unique = {
        item.receipt.digest: item.receipt
        for event in child.events
        for item in event.artifacts
        if item.role == role
    }
    if len(unique) != 1:
        _reject(f"E2E attempt requires exactly one {role} artifact")
    return next(iter(unique.values()))


def _read_json(
    artifacts: ArtifactStore,
    receipt: ArtifactReceipt,
    *,
    canonical: bool,
) -> Mapping[str, Any]:
    raw = artifacts.read_bytes(receipt)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise IntegrityError(
            "E2E evidence artifact is not JSON",
            "e2e_measurement_evidence_mismatch",
        ) from error
    if not isinstance(value, Mapping) or (canonical and canonical_json_bytes(value) != raw):
        _reject("E2E evidence artifact is not a canonical object")
    return value


def _digest_mapping(value: object) -> dict[str, str]:
    data = _mapping(value, "config digests")
    if set(data) != {"measurement", "diagnostic", "replay"} or any(
        not isinstance(item, str) or not _DIGEST.fullmatch(item)
        for item in data.values()
    ):
        _reject("Deployment config digest mapping is invalid")
    return {key: str(data[key]) for key in ("measurement", "diagnostic", "replay")}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _reject(f"E2E {label} is not an object")
    return value


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject("E2E measurement metric is not numeric")
    result = float(value)
    if not math.isfinite(result):
        _reject("E2E measurement metric is not finite")
    return result


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _reject("E2E completed request count is invalid")
    return value


def _event_has_role(event: EpisodeEvent, role: str) -> bool:
    return any(item.role == role for item in event.artifacts)


def _reject(message: str) -> None:
    raise IntegrityError(message, "e2e_measurement_evidence_mismatch")


__all__ = ["validate_measured_e2e_evidence"]
