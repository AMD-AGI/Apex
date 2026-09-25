"""Trusted readers for profiler-off E2E benchmark and delivery evidence."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from apex.benchmark import (
    parse_serving_runtime_evidence,
    validate_magpie_execution_attestation_document,
)
from apex.core import ContractError, IntegrityError, canonical_json_bytes
from apex.evaluation import E2EObservation
from apex.ports import BenchmarkPass
from apex.storage import ArtifactReceipt, ArtifactStore

from .e2e_quality_validation import (
    metric_documents,
    validate_quality_evidence,
    validate_quality_hard_failure,
)
from .models import CandidateEpisode, EpisodeEvent


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BenchmarkBundle:
    """One normal-lane benchmark rebuilt from its exact CAS artifacts."""

    event: EpisodeEvent
    config: ArtifactReceipt
    normalized_receipt: ArtifactReceipt
    quality_receipt: ArtifactReceipt
    normalized: Mapping[str, Any]
    quality: Mapping[str, Any]
    report: Mapping[str, Any]
    measurement: E2EObservation


@dataclass(frozen=True, slots=True)
class DeliveryEvidence:
    """The immutable candidate image and measurement config proven by delivery."""

    receipt: ArtifactReceipt
    measurement_config: ArtifactReceipt
    image_id: str


@dataclass(frozen=True, slots=True)
class QualityGateFailureBundle:
    """One normal-runtime candidate observation stopped by trusted quality."""

    event: EpisodeEvent
    config: ArtifactReceipt
    normalized_receipt: ArtifactReceipt
    quality_receipt: ArtifactReceipt
    normalized: Mapping[str, Any]
    quality: Mapping[str, Any]
    report: Mapping[str, Any]


def load_benchmark_bundle(
    event: EpisodeEvent,
    artifacts: ArtifactStore,
    protocol_hash: str,
) -> BenchmarkBundle:
    """Rebuild one reward-eligible measurement without trusting event summaries."""

    config = single_event_receipt(event, "benchmark_config")
    normalized_receipt = single_event_receipt(event, "normalized_benchmark")
    quality_receipt = single_event_receipt(event, "quality_evidence")
    report_receipt = single_event_receipt(event, "benchmark_report")
    normalized = read_json(artifacts, normalized_receipt, canonical=True)
    quality = read_json(artifacts, quality_receipt, canonical=True)
    report = read_json(artifacts, report_receipt, canonical=False)
    attestation = _load_execution_attestation(
        event,
        artifacts,
        normalized,
        report_receipt,
        report,
        config.digest,
    )
    _validate_event_receipts(event, config, normalized_receipt, quality_receipt)
    _validate_normal_lane(event, normalized)
    accuracy = validate_quality_evidence(event, artifacts, normalized, quality)
    _validate_report(report, normalized, attestation)
    serving = mapping(normalized.get("serving_runtime"), "serving runtime")
    if serving.get("input_config_sha256") != config.digest:
        reject("Benchmark config differs from the executed scoring runtime")
    measurement = _measurement(
        normalized,
        accuracy,
        protocol_hash,
        quality_receipt.digest,
        normalized_receipt.digest,
    )
    return BenchmarkBundle(
        event,
        config,
        normalized_receipt,
        quality_receipt,
        normalized,
        quality,
        report,
        measurement,
    )


def load_delivery(
    child: CandidateEpisode,
    artifacts: ArtifactStore,
) -> DeliveryEvidence:
    """Load the unique engaged candidate delivery bound to an E2E attempt."""

    events = tuple(
        event
        for event in child.events
        if event.event_type.replace(".", "_") == "delivery_result"
    )
    if len(events) != 1:
        reject("Measured E2E attempt has no unique primary delivery")
    event = events[0]
    delivery_receipt = single_event_receipt(event, "primary_delivery")
    document = read_json(artifacts, delivery_receipt, canonical=True)
    digests = _digest_mapping(document.get("config_sha256"))
    event_digests = _digest_mapping(event.payload.get("config_sha256"))
    configs = {
        "measurement": single_event_receipt(event, "delivery_measurement_config"),
        "diagnostic": single_event_receipt(event, "delivery_diagnostic_config"),
        "replay": single_event_receipt(event, "delivery_replay_config"),
    }
    if digests != event_digests or any(
        configs[key].digest != digest for key, digest in digests.items()
    ):
        reject("Delivery config receipt A differs from bound config artifact B")
    evidence = mapping(document.get("evidence"), "delivery evidence")
    derived = mapping(evidence.get("derived_image"), "derived image")
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
        reject("Primary delivery does not prove one engaged immutable image")
    return DeliveryEvidence(delivery_receipt, configs["measurement"], image_id)


def load_quality_gate_failure_bundle(
    event: EpisodeEvent,
    artifacts: ArtifactStore,
) -> QualityGateFailureBundle:
    """Rebuild a runtime-valid quality hard failure without performance metrics."""

    config = single_event_receipt(event, "benchmark_config")
    normalized_receipt = single_event_receipt(event, "normalized_benchmark")
    quality_receipt = single_event_receipt(event, "quality_evidence")
    report_receipt = single_event_receipt(event, "benchmark_report")
    normalized = read_json(artifacts, normalized_receipt, canonical=True)
    quality = read_json(artifacts, quality_receipt, canonical=True)
    report = read_json(artifacts, report_receipt, canonical=False)
    attestation = _load_execution_attestation(
        event,
        artifacts,
        normalized,
        report_receipt,
        report,
        config.digest,
    )
    _validate_event_receipts(event, config, normalized_receipt, quality_receipt)
    _validate_quality_failure_lane(event, normalized)
    validate_quality_hard_failure(
        event,
        artifacts,
        normalized,
        quality,
        _quality_gate_receipt(attestation),
    )
    _validate_report(report, normalized, attestation)
    serving = mapping(normalized.get("serving_runtime"), "serving runtime")
    if serving.get("input_config_sha256") != config.digest:
        reject("Quality failure config differs from the executed runtime")
    return QualityGateFailureBundle(
        event,
        config,
        normalized_receipt,
        quality_receipt,
        normalized,
        quality,
        report,
    )


def validate_candidate_runtime(
    benchmark: BenchmarkBundle,
    delivery: DeliveryEvidence,
) -> None:
    """Bind a candidate benchmark to the delivered config and immutable image."""

    serving = mapping(benchmark.normalized.get("serving_runtime"), "serving runtime")
    if (
        benchmark.config.digest != delivery.measurement_config.digest
        or serving.get("input_config_sha256") != benchmark.config.digest
        or serving.get("requested_image") != delivery.image_id
        or serving.get("resolved_image_id") != delivery.image_id
    ):
        reject("Delivery image/config differs from the executed scoring runtime")


def validate_failed_candidate_runtime(
    benchmark: QualityGateFailureBundle,
    delivery: DeliveryEvidence,
) -> None:
    """Bind a quality-stopped observation to the delivered image and config."""

    serving = mapping(benchmark.normalized.get("serving_runtime"), "serving runtime")
    if (
        benchmark.config.digest != delivery.measurement_config.digest
        or serving.get("required") is not True
        or serving.get("passed") is not True
        or serving.get("process_succeeded") is not True
        or serving.get("input_config_sha256") != benchmark.config.digest
        or serving.get("requested_image") != delivery.image_id
        or serving.get("resolved_image_id") != delivery.image_id
    ):
        reject("Quality failure does not prove the delivered runtime")


def single_event_receipt(event: EpisodeEvent, role: str) -> ArtifactReceipt:
    """Return one exact role binding from an event or fail closed."""

    found = tuple(item.receipt for item in event.artifacts if item.role == role)
    if len(found) != 1:
        reject(f"Benchmark event requires exactly one {role} artifact")
    return found[0]


def single_child_receipt(child: CandidateEpisode, role: str) -> ArtifactReceipt:
    """Return one unique digest for a role across an entire child episode."""

    unique = {
        item.receipt.digest: item.receipt
        for event in child.events
        for item in event.artifacts
        if item.role == role
    }
    if len(unique) != 1:
        reject(f"E2E attempt requires exactly one {role} artifact")
    return next(iter(unique.values()))


def read_json(
    artifacts: ArtifactStore,
    receipt: ArtifactReceipt,
    *,
    canonical: bool,
) -> Mapping[str, Any]:
    """Read a receipt-bound JSON object, optionally requiring canonical bytes."""

    raw = artifacts.read_bytes(receipt)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise IntegrityError(
            "E2E evidence artifact is not JSON",
            "e2e_measurement_evidence_mismatch",
        ) from error
    if not isinstance(value, Mapping) or (canonical and canonical_json_bytes(value) != raw):
        reject("E2E evidence artifact is not a canonical object")
    return value


def mapping(value: object, label: str) -> Mapping[str, Any]:
    """Narrow an evidence value to an object with the shared failure code."""

    if not isinstance(value, Mapping):
        reject(f"E2E {label} is not an object")
    return value


def event_has_role(event: EpisodeEvent, role: str) -> bool:
    """Whether an episode event contains an exact artifact role."""

    return any(item.role == role for item in event.artifacts)


def reject(message: str) -> None:
    """Raise the single fail-closed reason used by measured E2E replay."""

    raise IntegrityError(message, "e2e_measurement_evidence_mismatch")


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
        reject("Benchmark event receipt fields differ from bound artifacts")


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
        reject("Diagnostic or failed benchmark cannot provide E2E reward evidence")
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
        reject("Benchmark event summary differs from its normalized artifact")


def _validate_quality_failure_lane(
    event: EpisodeEvent,
    normalized: Mapping[str, Any],
) -> None:
    if (
        event.evidence_class.value != "measured"
        or normalized.get("schema_version") != 1
        or normalized.get("pass_type") != "measurement"
        or normalized.get("succeeded") is not False
        or normalized.get("profiling_enabled") is not False
        or normalized.get("run_kind") != "measurement"
        or normalized.get("reward_eligible") is not True
        or normalized.get("errors") != ["quality_gate_not_passed"]
        or normalized.get("command_exit_code") != 0
        or normalized.get("timed_out") is not False
    ):
        reject("Quality failure lane contains invalid or incomplete evidence")
    checks = {
        "pass_type": "measurement",
        "succeeded": False,
        "run_kind": "measurement",
        "reward_eligible": True,
        "metrics": _event_metrics(normalized),
    }
    if any(event.payload.get(key) != value for key, value in checks.items()):
        reject("Quality failure event differs from its normalized artifact")


def _event_metrics(normalized: Mapping[str, Any]) -> dict[str, object]:
    throughput = mapping(normalized.get("throughput"), "throughput")
    latency = mapping(normalized.get("latency"), "latency")
    quality = mapping(normalized.get("quality"), "quality")
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
        distribution = mapping(latency.get(prefix), prefix)
        for label in ("mean", "median", "p99"):
            values[f"{prefix}_{label}_ms"] = distribution.get(f"{label}_ms")
    for metric in metric_documents(quality.get("metrics")):
        values[f"quality.{metric['task']}.{metric['name']}"] = metric["value"]
    return {key: value for key, value in values.items() if value is not None}


def _validate_report(
    report: Mapping[str, Any],
    normalized: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> None:
    if (
        report.get("success") is not True
        or report.get("errors") != []
        or report.get("profiling_enabled") is not False
        or report.get("framework") != normalized.get("framework")
        or report.get("model") != normalized.get("model")
    ):
        reject("Raw benchmark report does not prove a normal successful lane")
    raw_throughput = mapping(report.get("throughput"), "raw throughput")
    throughput = mapping(normalized.get("throughput"), "throughput")
    pairs = {
        "request_throughput": "request_per_second",
        "output_throughput": "output_tokens_per_second",
        "total_token_throughput": "total_tokens_per_second",
        "completed_requests": "completed_requests",
        "duration_seconds": "duration_seconds",
    }
    if any(raw_throughput.get(raw) != throughput.get(norm) for raw, norm in pairs.items()):
        reject("Normalized throughput differs from the raw benchmark report")
    raw_latency = mapping(report.get("latency"), "raw latency")
    latency = mapping(normalized.get("latency"), "latency")
    for name in ("ttft", "tpot"):
        if mapping(raw_latency.get(name), name).get("p99_ms") != mapping(
            latency.get(name), name
        ).get("p99_ms"):
            reject("Normalized tail latency differs from the raw benchmark report")
    _validate_serving_runtime(attestation, normalized)


def _validate_serving_runtime(
    attestation: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> None:
    serving = mapping(normalized.get("serving_runtime"), "serving runtime")
    if serving.get("required") is not True:
        return
    runtime = mapping(attestation.get("runtime"), "runtime attestation")
    receipt = mapping(runtime.get("serving_runtime_receipt"), "runtime receipt")
    parsed = parse_serving_runtime_evidence(
        {
            "framework": normalized.get("framework"),
            "serving_runtime_receipt": receipt,
        },
        expected_config_sha256=serving.get("input_config_sha256"),
        expected_requested_image=serving.get("input_image"),
        expected_execution_mode="docker",
    )
    expected = {
        "input_config_sha256": serving.get("input_config_sha256"),
        "input_image": serving.get("input_image"),
        "input_image_id": serving.get("input_image_id"),
        "requested_image": serving.get("requested_image"),
        "resolved_image_id": serving.get("resolved_image_id"),
        "image_derivation": serving.get("image_derivation"),
        "container_name": serving.get("container_name"),
        "container_spec_sha256": serving.get("container_spec_sha256"),
        "process_succeeded": serving.get("process_succeeded"),
    }
    if (
        serving.get("passed") is not True
        or serving.get("error") is not None
        or asdict(parsed) != dict(serving)
        or receipt.get("schema") != "apex.magpie-serving-runtime-observation/v3"
        or receipt.get("execution_mode") != "docker"
        or receipt.get("verified") is not True
        or receipt.get("errors") != []
        or any(receipt.get(key) != value for key, value in expected.items())
    ):
        reject("Normalized serving runtime differs from the raw runtime receipt")


def _load_execution_attestation(
    event: EpisodeEvent,
    artifacts: ArtifactStore,
    normalized: Mapping[str, Any],
    report_receipt: ArtifactReceipt,
    report: Mapping[str, Any],
    config_sha256: str,
) -> Mapping[str, Any]:
    receipt = single_event_receipt(event, "benchmark_execution_attestation")
    document = read_json(artifacts, receipt, canonical=False)
    validated = validate_magpie_execution_attestation_document(
        document,
        report_sha256=report_receipt.digest,
        report_size_bytes=report_receipt.size,
        report=report,
        expected_config_sha256=config_sha256,
        expected_run_id=str(normalized.get("run_id")),
        expected_pass_type=BenchmarkPass.MEASUREMENT,
        command_exit_code=normalized.get("command_exit_code"),
        timed_out=normalized.get("timed_out") is True,
    )
    nested = (
        mapping(validated.get("process"), "process attestation"),
        mapping(validated.get("dependencies"), "dependency attestation"),
        mapping(validated.get("runtime"), "runtime attestation"),
        mapping(validated.get("gpu_engagement"), "GPU attestation"),
        mapping(validated.get("quality_gate"), "quality attestation"),
    )
    if (
        validated.get("lane_verified") is not True
        or validated.get("reward_eligible") is not True
        or validated.get("profiling_enabled") is not False
        or validated.get("errors") != []
        or any(item.get("verified") is not True for item in nested)
    ):
        reject("Execution attestation does not prove a reward-eligible lane")
    return validated


def _quality_gate_receipt(
    attestation: Mapping[str, Any],
) -> Mapping[str, Any]:
    quality = mapping(attestation.get("quality_gate"), "quality attestation")
    return mapping(quality.get("receipt"), "quality gate receipt")


def _measurement(
    normalized: Mapping[str, Any],
    accuracy: float,
    protocol_hash: str,
    quality_receipt: str,
    measurement_receipt: str,
) -> E2EObservation:
    throughput = mapping(normalized.get("throughput"), "throughput")
    latency = mapping(normalized.get("latency"), "latency")
    total = throughput.get("total_tokens_per_second")
    selected = total if total is not None else throughput.get("output_tokens_per_second")
    try:
        return E2EObservation(
            throughput=_finite(selected),
            ttft_p99_ms=_finite(mapping(latency.get("ttft"), "ttft").get("p99_ms")),
            tpot_p99_ms=_finite(mapping(latency.get("tpot"), "tpot").get("p99_ms")),
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


def _digest_mapping(value: object) -> dict[str, str]:
    data = mapping(value, "config digests")
    if set(data) != {"measurement", "diagnostic", "replay"} or any(
        not isinstance(item, str) or not _DIGEST.fullmatch(item)
        for item in data.values()
    ):
        reject("Deployment config digest mapping is invalid")
    return {key: str(data[key]) for key in ("measurement", "diagnostic", "replay")}


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reject("E2E measurement metric is not numeric")
    result = float(value)
    if not math.isfinite(result):
        reject("E2E measurement metric is not finite")
    return result


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        reject("E2E completed request count is invalid")
    return value


__all__ = [
    "BenchmarkBundle",
    "DeliveryEvidence",
    "QualityGateFailureBundle",
    "event_has_role",
    "load_benchmark_bundle",
    "load_delivery",
    "load_quality_gate_failure_bundle",
    "mapping",
    "read_json",
    "reject",
    "single_child_receipt",
    "single_event_receipt",
    "validate_candidate_runtime",
    "validate_failed_candidate_runtime",
]
