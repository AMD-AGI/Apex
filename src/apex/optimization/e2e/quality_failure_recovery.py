"""CAS replay of a trusted E2E quality-gate failure."""

from __future__ import annotations

from typing import Any, Mapping

from apex.benchmark import validate_magpie_execution_attestation_document
from apex.core import ContractError, IntegrityError
from apex.evaluation import replay_e2e_reward
from apex.ports import BenchmarkPass
from apex.storage import ArtifactReceipt, EventRecord

from .recovery_artifacts import read_json_object
from .recovery_bindings import verify_benchmark_config, verify_benchmark_event
from .recovery import receipt_for_digest
from .run_record import E2ERunRecord
from .services import CandidateDeployment


def verify_quality_gate_reward(
    event: EventRecord,
    *,
    verdict: str,
    reason: str,
) -> bool:
    """Return whether a missing paired promotion is a valid quality stop."""

    if verdict != "revert" or reason != "quality_gate_failed":
        return False
    vector = event.payload.get("reward_vector")
    try:
        valid = (
            isinstance(vector, Mapping)
            and vector.get("performance_skipped") == "quality_gate"
            and replay_e2e_reward(vector) == 20.0
        )
    except ContractError as error:
        raise IntegrityError(
            "Quality-gate reward cannot be replayed",
            "e2e_decision_replay_mismatch",
        ) from error
    if not valid:
        raise IntegrityError(
            "Quality-gate reward replay drifted",
            "e2e_decision_replay_mismatch",
        )
    return True


def try_verify_quality_gate_decision(
    record: E2ERunRecord,
    events: tuple[EventRecord, ...],
    value: Mapping[str, Any],
    *,
    promotion: object | None,
    deployment_pair: tuple[CandidateDeployment, ArtifactReceipt] | None,
    attempt_id: str,
    candidate_id: str | None,
    opportunity_id: str,
) -> bool:
    """Validate the clean-cut decision shape when performance was skipped."""

    if value.get("performance_skipped") != "quality_gate":
        return False
    benchmark_digest = value.get("benchmark_receipt")
    if (
        promotion is not None
        or deployment_pair is None
        or candidate_id is None
        or not isinstance(benchmark_digest, str)
        or value.get("verdict") != "revert"
        or value.get("reason") != "quality_gate_failed"
    ):
        _reject("Quality-gate decision evidence is incomplete")
    verify_quality_gate_failure(
        record,
        events,
        normalized=receipt_for_digest(record, benchmark_digest),
        deployment=deployment_pair[0],
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        opportunity_id=opportunity_id,
    )
    return True


def verify_quality_gate_failure(
    record: E2ERunRecord,
    events: tuple[EventRecord, ...],
    *,
    normalized: ArtifactReceipt,
    deployment: CandidateDeployment,
    attempt_id: str,
    candidate_id: str,
    opportunity_id: str,
) -> None:
    """Rebuild explicit quality failure semantics from event-bound CAS objects."""

    event = _measurement_event(events, normalized)
    quality_receipt = _role(event, "quality_evidence")
    config_receipt = _role(event, "benchmark_config")
    report_receipt = _role(event, "benchmark_report")
    attestation_receipt = _role(event, "benchmark_execution_attestation")
    verify_benchmark_event(
        events,
        normalized=normalized,
        quality=quality_receipt,
        config=config_receipt,
    )
    verify_benchmark_config(record, config_receipt, deployment)
    result = read_json_object(record, normalized, label="quality-gate benchmark")
    quality = read_json_object(record, quality_receipt, label="quality evidence")
    report = read_json_object(record, report_receipt, label="benchmark report")
    attestation = read_json_object(
        record, attestation_receipt, label="benchmark execution attestation"
    )
    _validate_lineage(event, attempt_id, candidate_id, opportunity_id)
    _validate_result(result, quality, config_receipt, deployment)
    gate = _validate_execution_attestation(
        result,
        report,
        report_receipt,
        config_receipt,
        attestation,
    )
    _validate_quality(record, event, quality, gate)


def _measurement_event(
    events: tuple[EventRecord, ...], normalized: ArtifactReceipt
) -> EventRecord:
    matches = tuple(
        event
        for event in events
        if event.event_type == "measurement_result"
        and _has_role(event, "normalized_benchmark", normalized.digest)
    )
    if len(matches) != 1:
        _reject("Quality-gate benchmark event is missing or ambiguous")
    return matches[0]


def _validate_lineage(
    event: EventRecord,
    attempt_id: str,
    candidate_id: str,
    opportunity_id: str,
) -> None:
    expected = {
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "opportunity_id": opportunity_id,
        "pass_type": "measurement",
        "succeeded": False,
        "run_kind": "measurement",
        "reward_eligible": True,
    }
    if any(event.payload.get(key) != value for key, value in expected.items()):
        _reject("Quality-gate benchmark lineage differs")


def _validate_result(
    result: Mapping[str, Any],
    quality: Mapping[str, Any],
    config: ArtifactReceipt,
    deployment: CandidateDeployment,
) -> None:
    normalized_quality = _mapping(result.get("quality"), "normalized quality")
    serving = _mapping(result.get("serving_runtime"), "serving runtime")
    expected_image = deployment.deployed_image_id
    if (
        result.get("schema_version") != 1
        or result.get("pass_type") != "measurement"
        or result.get("succeeded") is not False
        or result.get("profiling_enabled") is not False
        or result.get("run_kind") != "measurement"
        or result.get("reward_eligible") is not True
        or result.get("errors") != ["quality_gate_not_passed"]
        or result.get("command_exit_code") != 0
        or result.get("timed_out") is not False
        or quality.get("schema") != "apex.e2e-quality-evidence/v1"
        or any(
            quality.get(key) != normalized_quality.get(key)
            for key in (
                "required",
                "kind",
                "passed",
                "hard_failure",
                "metrics",
                "primary_metrics",
                "error",
                "outcome_digest",
                "sample_set_digest",
            )
        )
        or serving.get("required") is not True
        or serving.get("passed") is not True
        or serving.get("process_succeeded") is not True
        or serving.get("input_config_sha256") != config.digest
        or serving.get("requested_image") != expected_image
        or serving.get("resolved_image_id") != expected_image
    ):
        _reject("Quality-gate normalized evidence is invalid")


def _validate_quality(
    record: E2ERunRecord,
    event: EventRecord,
    quality: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    if (
        quality.get("required") is not True
        or quality.get("passed") is not False
        or quality.get("hard_failure") is not True
        or quality.get("error") != "quality_gate_not_passed"
    ):
        _reject("Quality evidence is not an explicit hard failure")
    receipts = _quality_receipts(record, quality)
    kind = quality.get("kind")
    if kind == "lm_eval":
        valid = (
            gate.get("requested") is True
            and gate.get("status") == "failed"
            and gate.get("passed") is False
            and gate.get("evidence_present") is True
            and gate.get("errors") == []
            and _gate_receipts(gate.get("result_artifact_receipts"), receipts[0])
            and _gate_receipts(gate.get("sample_artifact_receipts"), receipts[1])
        )
    else:
        valid = (
            kind == "framework_quality_gate"
            and gate.get("passed") is False
            and gate.get("skipped") is not True
            and bool(quality.get("metrics"))
        )
    bound_roles = (
        ("benchmark_report",)
        if kind == "framework_quality_gate"
        else ("quality_result", "quality_sample", "quality_raw_artifact")
    )
    bound = {
        receipt.digest
        for role in bound_roles
        for receipt in _roles(event, role)
    }
    expected = {item.digest for group in receipts for item in group}
    if not valid or bound != expected:
        _reject("Raw quality failure receipts are incomplete")


def _quality_receipts(
    record: E2ERunRecord, quality: Mapping[str, Any]
) -> tuple[tuple[ArtifactReceipt, ...], tuple[ArtifactReceipt, ...]]:
    groups = []
    for name in ("result_receipts", "raw_artifact_receipts"):
        raw = quality.get(name)
        if not isinstance(raw, list):
            _reject("Quality receipt list is invalid")
        try:
            receipts = tuple(ArtifactReceipt.from_dict(dict(item)) for item in raw)
        except (ContractError, TypeError, ValueError) as error:
            raise IntegrityError(
                "Quality receipt list is invalid", "quality_gate_replay_mismatch"
            ) from error
        for receipt in receipts:
            record.artifacts.verify(receipt)
        groups.append(receipts)
    return groups[0], groups[1]


def _gate_receipts(raw: object, receipts: tuple[ArtifactReceipt, ...]) -> bool:
    if not isinstance(raw, list) or len(raw) != len(receipts):
        return False
    expected = sorted((item.digest, item.size) for item in receipts)
    observed = sorted(
        (item.get("sha256"), item.get("size_bytes"))
        for item in raw
        if isinstance(item, Mapping)
    )
    return len(observed) == len(raw) and observed == expected


def _validate_execution_attestation(
    result: Mapping[str, Any],
    report: Mapping[str, Any],
    report_receipt: ArtifactReceipt,
    config_receipt: ArtifactReceipt,
    attestation: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = validate_magpie_execution_attestation_document(
        attestation,
        report_sha256=report_receipt.digest,
        report_size_bytes=report_receipt.size,
        report=report,
        expected_config_sha256=config_receipt.digest,
        expected_run_id=str(result.get("run_id")),
        expected_pass_type=BenchmarkPass.MEASUREMENT,
        command_exit_code=result.get("command_exit_code"),
        timed_out=result.get("timed_out") is True,
    )
    nested = tuple(
        _mapping(value.get(name), f"{name} attestation")
        for name in (
            "process",
            "dependencies",
            "runtime",
            "gpu_engagement",
            "quality_gate",
        )
    )
    if (
        value.get("lane_verified") is not True
        or value.get("reward_eligible") is not True
        or value.get("profiling_enabled") is not False
        or value.get("errors") != []
        or any(item.get("verified") is not True for item in nested)
    ):
        _reject("Quality-gate execution attestation is incomplete")
    quality = _mapping(value.get("quality_gate"), "quality attestation")
    return _mapping(quality.get("receipt"), "quality gate")


def _role(event: EventRecord, role: str) -> ArtifactReceipt:
    receipts = _roles(event, role)
    if len(receipts) != 1:
        _reject(f"Quality-gate {role} binding is invalid")
    return receipts[0]


def _roles(event: EventRecord, role: str) -> tuple[ArtifactReceipt, ...]:
    artifacts = event.payload.get("artifacts")
    if not isinstance(artifacts, list):
        return ()
    return tuple(
        ArtifactReceipt.from_dict(dict(item["receipt"]))
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("role") == role
        and isinstance(item.get("receipt"), Mapping)
    )


def _has_role(event: EventRecord, role: str, digest: str) -> bool:
    return any(item.digest == digest for item in _roles(event, role))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _reject(f"E2E {label} is invalid")
    return value


def _reject(message: str) -> None:
    raise IntegrityError(message, "quality_gate_replay_mismatch")


__all__ = [
    "try_verify_quality_gate_decision",
    "verify_quality_gate_failure",
    "verify_quality_gate_reward",
]
