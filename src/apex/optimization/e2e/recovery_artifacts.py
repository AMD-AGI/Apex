"""Rebuild typed E2E values exclusively from journal-bound CAS documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from apex.core import (
    AgentBackendName,
    ContractError,
    IntegrityError,
    TaskStatus,
    ValidationLevel,
    sha256_file,
)
from apex.evaluation import (
    E2EObservation,
    GateVerdict,
    GradeAggregation,
    KernelGrade,
    MeasurementStatus,
    Quantiles,
)
from apex.evaluation.kernel import CaseGrade
from apex.ports import AgentResult
from apex.storage import ArtifactReceipt

from .candidate import E2ECandidate, FrozenCandidateSource, validate_frozen_sources
from .run_record import E2ERunRecord
from .services import (
    CandidateDeployment,
    DeploymentConfigDigests,
    MicroQualification,
    SafetyQualification,
)


def read_json_object(
    record: E2ERunRecord, receipt: ArtifactReceipt, *, label: str
) -> Mapping[str, Any]:
    """Verify a receipt in the run CAS, then decode one JSON object."""

    record.artifacts.verify(receipt)
    try:
        value = json.loads(record.artifacts.read_bytes(receipt))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            f"{label} CAS document is invalid", "invalid_recovery_artifact"
        ) from error
    if not isinstance(value, Mapping):
        raise IntegrityError(
            f"{label} CAS document must be an object", "invalid_recovery_artifact"
        )
    return value


def recover_candidate(
    record: E2ERunRecord,
    receipt: ArtifactReceipt,
    *,
    attempt_id: str,
    agent_identity: Mapping[str, Any] | None,
) -> E2ECandidate:
    """Rebuild frozen source bytes; agent prose is intentionally not restored."""

    value = read_json_object(record, receipt, label="candidate manifest")
    if value.get("schema_version") != 1 or value.get("attempt_id") != attempt_id:
        raise IntegrityError(
            "Candidate manifest lineage is invalid", "candidate_lineage_mismatch"
        )
    sources = _recover_frozen_sources(record, value)
    workspace = _safe_record_path(record, value.get("workspace"), require_file=False)
    candidate = E2ECandidate(
        attempt_id=attempt_id,
        candidate_id=_optional_text(value.get("candidate_id")),
        succeeded=_boolean(value.get("succeeded"), "candidate succeeded"),
        reason_code=_text(value.get("reason_code"), "candidate reason"),
        workspace=workspace,
        editable_files=_text_tuple(value.get("editable_files"), "editable files"),
        changed_files=_text_tuple(value.get("changed_files"), "changed files"),
        baseline_source_sha256=_sha256(
            value.get("baseline_source_sha256"), "baseline source"
        ),
        candidate_source_sha256=_optional_sha256(
            value.get("candidate_source_sha256"), "candidate source"
        ),
        agent_result=_recovered_agent_result(agent_identity),
        frozen_sources=sources,
    )
    if candidate.succeeded:
        validate_frozen_sources(candidate)
    elif candidate.candidate_id is not None or sources:
        raise IntegrityError(
            "Rejected candidate claims frozen source", "invalid_frozen_candidate"
        )
    return candidate


def recover_micro(
    record: E2ERunRecord, receipt: ArtifactReceipt, *, candidate_id: str
) -> MicroQualification:
    value = read_json_object(record, receipt, label="micro qualification")
    if value.get("candidate_id") != candidate_id:
        raise IntegrityError("Micro lineage differs", "candidate_id_mismatch")
    evidence = _mapping(value.get("evidence"), "micro evidence")
    mode = _text(value.get("qualification_mode"), "qualification mode")
    grade_value = value.get("grade")
    grade = _kernel_grade(_mapping(grade_value, "kernel grade")) if grade_value else None
    result = MicroQualification(
        candidate_id=candidate_id,
        grade=grade,
        evidence=dict(evidence),
        qualification_mode=mode,
        deferred_candidate_valid=_boolean(
            value.get("deferred_candidate_valid"), "deferred candidate validity"
        ),
    )
    if result.qualified is not (value.get("qualified") is True):
        raise IntegrityError("Micro verdict drifted", "invalid_micro_qualification")
    return result


def recover_safety(
    record: E2ERunRecord, receipt: ArtifactReceipt, *, candidate_id: str
) -> SafetyQualification:
    value = read_json_object(record, receipt, label="safety qualification")
    if value.get("candidate_id") != candidate_id:
        raise IntegrityError("Safety lineage differs", "candidate_id_mismatch")
    return SafetyQualification(
        candidate_id=candidate_id,
        allowed_to_measure=_boolean(
            value.get("allowed_to_measure"), "allowed_to_measure"
        ),
        promotion_eligible=_boolean(
            value.get("promotion_eligible"), "promotion_eligible"
        ),
        safety_certified=_boolean(
            value.get("safety_certified"), "safety_certified"
        ),
        finding=_boolean(value.get("finding"), "safety finding"),
        reason_codes=_text_tuple(value.get("reason_codes"), "safety reasons"),
        evidence=dict(_mapping(value.get("evidence"), "safety evidence")),
    )


def recover_deployment(
    record: E2ERunRecord, receipt: ArtifactReceipt, *, candidate_id: str
) -> CandidateDeployment:
    value = read_json_object(record, receipt, label="candidate deployment")
    if value.get("candidate_id") != candidate_id:
        raise IntegrityError("Deployment lineage differs", "candidate_id_mismatch")
    evidence = _mapping(value.get("evidence"), "deployment evidence")
    configs = tuple(
        _safe_record_path(record, value.get(name), require_file=True)
        for name in ("measurement_config", "diagnostic_config", "replay_config")
    )
    config_sha256 = _deployment_config_digests(value, configs)
    return CandidateDeployment(
        candidate_id=candidate_id,
        deployed=_boolean(value.get("deployed"), "deployment status"),
        reason_code=_text(value.get("reason_code"), "deployment reason"),
        measurement_config=configs[0],
        diagnostic_config=configs[1],
        replay_config=configs[2],
        workload_semantics_sha256=_sha256(
            value.get("workload_semantics_sha256"), "workload semantics"
        ),
        deployed_source_sha256=_sha256(
            value.get("deployed_source_sha256"), "deployed source"
        ),
        deployed_image_id=_optional_text(value.get("deployed_image_id")),
        validation_level=ValidationLevel(
            _text(value.get("validation_level"), "validation level")
        ),
        engagement_verified=_boolean(
            value.get("engagement_verified"), "engagement status"
        ),
        evidence=dict(evidence),
        infrastructure_failure=_boolean(
            value.get("infrastructure_failure", False), "infrastructure failure"
        ),
        config_sha256=config_sha256,
    )


def recover_measurement(
    record: E2ERunRecord,
    receipt: ArtifactReceipt,
    *,
    protocol_hash: str,
    quality_receipt: ArtifactReceipt,
) -> E2EObservation:
    value = read_json_object(record, receipt, label="benchmark measurement")
    if value.get("succeeded") is not True or value.get("pass_type") != "measurement":
        raise IntegrityError("Measurement did not succeed", "invalid_measurement_receipt")
    throughput = _mapping(value.get("throughput"), "throughput")
    latency = _mapping(value.get("latency"), "latency")
    quality = read_json_object(record, quality_receipt, label="quality evidence")
    if (
        quality.get("schema") != "apex.e2e-quality-evidence/v1"
        or quality.get("passed") is not True
    ):
        raise IntegrityError("Quality evidence is invalid", "invalid_measurement_receipt")
    normalized_quality = _mapping(value.get("quality"), "normalized quality")
    if normalized_quality.get("metrics") != quality.get("metrics"):
        raise IntegrityError("Quality projections differ", "invalid_measurement_receipt")
    _verify_quality_receipts(record, quality)
    total = throughput.get("total_tokens_per_second")
    selected = total if total is not None else throughput.get("output_tokens_per_second")
    metrics = quality.get("metrics")
    primary = _primary_quality(metrics)
    return E2EObservation(
        throughput=_float(selected, "throughput"),
        ttft_p99_ms=_latency_p99(latency, "ttft"),
        tpot_p99_ms=_latency_p99(latency, "tpot"),
        accuracy=primary,
        completed_requests=_integer(
            throughput.get("completed_requests"), "completed requests"
        ),
        protocol_hash=protocol_hash,
        quality_receipt=quality_receipt.digest,
        measurement_receipt=receipt.digest,
    )


def recover_measurement_result(
    record: E2ERunRecord,
    receipt: ArtifactReceipt,
    *,
    protocol_hash: str,
    quality_receipt: ArtifactReceipt,
) -> E2EObservation | None:
    """Recover a successful measurement, or preserve a proven failed result."""

    value = read_json_object(record, receipt, label="benchmark measurement")
    if value.get("pass_type") != "measurement":
        raise IntegrityError("Benchmark lane drifted", "invalid_measurement_receipt")
    if value.get("succeeded") is not True:
        return None
    return recover_measurement(
        record,
        receipt,
        protocol_hash=protocol_hash,
        quality_receipt=quality_receipt,
    )


def verify_candidate_runtime_document(
    record: E2ERunRecord,
    receipt: ArtifactReceipt,
    deployment: CandidateDeployment,
) -> None:
    """Recheck immutable serving identity without trusting a local report."""

    value = read_json_object(record, receipt, label="benchmark measurement")
    runtime = _mapping(value.get("serving_runtime"), "serving runtime")
    expected = deployment.deployed_image_id
    config = deployment.config_sha256
    if (
        expected is None
        or config is None
        or runtime.get("required") is not True
        or runtime.get("input_config_sha256") != config.measurement
        or runtime.get("requested_image") != expected
        or runtime.get("resolved_image_id") != expected
        or (
            value.get("succeeded") is True
            and (
                runtime.get("passed") is not True
                or runtime.get("process_succeeded") is not True
            )
        )
    ):
        raise IntegrityError(
            "Recovered benchmark used another serving image",
            "candidate_runtime_image_mismatch",
        )


def _recover_frozen_sources(
    record: E2ERunRecord, value: Mapping[str, Any]
) -> tuple[FrozenCandidateSource, ...]:
    metadata = value.get("frozen_sources")
    receipts = value.get("source_receipts")
    if not isinstance(metadata, list) or not isinstance(receipts, list):
        raise IntegrityError("Frozen source bindings are invalid", "invalid_frozen_candidate")
    if len(metadata) != len(receipts):
        raise IntegrityError("Frozen source bindings are partial", "invalid_frozen_candidate")
    recovered = []
    for raw_metadata, raw_receipt in zip(metadata, receipts, strict=True):
        item = _mapping(raw_metadata, "frozen source")
        receipt = ArtifactReceipt.from_dict(dict(_mapping(raw_receipt, "source receipt")))
        content = record.artifacts.read_bytes(receipt)
        if receipt.digest != item.get("sha256") or len(content) != item.get("size"):
            raise IntegrityError("Frozen source receipt drifted", "invalid_frozen_candidate")
        recovered.append(
            FrozenCandidateSource(
                _text(item.get("path"), "frozen source path"),
                receipt.digest,
                _integer(item.get("mode"), "frozen source mode"),
                content,
            )
        )
    return tuple(recovered)


def _verify_quality_receipts(
    record: E2ERunRecord, quality: Mapping[str, Any]
) -> None:
    for field in ("result_receipts", "raw_artifact_receipts"):
        raw = quality.get(field)
        if not isinstance(raw, list):
            raise IntegrityError("Quality receipt list is invalid", "invalid_measurement_receipt")
        for item in raw:
            receipt = ArtifactReceipt.from_dict(
                dict(_mapping(item, "quality artifact receipt"))
            )
            record.artifacts.verify(receipt)


def _recovered_agent_result(identity: Mapping[str, Any] | None) -> AgentResult:
    if identity is None:
        raise IntegrityError("Frozen candidate lacks agent identity", "agent_lineage_missing")
    try:
        backend = AgentBackendName(str(identity.get("backend", "")))
    except ValueError as error:
        raise IntegrityError("Agent backend is invalid", "agent_lineage_missing") from error
    model = identity.get("model")
    if model is not None and not isinstance(model, str):
        raise IntegrityError("Agent model is invalid", "agent_lineage_missing")
    # Transcripts are retained in CAS/events but never become executable recovery input.
    return AgentResult(backend, model, 0, False, (), "", "", 0.0)


def _kernel_grade(value: Mapping[str, Any]) -> KernelGrade:
    gates = _mapping(value.get("gates"), "kernel gates")
    cases = tuple(_case_grade(_mapping(item, "case grade")) for item in _list(value.get("cases")))
    return KernelGrade(
        policy_id=_text(value.get("policy_id"), "grade policy"),
        measurement_status=MeasurementStatus(
            _text(value.get("measurement_status"), "measurement status")
        ),
        task_status=TaskStatus(_text(value.get("task_status"), "task status")),
        gates=GateVerdict(
            _boolean(gates.get("compiled"), "compiled"),
            _boolean(gates.get("correct"), "correct"),
            _boolean(gates.get("integrity_passed"), "integrity"),
            _boolean(gates.get("tampering_passed"), "tampering"),
            _boolean(gates.get("safety_finding", False), "safety finding"),
        ),
        cases=cases,
        aggregation=GradeAggregation(_text(value.get("aggregation"), "aggregation")),
        s50=_optional_float(value.get("s50"), "s50"),
        s99=_optional_float(value.get("s99"), "s99"),
        srobust=_optional_float(value.get("srobust"), "srobust"),
        worst_case_srobust=_optional_float(value.get("worst_case_srobust"), "worst case"),
        reward=_optional_float(value.get("reward"), "reward"),
        max_cv=_optional_float(value.get("max_cv"), "max cv"),
        s50_ci_lower=_optional_float(value.get("s50_ci_lower"), "s50 ci lower"),
        s50_ci_upper=_optional_float(value.get("s50_ci_upper"), "s50 ci upper"),
        s99_ci_lower=_optional_float(value.get("s99_ci_lower"), "s99 ci lower"),
        s99_ci_upper=_optional_float(value.get("s99_ci_upper"), "s99 ci upper"),
        srobust_ci_lower=_optional_float(value.get("srobust_ci_lower"), "srobust ci lower"),
        srobust_ci_upper=_optional_float(value.get("srobust_ci_upper"), "srobust ci upper"),
        confidence_level=_float(value.get("confidence_level"), "confidence"),
        bootstrap_seed=_integer(value.get("bootstrap_seed"), "bootstrap seed"),
        bootstrap_repetitions=_integer(value.get("bootstrap_repetitions"), "bootstrap repetitions"),
        min_bootstrap_units=_integer(value.get("min_bootstrap_units"), "bootstrap units"),
        keep_srobust_threshold=_float(value.get("keep_srobust_threshold"), "keep threshold"),
        confidence_srobust_floor=_float(value.get("confidence_srobust_floor"), "confidence floor"),
        worst_case_srobust_floor=_float(value.get("worst_case_srobust_floor"), "worst floor"),
        max_cv_threshold=_float(value.get("max_cv_threshold"), "cv threshold"),
        threshold_pass=_boolean(value.get("threshold_pass"), "threshold pass"),
        confidence_pass=_boolean(value.get("confidence_pass"), "confidence pass"),
        noise_pass=_boolean(value.get("noise_pass"), "noise pass"),
        worst_case_pass=_boolean(value.get("worst_case_pass"), "worst-case pass"),
        promotion_eligible=_boolean(value.get("promotion_eligible"), "promotion"),
        promotion_reason_code=_text(value.get("promotion_reason_code"), "promotion reason"),
        reward_bounds=tuple(_float(item, "reward bound") for item in _list(value.get("reward_bounds"))),
        reason_code=_optional_text(value.get("reason_code")),
    )


def _case_grade(value: Mapping[str, Any]) -> CaseGrade:
    return CaseGrade(
        case_id=_text(value.get("case_id"), "case id"),
        reference=_quantiles(_mapping(value.get("reference"), "reference quantiles")),
        optimized=_quantiles(_mapping(value.get("optimized"), "optimized quantiles")),
        s50=_float(value.get("s50"), "case s50"),
        s99=_float(value.get("s99"), "case s99"),
        srobust=_float(value.get("srobust"), "case srobust"),
        workload_count=_integer(value.get("workload_count"), "workload count"),
        reference_cv=_float(value.get("reference_cv"), "reference cv"),
        optimized_cv=_float(value.get("optimized_cv"), "optimized cv"),
        s50_ci_lower=_optional_float(value.get("s50_ci_lower"), "case s50 lower"),
        s50_ci_upper=_optional_float(value.get("s50_ci_upper"), "case s50 upper"),
        s99_ci_lower=_optional_float(value.get("s99_ci_lower"), "case s99 lower"),
        s99_ci_upper=_optional_float(value.get("s99_ci_upper"), "case s99 upper"),
        srobust_ci_lower=_optional_float(value.get("srobust_ci_lower"), "case robust lower"),
        srobust_ci_upper=_optional_float(value.get("srobust_ci_upper"), "case robust upper"),
        bootstrap_unit_count=_integer(value.get("bootstrap_unit_count"), "bootstrap units"),
        bootstrap_repetitions=_integer(value.get("bootstrap_repetitions"), "bootstrap repetitions"),
    )


def _quantiles(value: Mapping[str, Any]) -> Quantiles:
    return Quantiles(
        _float(value.get("p50_ms"), "p50"),
        _float(value.get("p99_ms"), "p99"),
        _integer(value.get("sample_count"), "sample count"),
        _text(value.get("sample_unit"), "sample unit"),
        _text(value.get("quantile_method"), "quantile method"),
        _sha256(value.get("artifact_sha256"), "sample artifact"),
    )


def _primary_quality(value: Any) -> float:
    values = tuple(_mapping(item, "quality metric") for item in _list(value))
    preferred = (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "exact_match",
        "acc_norm,none",
        "acc,none",
        "acc_norm",
        "acc",
    )
    eligible = tuple(item for item in values if item.get("higher_is_better") is True)
    for name in preferred:
        match = next((item for item in eligible if item.get("name") == name), None)
        if match is not None:
            return _float(match.get("value"), "quality value")
    if eligible:
        return _float(eligible[0].get("value"), "quality value")
    raise IntegrityError("Primary quality is missing", "invalid_measurement_receipt")


def _deployment_config_digests(
    value: Mapping[str, Any], configs: tuple[Path, ...]
) -> DeploymentConfigDigests | None:
    raw = value.get("config_sha256")
    if value.get("deployed") is not True:
        if raw is not None:
            raise IntegrityError(
                "Failed deployment claims config identity",
                "invalid_deployment_identity",
            )
        return None
    mapping = _mapping(raw, "deployment config hashes")
    result = DeploymentConfigDigests(
        _sha256(mapping.get("measurement"), "measurement config"),
        _sha256(mapping.get("diagnostic"), "diagnostic config"),
        _sha256(mapping.get("replay"), "replay config"),
    )
    for path, expected in zip(configs, result.to_dict().values(), strict=True):
        if sha256_file(path) != expected:
            raise IntegrityError("Deployment config drifted", "deployment_config_drift")
    return result


def _safe_record_path(
    record: E2ERunRecord, value: Any, *, require_file: bool
) -> Path:
    path = Path(_text(value, "run path"))
    if not path.is_absolute() or path.is_symlink():
        raise IntegrityError("Recovered path is unsafe", "unsafe_recovery_path")
    try:
        resolved = path.resolve(strict=require_file)
        resolved.relative_to(record.root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise IntegrityError("Recovered path escapes run", "unsafe_recovery_path") from error
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        raise IntegrityError("Recovered file is unsafe", "unsafe_recovery_path")
    return resolved


def _latency_p99(latency: Mapping[str, Any], name: str) -> float:
    return _float(_mapping(latency.get(name), name).get("p99_ms"), f"{name} p99")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} is invalid", "invalid_recovery_artifact")
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise IntegrityError("Recovery list is invalid", "invalid_recovery_artifact")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{label} is invalid", "invalid_recovery_artifact")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return _text(value, "optional text")


def _text_tuple(value: Any, label: str) -> tuple[str, ...]:
    values = _list(value)
    return tuple(_text(item, label) for item in values)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise IntegrityError(f"{label} is invalid", "invalid_recovery_artifact")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegrityError(f"{label} is invalid", "invalid_recovery_artifact")
    return value


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrityError(f"{label} is invalid", "invalid_recovery_artifact")
    return float(value)


def _optional_float(value: Any, label: str) -> float | None:
    return None if value is None else _float(value, label)


def _sha256(value: Any, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise IntegrityError(f"{label} digest is invalid", "invalid_recovery_artifact")
    return text


def _optional_sha256(value: Any, label: str) -> str | None:
    return None if value is None else _sha256(value, label)


__all__ = [
    "read_json_object",
    "recover_candidate",
    "recover_deployment",
    "recover_measurement",
    "recover_measurement_result",
    "recover_micro",
    "recover_safety",
    "verify_candidate_runtime_document",
]
