"""Evaluator-owned quality evidence normalization and binding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import sha256_json

from .evaluator_execution import (
    LmEvalExecutionReceipt,
    validate_execution_binding,
)
from .quality_declared import (
    DeclaredQualityArtifacts,
    load_declared_quality_outputs,
)
from .quality_discovery import artifact_receipt, discover_quality_artifacts


PRIMARY_METRICS = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "exact_match",
    "acc_norm,none",
    "acc,none",
    "acc_norm",
    "acc",
    "pass@1,none",
    "pass@1",
)


@dataclass(frozen=True, slots=True)
class QualityMetric:
    task: str
    name: str
    value: float
    higher_is_better: bool


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    required: bool
    kind: str
    passed: bool
    metrics: tuple[QualityMetric, ...]
    source_paths: tuple[Path, ...]
    error: str | None = None
    primary_metrics: tuple[QualityMetric, ...] = ()
    raw_artifact_paths: tuple[Path, ...] = ()
    outcome_digest: str | None = None
    sample_set_digest: str | None = None
    hard_failure: bool = False


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _higher_is_better(name: str) -> bool:
    lowered = name.lower()
    return not any(
        marker in lowered
        for marker in ("loss", "perplexity", "ppl", "error", "wer", "cer")
    )


def _select_primary(metrics: tuple[QualityMetric, ...]) -> QualityMetric | None:
    by_name = {item.name: item for item in metrics if item.higher_is_better}
    return next((by_name[name] for name in PRIMARY_METRICS if name in by_name), None)


def _parse_metrics(data: Mapping[str, Any]) -> tuple[QualityMetric, ...]:
    metrics: list[QualityMetric] = []
    results = data.get("results")
    if not isinstance(results, Mapping):
        return ()
    for task, task_values in sorted(results.items()):
        if not isinstance(task, str) or not isinstance(task_values, Mapping):
            continue
        for name, raw_value in sorted(task_values.items()):
            if not isinstance(name, str):
                continue
            metric_name = name.split(",", 1)[0]
            if "stderr" in metric_name.lower():
                continue
            value = _finite_number(raw_value)
            if value is not None:
                metrics.append(
                    QualityMetric(
                        task=task,
                        name=name,
                        value=value,
                        higher_is_better=_higher_is_better(metric_name),
                    )
                )
    return tuple(metrics)


def _binding_error(
    *,
    gate: Mapping[str, Any] | None,
    expected_policy: Mapping[str, Any] | None,
    primary: tuple[QualityMetric, ...],
    primary_outcomes: Mapping[str, Any],
    result_receipts: tuple[Mapping[str, Any], ...],
    sample_receipts: tuple[Mapping[str, Any], ...],
    outcome_digest: str,
    sample_set_digest: str | None,
    expected_runtime_sha256: str | None,
    expected_image_repo_digest: str | None,
) -> str | None:
    if expected_policy is None:
        return None
    if not isinstance(gate, Mapping):
        return "quality_receipt_missing"
    envelope_error = _gate_envelope_error(gate)
    if envelope_error:
        return envelope_error
    expected_name = expected_policy.get("primary_metric")
    raw_tasks = expected_policy.get("tasks", ())
    expected_tasks = (
        tuple(item.strip() for item in raw_tasks.split(",") if item.strip())
        if isinstance(raw_tasks, str)
        else tuple(raw_tasks)
    )
    if not primary or any(item.name != expected_name for item in primary):
        return "quality_primary_metric_mismatch"
    if expected_tasks and tuple(item.task for item in primary) != expected_tasks:
        return "quality_task_set_mismatch"
    artifact_error = _artifact_binding_error(
        gate, primary, result_receipts, sample_receipts
    )
    if artifact_error:
        return artifact_error
    execution_error = validate_execution_binding(
        gate.get("evaluator_execution_receipt"),
        expected_policy=expected_policy,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_image_repo_digest=expected_image_repo_digest,
        result_artifacts=result_receipts,
        sample_artifacts=sample_receipts,
    )
    if execution_error:
        return execution_error
    checks = (
        ("primary_outcomes", primary_outcomes, "quality_primary_outcome_mismatch"),
        ("result_artifact_receipts", list(result_receipts), "quality_result_receipt_mismatch"),
        ("sample_artifact_receipts", list(sample_receipts), "quality_sample_receipt_mismatch"),
        ("outcome_digest", outcome_digest, "quality_outcome_digest_mismatch"),
        ("sample_set_digest", sample_set_digest, "quality_sample_set_digest_mismatch"),
    )
    for name, expected, error in checks:
        if gate.get(name) != expected:
            return error
    if sample_set_digest is None:
        return "quality_samples_missing"
    return None


def _gate_envelope_error(gate: Mapping[str, Any]) -> str | None:
    if (
        gate.get("requested") is not True
        or gate.get("status") != "passed"
        or gate.get("passed") is not True
        or gate.get("evidence_present") is not True
    ):
        return "quality_gate_not_passed"
    if gate.get("primary_metric_policy") != list(PRIMARY_METRICS):
        return "quality_primary_metric_policy_mismatch"
    if (
        gate.get("errors") != []
        or gate.get("error_count") != 0
        or gate.get("errors_truncated") is not False
        or gate.get("tasks_truncated") is not False
        or gate.get("result_artifacts_truncated") is not False
    ):
        return "quality_gate_incomplete"
    return None


def _artifact_binding_error(
    gate: Mapping[str, Any],
    primary: tuple[QualityMetric, ...],
    result_receipts: tuple[Mapping[str, Any], ...],
    sample_receipts: tuple[Mapping[str, Any], ...],
) -> str | None:
    if gate.get("task_count") != len(primary):
        return "quality_task_count_mismatch"
    if gate.get("result_artifact_count") != len(result_receipts):
        return "quality_result_count_mismatch"
    if any(receipt.get("size_bytes", 0) <= 0 for receipt in result_receipts):
        return "quality_result_artifact_empty"
    if any(receipt.get("size_bytes", 0) <= 0 for receipt in sample_receipts):
        return "quality_sample_artifact_empty"
    return None


def _lm_eval_binding_data(
    workspace: Path,
    sources: tuple[Path, ...],
    artifacts: tuple[Path, ...],
    metrics: tuple[QualityMetric, ...],
    declared: DeclaredQualityArtifacts | None = None,
) -> dict[str, Any]:
    primary = tuple(
        selected
        for task in sorted({item.task for item in metrics})
        if (selected := _select_primary(tuple(item for item in metrics if item.task == task)))
        is not None
    )
    results = (
        declared.result_receipts
        if declared is not None
        else tuple(artifact_receipt(path, workspace) for path in sources)
    )
    samples = (
        declared.sample_paths
        if declared is not None
        else tuple(
            path
            for path in artifacts
            if path.name.startswith("samples") and path.suffix == ".jsonl"
        )
    )
    sample_receipts = (
        declared.sample_receipts
        if declared is not None
        else tuple(artifact_receipt(path, workspace) for path in samples)
    )
    sample_digest = (
        sha256_json(
            {"schema": "magpie.lm-eval-sample-set/v1", "artifacts": sample_receipts}
        )
        if samples
        else None
    )
    outcomes = {
        item.task: {
            "metric": item.name,
            "value": item.value,
            "source": str(sources[0].relative_to(workspace.resolve())),
        }
        for item in primary
    }
    outcome_digest = sha256_json(
        {
            "schema": "magpie.lm-eval-outcomes/v1",
            "primary_metric_policy": list(PRIMARY_METRICS),
            "outcomes": outcomes,
            "result_artifacts": results,
            "sample_set_digest": sample_digest,
        }
    )
    return {
        "primary": primary,
        "outcomes": outcomes,
        "results": results,
        "samples": sample_receipts,
        "sample_digest": sample_digest,
        "outcome_digest": outcome_digest,
    }


def _parse_lm_eval(
    workspace: Path,
    *,
    required: bool,
    gate: Mapping[str, Any] | None,
    expected_policy: Mapping[str, Any] | None,
    expected_runtime_sha256: str | None,
    expected_image_repo_digest: str | None,
) -> QualityEvidence:
    declared, declaration_error = _declared_outputs(workspace, gate, expected_policy)
    if declaration_error:
        return QualityEvidence(
            required, "lm_eval", False, (), (), declaration_error
        )
    artifacts = (
        declared.all_paths
        if declared is not None
        else discover_quality_artifacts(workspace)
    )
    sources = (
        declared.result_paths
        if declared is not None
        else tuple(
            path for path in artifacts
            if path.name.startswith("results") and path.suffix == ".json"
        )
    )
    if not sources:
        return QualityEvidence(
            required,
            "lm_eval",
            not required,
            (),
            (),
            "quality_evidence_missing" if required else None,
            raw_artifact_paths=artifacts,
        )
    if len(sources) != 1:
        return QualityEvidence(
            required,
            "lm_eval",
            False,
            (),
            sources,
            "ambiguous_quality_evidence",
            raw_artifact_paths=artifacts,
        )
    data = (
        declared.result_document
        if declared is not None
        else json.loads(sources[0].read_text(encoding="utf-8"))
    )
    metrics = _parse_metrics(data)
    binding = _lm_eval_binding_data(
        workspace, sources, artifacts, metrics, declared
    )
    error = _binding_error(
        gate=gate,
        expected_policy=expected_policy,
        primary=binding["primary"],
        primary_outcomes=binding["outcomes"],
        result_receipts=binding["results"],
        sample_receipts=binding["samples"],
        outcome_digest=binding["outcome_digest"],
        sample_set_digest=binding["sample_digest"],
        expected_runtime_sha256=expected_runtime_sha256,
        expected_image_repo_digest=expected_image_repo_digest,
    )
    return _bound_quality_evidence(
        required=required,
        sources=sources,
        artifacts=artifacts,
        metrics=metrics,
        binding=binding,
        error=error,
        gate=gate,
        expected_policy=expected_policy,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_image_repo_digest=expected_image_repo_digest,
    )


def _bound_quality_evidence(
    *,
    required: bool,
    sources: tuple[Path, ...],
    artifacts: tuple[Path, ...],
    metrics: tuple[QualityMetric, ...],
    binding: Mapping[str, Any],
    error: str | None,
    gate: Mapping[str, Any] | None,
    expected_policy: Mapping[str, Any] | None,
    expected_runtime_sha256: str | None,
    expected_image_repo_digest: str | None,
) -> QualityEvidence:
    hard_failure = _trusted_lm_eval_failure(
        gate=gate,
        expected_policy=expected_policy,
        primary=binding["primary"],
        primary_outcomes=binding["outcomes"],
        result_receipts=binding["results"],
        sample_receipts=binding["samples"],
        outcome_digest=binding["outcome_digest"],
        sample_set_digest=binding["sample_digest"],
        expected_runtime_sha256=expected_runtime_sha256,
        expected_image_repo_digest=expected_image_repo_digest,
    )
    return QualityEvidence(
        required=required,
        kind="lm_eval",
        passed=bool(metrics) and error is None,
        metrics=metrics,
        source_paths=sources,
        error=error or (None if metrics else "quality_metrics_missing"),
        primary_metrics=binding["primary"],
        raw_artifact_paths=artifacts,
        outcome_digest=binding["outcome_digest"],
        sample_set_digest=binding["sample_digest"],
        hard_failure=hard_failure,
    )


def _declared_outputs(
    workspace: Path,
    gate: Mapping[str, Any] | None,
    expected_policy: Mapping[str, Any] | None,
) -> tuple[DeclaredQualityArtifacts | None, str | None]:
    if expected_policy is None:
        return None, None
    value = gate.get("evaluator_execution_receipt") if isinstance(gate, Mapping) else None
    if not isinstance(value, Mapping):
        return None, "quality_evaluator_execution_receipt_missing"
    try:
        return load_declared_quality_outputs(workspace, value), None
    except ValueError:
        return None, "quality_evaluator_execution_receipt_invalid"


def _trusted_lm_eval_failure(
    *,
    gate: Mapping[str, Any] | None,
    expected_policy: Mapping[str, Any] | None,
    primary: tuple[QualityMetric, ...],
    primary_outcomes: Mapping[str, Any],
    result_receipts: tuple[Mapping[str, Any], ...],
    sample_receipts: tuple[Mapping[str, Any], ...],
    outcome_digest: str,
    sample_set_digest: str | None,
    expected_runtime_sha256: str | None,
    expected_image_repo_digest: str | None,
) -> bool:
    """Recognize an explicit failed verdict only after all evidence binds."""

    if (
        expected_policy is None
        or not isinstance(gate, Mapping)
        or gate.get("status") != "failed"
        or gate.get("passed") is not False
    ):
        return False
    replay_gate = {**dict(gate), "status": "passed", "passed": True}
    return _binding_error(
        gate=replay_gate,
        expected_policy=expected_policy,
        primary=primary,
        primary_outcomes=primary_outcomes,
        result_receipts=result_receipts,
        sample_receipts=sample_receipts,
        outcome_digest=outcome_digest,
        sample_set_digest=sample_set_digest,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_image_repo_digest=expected_image_repo_digest,
    ) is None


def _parse_quality_gate(
    gate: Mapping[str, Any], report_path: Path, *, required: bool
) -> QualityEvidence:
    metrics = tuple(
        QualityMetric(
            task="framework_quality_gate",
            name=name,
            value=value,
            higher_is_better=_higher_is_better(name),
        )
        for name, raw in sorted(gate.items())
        if name not in {"passed", "skipped"}
        and (value := _finite_number(raw)) is not None
    )
    passed = gate.get("passed") is True and gate.get("skipped") is not True
    hard_failure = (
        gate.get("passed") is False
        and gate.get("skipped") is not True
        and bool(metrics)
    )
    return QualityEvidence(
        required,
        "framework_quality_gate",
        passed,
        metrics,
        (report_path,),
        None if passed else "quality_gate_not_passed",
        hard_failure=hard_failure,
    )


def parse_quality_evidence(
    report: Mapping[str, Any],
    workspace: Path,
    report_path: Path,
    required: bool,
    expected_evaluator_policy: Mapping[str, Any] | None,
    expected_quality_kind: str | None,
    expected_runtime_sha256: str | None = None,
    expected_image_repo_digest: str | None = None,
) -> QualityEvidence:
    """Normalize the frozen quality lane and bind formal evidence."""

    gate = report.get("quality_gate")
    kind = expected_quality_kind or (
        "framework_quality_gate"
        if isinstance(gate, Mapping) and not discover_quality_artifacts(workspace)
        else "lm_eval"
    )
    if kind == "trace_only" and required is False:
        return QualityEvidence(False, "trace_only", True, (), ())
    if kind in {"lm_eval", "trace_only"}:
        return _parse_lm_eval(
            workspace,
            required=required,
            gate=gate if isinstance(gate, Mapping) else None,
            expected_policy=expected_evaluator_policy,
            expected_runtime_sha256=expected_runtime_sha256,
            expected_image_repo_digest=expected_image_repo_digest,
        )
    if kind == "framework_quality_gate" and isinstance(gate, Mapping):
        return _parse_quality_gate(gate, report_path, required=required)
    return QualityEvidence(
        required,
        str(kind),
        False,
        (),
        (),
        "quality_contract_kind_mismatch",
    )


def build_lm_eval_quality_gate(
    workspace: Path,
    *,
    execution_receipt: LmEvalExecutionReceipt | None,
) -> Mapping[str, Any] | None:
    """Build bindings only from outputs declared by an evaluator authority."""

    if execution_receipt is None:
        return None
    try:
        declared = load_declared_quality_outputs(
            workspace.resolve(), execution_receipt.to_dict()
        )
    except (OSError, ValueError):
        return None
    binding = _lm_eval_binding_data(
        workspace.resolve(),
        declared.result_paths,
        declared.all_paths,
        _parse_metrics(declared.result_document),
        declared,
    )
    if not binding["primary"] or not binding["samples"]:
        return None
    return {
        "requested": True,
        "status": "passed",
        "passed": True,
        "evidence_present": True,
        "evaluator_execution_receipt": execution_receipt.to_dict(),
        "primary_metric_policy": list(PRIMARY_METRICS),
        "primary_outcomes": binding["outcomes"],
        "result_artifact_receipts": list(binding["results"]),
        "sample_artifact_receipts": list(binding["samples"]),
        "outcome_digest": binding["outcome_digest"],
        "sample_set_digest": binding["sample_digest"],
        "task_count": len(binding["primary"]),
        "tasks_truncated": False,
        "result_artifact_count": len(binding["results"]),
        "result_artifacts_truncated": False,
        "errors": [],
        "error_count": 0,
        "errors_truncated": False,
    }


__all__ = [
    "QualityEvidence",
    "QualityMetric",
    "build_lm_eval_quality_gate",
    "parse_quality_evidence",
]
