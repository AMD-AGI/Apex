"""Evaluator-owned quality evidence normalization and binding."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError, sha256_file, sha256_json


SERVING_FRAMEWORKS = frozenset({"vllm", "sglang", "atom"})
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


def _quality_artifacts(workspace: Path) -> tuple[Path, ...]:
    quality_root = workspace.resolve() / "lm_eval"
    if quality_root.is_symlink() or not quality_root.is_dir():
        return ()
    files: list[Path] = []
    for path in sorted(quality_root.rglob("*")):
        resolved = path.resolve()
        try:
            resolved.relative_to(quality_root)
        except ValueError as error:
            raise IntegrityError(
                f"Quality artifact escapes benchmark workspace: {path}",
                "unsafe_quality_artifact",
            ) from error
        if path.is_symlink():
            raise IntegrityError(
                f"Quality artifact must be a regular file: {path}",
                "unsafe_quality_artifact",
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise IntegrityError(
                f"Quality artifact must be a regular file: {path}",
                "unsafe_quality_artifact",
            )
        if path.name.startswith("results") and path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("results"), dict):
                continue
        if path.name.startswith(("results", "samples")):
            files.append(resolved)
    return tuple(files)


def _artifact_receipt(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(workspace.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


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
) -> str | None:
    if expected_policy is None:
        return None
    if not isinstance(gate, Mapping):
        return "quality_receipt_missing"
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
    checks = (
        ("primary_outcomes", primary_outcomes, "quality_primary_outcome_mismatch"),
        (
            "result_artifact_receipts",
            list(result_receipts),
            "quality_result_receipt_mismatch",
        ),
        (
            "sample_artifact_receipts",
            list(sample_receipts),
            "quality_sample_receipt_mismatch",
        ),
        ("outcome_digest", outcome_digest, "quality_outcome_digest_mismatch"),
        (
            "sample_set_digest",
            sample_set_digest,
            "quality_sample_set_digest_mismatch",
        ),
    )
    for name, expected, error in checks:
        if gate.get(name) != expected:
            return error
    if sample_set_digest is None:
        return "quality_samples_missing"
    return None


def _lm_eval_binding_data(
    workspace: Path,
    sources: tuple[Path, ...],
    artifacts: tuple[Path, ...],
    metrics: tuple[QualityMetric, ...],
) -> dict[str, Any]:
    primary = tuple(
        selected
        for task in sorted({item.task for item in metrics})
        if (selected := _select_primary(tuple(item for item in metrics if item.task == task)))
        is not None
    )
    results = tuple(_artifact_receipt(path, workspace) for path in sources)
    samples = tuple(
        path
        for path in artifacts
        if path.name.startswith("samples") and path.suffix == ".jsonl"
    )
    sample_receipts = tuple(_artifact_receipt(path, workspace) for path in samples)
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
) -> QualityEvidence:
    artifacts = _quality_artifacts(workspace)
    sources = tuple(
        path
        for path in artifacts
        if path.name.startswith("results") and path.suffix == ".json"
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
    data = json.loads(sources[0].read_text(encoding="utf-8"))
    metrics = _parse_metrics(data)
    binding = _lm_eval_binding_data(workspace, sources, artifacts, metrics)
    error = _binding_error(
        gate=gate,
        expected_policy=expected_policy,
        primary=binding["primary"],
        primary_outcomes=binding["outcomes"],
        result_receipts=binding["results"],
        sample_receipts=binding["samples"],
        outcome_digest=binding["outcome_digest"],
        sample_set_digest=binding["sample_digest"],
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
    )


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
    return QualityEvidence(
        required,
        "framework_quality_gate",
        passed,
        metrics,
        (report_path,),
        None if passed else "quality_gate_not_passed",
    )


def parse_quality_evidence(
    report: Mapping[str, Any],
    workspace: Path,
    report_path: Path,
    framework: str,
    required: bool,
    expected_evaluator_policy: Mapping[str, Any] | None,
) -> QualityEvidence:
    """Normalize serving or framework quality and bind formal evidence."""

    gate = report.get("quality_gate")
    if framework in SERVING_FRAMEWORKS or not isinstance(gate, Mapping):
        return _parse_lm_eval(
            workspace,
            required=required,
            gate=gate if isinstance(gate, Mapping) else None,
            expected_policy=expected_evaluator_policy,
        )
    return _parse_quality_gate(gate, report_path, required=required)


__all__ = ["QualityEvidence", "QualityMetric", "parse_quality_evidence"]
