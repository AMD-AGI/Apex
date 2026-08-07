"""Strict normalization of Magpie benchmark and quality artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError, IntegrityError
from apex.ports import BenchmarkPass

from .inferencex_runtime import InferenceXRuntimeEvidence
from .lm_eval_runtime import LmEvalRuntimeEvidence
from .model_revision import ModelRevisionEvidence
from .result_evidence import (
    Attestations,
    evidence_artifacts,
    parse_attestations,
    result_verdict,
)
from apex.runtime import LmEvalRuntimeReceipt


_SERVING_FRAMEWORKS = frozenset({"vllm", "sglang", "atom"})


@dataclass(frozen=True, slots=True)
class ThroughputMetrics:
    request_per_second: float | None
    output_tokens_per_second: float | None
    total_tokens_per_second: float | None
    completed_requests: int | None
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    mean_ms: float | None
    median_ms: float | None
    p99_ms: float | None
    std_ms: float | None


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    ttft: LatencyDistribution
    tpot: LatencyDistribution
    itl: LatencyDistribution
    e2el: LatencyDistribution


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


@dataclass(frozen=True, slots=True)
class NormalizedBenchmarkResult:
    """Caller-neutral metrics; diagnostic throughput is never reward truth."""

    schema_version: int
    run_id: str
    pass_type: BenchmarkPass
    succeeded: bool
    framework: str
    model: str
    workspace_path: Path
    report_path: Path | None
    throughput: ThroughputMetrics
    latency: LatencyMetrics
    quality: QualityEvidence
    profiling_enabled: bool
    run_kind: str
    reward_eligible: bool
    model_revision: ModelRevisionEvidence
    inferencex_runtime: InferenceXRuntimeEvidence
    artifacts: tuple[Path, ...]
    errors: tuple[str, ...]
    command_exit_code: int | None = None
    timed_out: bool = False
    lm_eval_runtime: LmEvalRuntimeEvidence = LmEvalRuntimeEvidence(
        False, True, None, None, None, None, None, None
    )

    def metric_mapping(self) -> Mapping[str, float | int | str | None]:
        """Flatten typed metrics for the stable generic BenchmarkPort."""

        values: dict[str, float | int | str | None] = {
            "request_throughput": self.throughput.request_per_second,
            "output_throughput": self.throughput.output_tokens_per_second,
            "total_token_throughput": self.throughput.total_tokens_per_second,
            "completed_requests": self.throughput.completed_requests,
            "duration_seconds": self.throughput.duration_seconds,
            "ttft_mean_ms": self.latency.ttft.mean_ms,
            "ttft_median_ms": self.latency.ttft.median_ms,
            "ttft_p99_ms": self.latency.ttft.p99_ms,
            "tpot_mean_ms": self.latency.tpot.mean_ms,
            "tpot_median_ms": self.latency.tpot.median_ms,
            "tpot_p99_ms": self.latency.tpot.p99_ms,
            "quality_required": int(self.quality.required),
            "quality_passed": int(self.quality.passed),
        }
        for metric in self.quality.metrics:
            values[f"quality.{metric.task}.{metric.name}"] = metric.value
        return values


def _finite_number(value: Any, *, nonnegative: bool = True) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        return None
    return result


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _distribution(value: Any) -> LatencyDistribution:
    data = value if isinstance(value, dict) else {}
    return LatencyDistribution(
        mean_ms=_finite_number(data.get("mean_ms")),
        median_ms=_finite_number(data.get("median_ms")),
        p99_ms=_finite_number(data.get("p99_ms")),
        std_ms=_finite_number(data.get("std_ms")),
    )


def _higher_is_better(name: str) -> bool:
    lowered = name.lower()
    return not any(
        marker in lowered
        for marker in ("loss", "perplexity", "ppl", "error", "wer", "cer")
    )


def _quality_files(workspace: Path) -> tuple[Path, ...]:
    quality_root = workspace.resolve() / "lm_eval"
    if quality_root.is_symlink() or not quality_root.is_dir():
        return ()
    files: list[Path] = []
    for path in sorted(quality_root.rglob("results*.json")):
        resolved = path.resolve()
        try:
            resolved.relative_to(quality_root)
        except ValueError as error:
            raise IntegrityError(
                f"Quality artifact escapes benchmark workspace: {path}",
                "unsafe_quality_artifact",
            ) from error
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(
                f"Quality artifact must be a regular file: {path}",
                "unsafe_quality_artifact",
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            files.append(resolved)
    return tuple(files)


def _parse_lm_eval(workspace: Path, *, required: bool) -> QualityEvidence:
    sources = _quality_files(workspace)
    if not sources:
        return QualityEvidence(
            required=required,
            kind="lm_eval",
            passed=not required,
            metrics=(),
            source_paths=(),
            error="quality_evidence_missing" if required else None,
        )
    if len(sources) != 1:
        return QualityEvidence(
            required=required,
            kind="lm_eval",
            passed=False,
            metrics=(),
            source_paths=sources,
            error="ambiguous_quality_evidence",
        )

    data = json.loads(sources[0].read_text(encoding="utf-8"))
    metrics: list[QualityMetric] = []
    for task, task_values in sorted(data["results"].items()):
        if not isinstance(task, str) or not isinstance(task_values, dict):
            continue
        for name, raw_value in sorted(task_values.items()):
            if not isinstance(name, str):
                continue
            metric_name = name.split(",", 1)[0]
            if "stderr" in metric_name.lower():
                continue
            value = _finite_number(raw_value, nonnegative=False)
            if value is None:
                continue
            metrics.append(
                QualityMetric(
                    task=task,
                    name=name,
                    value=value,
                    higher_is_better=_higher_is_better(metric_name),
                )
            )
    return QualityEvidence(
        required=required,
        kind="lm_eval",
        passed=bool(metrics),
        metrics=tuple(metrics),
        source_paths=sources,
        error=None if metrics else "quality_metrics_missing",
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
        and (value := _finite_number(raw, nonnegative=False)) is not None
    )
    passed = gate.get("passed") is True and gate.get("skipped") is not True
    return QualityEvidence(
        required=required,
        kind="framework_quality_gate",
        passed=passed,
        metrics=metrics,
        source_paths=(report_path,),
        error=None if passed else "quality_gate_not_passed",
    )


def _artifact_paths(
    report: Mapping[str, Any], report_path: Path, quality: QualityEvidence
) -> tuple[Path, ...]:
    workspace = report_path.parent.resolve()
    paths = {report_path.resolve(), *quality.source_paths}
    tracelens = report.get("tracelens_analysis")
    if isinstance(tracelens, dict):
        output_files = tracelens.get("output_files", [])
        if isinstance(output_files, list):
            for raw in output_files:
                if not isinstance(raw, str):
                    continue
                path = Path(raw)
                path = path if path.is_absolute() else workspace / path
                resolved = path.resolve()
                try:
                    resolved.relative_to(workspace)
                except ValueError:
                    continue
                if resolved.is_file() and not resolved.is_symlink():
                    paths.add(resolved)
    return tuple(sorted(paths))


def _normalized_result(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    workspace: Path,
    run_id: str,
    pass_type: BenchmarkPass,
    framework: str,
    throughput: ThroughputMetrics,
    latency: LatencyMetrics,
    quality: QualityEvidence,
    profiling_enabled: bool,
    run_kind: str,
    reward_eligible: bool,
    attestations: Attestations,
    success: bool,
    errors: tuple[str, ...],
    command_exit_code: int | None,
    timed_out: bool,
) -> NormalizedBenchmarkResult:
    model, inferencex, lm_eval = attestations
    return NormalizedBenchmarkResult(
        schema_version=1,
        run_id=run_id,
        pass_type=pass_type,
        succeeded=success,
        framework=framework,
        model=str(report.get("model", "")),
        workspace_path=workspace,
        report_path=report_path.resolve(),
        throughput=throughput,
        latency=latency,
        quality=quality,
        profiling_enabled=profiling_enabled,
        run_kind=run_kind,
        reward_eligible=reward_eligible,
        model_revision=model,
        inferencex_runtime=inferencex,
        artifacts=_artifact_paths(report, report_path.resolve(), quality)
        + evidence_artifacts(attestations),
        errors=errors,
        command_exit_code=command_exit_code,
        timed_out=timed_out,
        lm_eval_runtime=lm_eval,
    )


def parse_benchmark_report(
    report_path: Path,
    *,
    run_id: str,
    pass_type: BenchmarkPass,
    quality_required: bool,
    command_exit_code: int | None = 0,
    timed_out: bool = False,
    expected_model: str | None = None,
    expected_model_revision: str | None = None,
    expected_inferencex_root: Path | None = None,
    expected_inferencex_commit: str | None = None,
    expected_inferencex_tree: str | None = None,
    expected_lm_eval_runtime: LmEvalRuntimeReceipt | None = None,
    expected_lm_eval_execution_mode: str | None = None,
) -> NormalizedBenchmarkResult:
    """Parse one Magpie report plus its protected quality side artifacts."""

    report = _load_report(report_path)
    workspace = _report_workspace(report, report_path)
    throughput = _throughput_metrics(report.get("throughput"))
    latency = _latency_metrics(report.get("latency"))
    framework = str(report.get("framework", "")).strip().lower()
    quality = _quality_evidence(
        report, workspace, report_path.resolve(), framework, quality_required
    )
    run_kind = str(report.get("run_kind", "")).strip().lower()
    reward_eligible = report.get("reward_eligible") is True
    profiling_enabled = report.get("profiling_enabled") is True
    attestations = parse_attestations(
        report,
        report_path,
        expected_model,
        expected_model_revision,
        expected_inferencex_root,
        expected_inferencex_commit,
        expected_inferencex_tree,
        expected_lm_eval_runtime,
        expected_lm_eval_execution_mode,
    )
    lane_errors = _evidence_lane_errors(
        pass_type=pass_type,
        run_kind=run_kind,
        reward_eligible=reward_eligible,
        profiling_enabled=profiling_enabled,
    )
    base_errors = _result_errors(report, quality, command_exit_code, timed_out) + lane_errors
    success, errors = result_verdict(
        report,
        quality_passed=quality.passed,
        quality_required=quality.required,
        command_exit_code=command_exit_code,
        timed_out=timed_out,
        lane_errors=lane_errors,
        base_errors=base_errors,
        attestations=attestations,
    )
    return _normalized_result(
        report=report,
        report_path=report_path,
        workspace=workspace,
        run_id=run_id,
        pass_type=pass_type,
        framework=framework,
        throughput=throughput,
        latency=latency,
        quality=quality,
        profiling_enabled=profiling_enabled,
        run_kind=run_kind,
        reward_eligible=reward_eligible,
        attestations=attestations,
        success=success,
        errors=errors,
        command_exit_code=command_exit_code,
        timed_out=timed_out,
    )


def _load_report(report_path: Path) -> Mapping[str, Any]:
    if report_path.is_symlink() or not report_path.is_file():
        raise ConfigurationError(
            f"Magpie report is not a regular file: {report_path}",
            "benchmark_report_missing",
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            f"Invalid Magpie benchmark report {report_path}: {error}",
            "invalid_benchmark_report",
        ) from error
    if not isinstance(report, Mapping):
        raise ConfigurationError(
            "Magpie benchmark report root must be an object",
            "invalid_benchmark_report",
        )
    return report


def _report_workspace(report: Mapping[str, Any], report_path: Path) -> Path:
    workspace_raw = report.get("workspace_dir")
    workspace = (
        Path(workspace_raw).resolve()
        if isinstance(workspace_raw, str) and workspace_raw
        else report_path.parent.resolve()
    )
    if workspace != report_path.parent.resolve():
        raise IntegrityError(
            "Magpie report workspace_dir does not match its containing workspace",
            "benchmark_workspace_mismatch",
        )
    return workspace


def _throughput_metrics(value: Any) -> ThroughputMetrics:
    data = value if isinstance(value, Mapping) else {}
    return ThroughputMetrics(
        request_per_second=_finite_number(data.get("request_throughput")),
        output_tokens_per_second=_finite_number(data.get("output_throughput")),
        total_tokens_per_second=_finite_number(data.get("total_token_throughput")),
        completed_requests=_integer(data.get("completed_requests")),
        duration_seconds=_finite_number(data.get("duration_seconds")),
    )


def _latency_metrics(value: Any) -> LatencyMetrics:
    data = value if isinstance(value, Mapping) else {}
    return LatencyMetrics(
        ttft=_distribution(data.get("ttft")),
        tpot=_distribution(data.get("tpot")),
        itl=_distribution(data.get("itl")),
        e2el=_distribution(data.get("e2el")),
    )


def _quality_evidence(
    report: Mapping[str, Any],
    workspace: Path,
    report_path: Path,
    framework: str,
    required: bool,
) -> QualityEvidence:
    gate = report.get("quality_gate")
    if framework in _SERVING_FRAMEWORKS or not isinstance(gate, Mapping):
        return _parse_lm_eval(workspace, required=required)
    return _parse_quality_gate(gate, report_path, required=required)


def _result_errors(
    report: Mapping[str, Any],
    quality: QualityEvidence,
    command_exit_code: int | None,
    timed_out: bool,
) -> tuple[str, ...]:
    errors = tuple(str(error) for error in report.get("errors", []) if error)
    if quality.error:
        errors += (quality.error,)
    if timed_out:
        errors += ("benchmark_process_timeout",)
    elif command_exit_code != 0:
        errors += (f"benchmark_process_exit_{command_exit_code}",)
    return errors


def _evidence_lane_errors(
    *,
    pass_type: BenchmarkPass,
    run_kind: str,
    reward_eligible: bool,
    profiling_enabled: bool,
) -> tuple[str, ...]:
    expected_kind = (
        "measurement"
        if pass_type is BenchmarkPass.MEASUREMENT
        else "diagnostic"
    )
    errors: tuple[str, ...] = ()
    if run_kind != expected_kind:
        errors += ("benchmark_report_run_kind_mismatch",)
    expected_eligible = pass_type is BenchmarkPass.MEASUREMENT
    if reward_eligible is not expected_eligible:
        errors += ("benchmark_report_reward_eligibility_mismatch",)
    expected_profiling = pass_type is BenchmarkPass.DIAGNOSTIC
    if profiling_enabled is not expected_profiling:
        errors += ("benchmark_report_profiling_lane_mismatch",)
    return errors


def empty_result(
    *,
    run_id: str,
    pass_type: BenchmarkPass,
    workspace: Path,
    error: str,
    command_exit_code: int | None,
    timed_out: bool,
    expected_lm_eval_runtime: LmEvalRuntimeReceipt | None = None,
    expected_lm_eval_execution_mode: str | None = None,
) -> NormalizedBenchmarkResult:
    empty = LatencyDistribution(None, None, None, None)
    lm_eval_required = (
        expected_lm_eval_runtime is not None
        or expected_lm_eval_execution_mode is not None
    )
    return NormalizedBenchmarkResult(
        schema_version=1,
        run_id=run_id,
        pass_type=pass_type,
        succeeded=False,
        framework="",
        model="",
        workspace_path=workspace,
        report_path=None,
        throughput=ThroughputMetrics(None, None, None, None, None),
        latency=LatencyMetrics(empty, empty, empty, empty),
        quality=QualityEvidence(True, "lm_eval", False, (), (), error),
        profiling_enabled=False,
        run_kind="",
        reward_eligible=False,
        model_revision=ModelRevisionEvidence(
            False, False, None, None, None, error
        ),
        inferencex_runtime=InferenceXRuntimeEvidence(
            False, False, None, None, None, None, None, error
        ),
        artifacts=(),
        errors=(error,),
        command_exit_code=command_exit_code,
        timed_out=timed_out,
        lm_eval_runtime=LmEvalRuntimeEvidence(
            required=lm_eval_required,
            passed=not lm_eval_required,
            runtime_sha256=(
                expected_lm_eval_runtime.runtime_sha256
                if expected_lm_eval_runtime
                else None
            ),
            identity=(
                dict(expected_lm_eval_runtime.identity)
                if expected_lm_eval_runtime
                else None
            ),
            manifest_path=None,
            receipt_path=None,
            execution_mode=expected_lm_eval_execution_mode,
            read_only_mount=None,
            error=error if lm_eval_required else None,
        ),
    )


__all__ = [
    "LatencyDistribution",
    "LatencyMetrics",
    "NormalizedBenchmarkResult",
    "QualityEvidence",
    "QualityMetric",
    "ThroughputMetrics",
    "empty_result",
    "parse_benchmark_report",
]
