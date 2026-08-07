"""Phase-isolation validation for resolved benchmark configuration views."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError, sha256_json
from apex.ports import BenchmarkPass
from apex.runtime import DependencyReceipt


VIEW_SCHEMA = "apex.benchmark-view.v1"
SERVING_FRAMEWORKS = frozenset({"vllm", "sglang", "atom"})


def validate_view_contract(
    document: Mapping[str, Any],
    *,
    pass_type: BenchmarkPass,
    receipt: DependencyReceipt,
    tracelens_root: Path,
    tracelens_commit: str,
) -> None:
    benchmark = document["benchmark"]
    metadata = _view_metadata(document)
    _validate_view_kind(metadata, pass_type)
    _validate_dependency_identity(
        metadata, receipt, tracelens_root, tracelens_commit
    )
    _validate_semantics_identity(benchmark, metadata)
    _validate_run_kind(benchmark, pass_type)
    _validate_quality_enabled(benchmark)
    _validate_evaluator_policy(benchmark, metadata)
    _validate_lm_eval_runtime(benchmark, receipt)
    _validate_instrumentation(benchmark, pass_type, tracelens_root)


def _view_metadata(document: Mapping[str, Any]) -> Mapping[str, Any]:
    apex = document.get("apex", {})
    metadata = apex.get("benchmark_view", {}) if isinstance(apex, Mapping) else {}
    if not isinstance(metadata, Mapping) or metadata.get("schema") != VIEW_SCHEMA:
        raise ConfigurationError(
            "Magpie adapter accepts only Apex resolved benchmark views",
            "unresolved_benchmark_view",
        )
    return metadata


def _validate_view_kind(
    metadata: Mapping[str, Any], pass_type: BenchmarkPass
) -> None:
    kind = metadata.get("kind")
    expected = (
        {"measurement", "replay"}
        if pass_type is BenchmarkPass.MEASUREMENT
        else {"diagnostic"}
    )
    if kind not in expected:
        raise ConfigurationError(
            f"Benchmark view kind {kind!r} is invalid for pass {pass_type.value}",
            "benchmark_pass_mismatch",
        )


def _validate_dependency_identity(
    metadata: Mapping[str, Any],
    receipt: DependencyReceipt,
    tracelens_root: Path,
    tracelens_commit: str,
) -> None:
    dependencies = metadata.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ConfigurationError(
            "Benchmark view lacks dependency receipt identity",
            "benchmark_dependency_mismatch",
        )
    magpie = dependencies.get("magpie")
    tracelens = dependencies.get("tracelens")
    inferencex = dependencies.get("inferencex")
    matches = (
        dependencies.get("receipt_schema") == receipt.schema
        and dependencies.get("lock_sha256") == receipt.lock_sha256
        and dependencies.get("python") == str(receipt.python)
        and isinstance(magpie, Mapping)
        and magpie.get("root") == str(receipt.root("magpie").resolve())
        and magpie.get("commit") == receipt.commits.get("magpie")
        and isinstance(tracelens, Mapping)
        and tracelens.get("root") == str(tracelens_root)
        and tracelens.get("commit") == tracelens_commit
        and isinstance(inferencex, Mapping)
        and inferencex.get("root") == str(receipt.root("inferencex").resolve())
        and inferencex.get("commit") == receipt.commits.get("inferencex")
    )
    if not matches:
        raise ConfigurationError(
            "Benchmark view dependency identity differs from the runtime receipt",
            "benchmark_dependency_mismatch",
        )


def _validate_quality_enabled(benchmark: Mapping[str, Any]) -> None:
    envs = benchmark.get("envs")
    framework = str(benchmark.get("framework", "")).strip().lower()
    if framework in SERVING_FRAMEWORKS and (
        not isinstance(envs, Mapping) or not _enabled(envs.get("RUN_EVAL"))
    ):
        raise ConfigurationError(
            "Serving benchmark view is missing RUN_EVAL=true",
            "quality_contract_missing",
        )


def _validate_evaluator_policy(
    benchmark: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    quality = metadata.get("quality_contract")
    policy = quality.get("evaluator_policy") if isinstance(quality, Mapping) else None
    if policy is None:
        return
    envs = benchmark.get("envs")
    expected = {
        "MAGPIE_EVAL_POLICY_ID": policy.get("policy_id"),
        "MAGPIE_EVAL_TASKS": policy.get("tasks"),
        "MAGPIE_EVAL_PRIMARY_METRIC": policy.get("primary_metric"),
        "MAGPIE_EVAL_MAX_LENGTH": str(policy.get("max_length")),
        "MAGPIE_EVAL_MAX_GEN_TOKENS": str(policy.get("max_gen_tokens")),
    }
    if not isinstance(envs, Mapping) or any(
        str(envs.get(name)) != str(value) for name, value in expected.items()
    ):
        raise ConfigurationError(
            "Benchmark evaluator policy differs from its resolved receipt",
            "benchmark_evaluator_policy_mismatch",
        )


def _validate_lm_eval_runtime(
    benchmark: Mapping[str, Any], receipt: DependencyReceipt
) -> None:
    framework = str(benchmark.get("framework", "")).strip().lower()
    if framework not in SERVING_FRAMEWORKS:
        return
    runtime = receipt.lm_eval_runtime
    configured = benchmark.get("lm_eval_runtime")
    expected = (
        {
            "path": str(runtime.root),
            "sha256": runtime.runtime_sha256,
            "identity": dict(runtime.identity),
        }
        if runtime is not None
        else None
    )
    if not isinstance(configured, Mapping) or dict(configured) != expected:
        raise ConfigurationError(
            "Benchmark lm-eval runtime differs from the verified runtime receipt",
            "benchmark_lm_eval_runtime_mismatch",
        )


def _validate_run_kind(
    benchmark: Mapping[str, Any], pass_type: BenchmarkPass
) -> None:
    expected = (
        "measurement"
        if pass_type is BenchmarkPass.MEASUREMENT
        else "diagnostic"
    )
    if benchmark.get("run_kind") != expected:
        raise ConfigurationError(
            f"Benchmark pass {pass_type.value} requires run_kind={expected!r}",
            "benchmark_run_kind_mismatch",
        )


def _validate_semantics_identity(
    benchmark: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    projected = copy.deepcopy(dict(benchmark))
    for key in ("profiler", "gap_analysis", "docker_image", "run_kind"):
        projected.pop(key, None)
    observed = sha256_json(projected)
    if metadata.get("workload_semantics_sha256") != observed:
        raise ConfigurationError(
            "Benchmark workload semantics differ from the resolved view receipt",
            "benchmark_semantics_mismatch",
        )


def _validate_instrumentation(
    benchmark: Mapping[str, Any], pass_type: BenchmarkPass, tracelens_root: Path
) -> None:
    profiler = benchmark.get("profiler")
    gap = benchmark.get("gap_analysis")
    if not isinstance(profiler, Mapping) or not isinstance(gap, Mapping):
        raise ConfigurationError(
            "Resolved benchmark view lacks profiler/gap mappings",
            "invalid_benchmark_view",
        )
    if pass_type is BenchmarkPass.MEASUREMENT:
        _validate_measurement_instrumentation(profiler, gap)
    else:
        _validate_diagnostic_instrumentation(profiler, gap, tracelens_root)


def _validate_measurement_instrumentation(
    profiler: Mapping[str, Any], gap: Mapping[str, Any]
) -> None:
    enabled = [
        name
        for name, value in profiler.items()
        if isinstance(value, Mapping) and _enabled(value.get("enabled"))
    ]
    if enabled or _enabled(gap.get("enabled")):
        raise ConfigurationError(
            f"Measurement view enables instrumentation: {enabled}",
            "measurement_profiler_enabled",
        )


def _validate_diagnostic_instrumentation(
    profiler: Mapping[str, Any],
    gap: Mapping[str, Any],
    tracelens_root: Path,
) -> None:
    torch = profiler.get("torch_profiler")
    tracelens = profiler.get("tracelens")
    targeted = profiler.get("targeted_trace")
    valid = (
        isinstance(torch, Mapping)
        and _enabled(torch.get("enabled"))
        and isinstance(tracelens, Mapping)
        and _enabled(tracelens.get("enabled"))
        and Path(str(tracelens.get("tracelens_repo_path", ""))).resolve()
        == tracelens_root
        and isinstance(targeted, Mapping)
        and _enabled(targeted.get("enabled"))
        and bool(targeted.get("targets"))
        and _enabled(gap.get("enabled"))
    )
    if not valid:
        raise ConfigurationError(
            "Diagnostic view must enable Torch profiler, TraceLens at the pinned "
            "root, TargetedKernelTrace, and gap analysis",
            "invalid_diagnostic_view",
        )


def _enabled(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "1", "true", "yes", "on"
    }


__all__ = ["validate_view_contract"]
