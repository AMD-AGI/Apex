"""Phase-isolation validation for resolved benchmark configuration views."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError, IntegrityError, sha256_json
from apex.ports import BenchmarkPass
from apex.runtime import DependencyReceipt, MagpieConfigContract

from .evaluator_policy_validation import validate_evaluator_policy
from .instrumentation_validation import disabled as _disabled
from .instrumentation_validation import enabled as _enabled
from .instrumentation_validation import validate_instrumentation

VIEW_SCHEMA = "apex.benchmark-view.v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_QUALITY_FIELDS = frozenset({"required", "kind", "tasks", "evaluator_policy"})
_PHASE_PASSES = (BenchmarkPass.MEASUREMENT, BenchmarkPass.DIAGNOSTIC, BenchmarkPass.MEASUREMENT)


def validate_phase_set_contract(
    measurement: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    replay: Mapping[str, Any],
    expected_semantics_sha256: str,
) -> None:
    """Validate the self-contained invariants of one resolved phase set."""

    try:
        _validate_phase_set(measurement, diagnostic, replay, expected_semantics_sha256)
    except IntegrityError:
        raise
    except (
        AttributeError, ConfigurationError, KeyError, OSError,
        RuntimeError, TypeError, ValueError,
    ) as error:
        raise _phase_set_changed() from error


def _validate_phase_set(
    measurement: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    replay: Mapping[str, Any],
    expected: str,
) -> None:
    documents = (measurement, diagnostic, replay)
    benchmarks = tuple(_benchmark(document) for document in documents)
    metadata = tuple(_view_metadata(document) for document in documents)
    _validate_phase_metadata(metadata, expected)
    _validate_shared_envelope(documents)
    _validate_phase_roles(benchmarks, metadata)
    _validate_formal_replay_identity(documents, benchmarks)
    _validate_phase_quality(benchmarks, metadata)
    _validate_phase_instrumentation(benchmarks, metadata[0])
    _validate_phase_semantics(benchmarks, metadata, expected)


def _benchmark(document: Mapping[str, Any]) -> Mapping[str, Any]:
    benchmark = document.get("benchmark") if isinstance(document, Mapping) else None
    if not isinstance(benchmark, Mapping):
        raise ConfigurationError(
            "Resolved phase set lacks a benchmark mapping", "invalid_benchmark_view"
        )
    return benchmark


def _validate_shared_envelope(documents: tuple[Mapping[str, Any], ...]) -> None:
    envelopes = []
    for document in documents:
        projected = copy.deepcopy(dict(document))
        projected.pop("benchmark", None)
        apex = projected.get("apex")
        if not isinstance(apex, Mapping):
            raise _phase_set_changed()
        projected["apex"] = dict(apex)
        projected["apex"].pop("benchmark_view", None)
        envelopes.append(projected)
    if any(value != envelopes[0] for value in envelopes[1:]):
        raise _phase_set_changed()


def _validate_phase_metadata(
    metadata: tuple[Mapping[str, Any], ...], expected: str
) -> None:
    common: Mapping[str, Any] | None = None
    for observed, kind in zip(
        metadata, ("measurement", "diagnostic", "replay"), strict=True
    ):
        original = observed.get("original_sha256")
        valid = (
            observed.get("kind") == kind
            and isinstance(original, str)
            and bool(_SHA256.fullmatch(original))
            and observed.get("workload_semantics_sha256") == expected
        )
        _dependency_metadata(observed)
        _magpie_config_resolution_metadata(observed)
        _quality_metadata(observed)
        normalized = copy.deepcopy(dict(observed))
        normalized.pop("kind", None)
        normalized.pop("quality_contract", None)
        if not valid or (common is not None and normalized != common):
            raise _phase_set_changed()
        common = normalized


def _validate_phase_roles(
    benchmarks: tuple[Mapping[str, Any], ...],
    metadata: tuple[Mapping[str, Any], ...],
) -> None:
    for benchmark, view_metadata, pass_type in zip(
        benchmarks, metadata, _PHASE_PASSES, strict=True
    ):
        _validate_view_kind(view_metadata, pass_type)
        _validate_run_kind(benchmark, pass_type)
        _validate_quality_contract(benchmark, view_metadata, pass_type)
        validate_evaluator_policy(benchmark, view_metadata)


def _validate_formal_replay_identity(
    documents: tuple[Mapping[str, Any], ...],
    benchmarks: tuple[Mapping[str, Any], ...],
) -> None:
    images = tuple(_optional_image(benchmark) for benchmark in benchmarks)
    if images[0] != images[1]:
        raise _phase_set_changed()
    formal = _formal_document_projection(documents[0])
    replay = _formal_document_projection(documents[2])
    if formal != replay:
        raise _phase_set_changed()


def _optional_image(benchmark: Mapping[str, Any]) -> str | None:
    image = benchmark.get("docker_image")
    if image is not None and not isinstance(image, str):
        raise _phase_set_changed()
    return image.strip() or None if isinstance(image, str) else None


def _formal_document_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(document))
    projected["benchmark"].pop("docker_image", None)
    projected["apex"]["benchmark_view"]["kind"] = "measurement"
    return projected


def _validate_phase_quality(
    benchmarks: tuple[Mapping[str, Any], ...],
    metadata: tuple[Mapping[str, Any], ...],
) -> None:
    measurement, diagnostic, replay = benchmarks
    qualities = tuple(_quality_metadata(value) for value in metadata)
    formal, diagnosed, replayed = qualities
    if formal != replayed:
        raise _phase_set_changed()
    if formal.get("kind") != "lm_eval":
        expected = {
            "required": True,
            "kind": "framework_quality_gate",
            "tasks": "",
            "evaluator_policy": None,
        }
        if formal != expected or diagnosed != formal:
            raise _phase_set_changed()
        return
    expected_diagnostic = dict(formal)
    expected_diagnostic.update({"required": False, "kind": "trace_only"})
    runtime = measurement.get("lm_eval_runtime")
    valid = (
        diagnosed == expected_diagnostic
        and isinstance(formal.get("evaluator_policy"), Mapping)
        and isinstance(runtime, Mapping)
        and "lm_eval_runtime" not in diagnostic
        and replay.get("lm_eval_runtime") == runtime
    )
    if not valid:
        raise _phase_set_changed()


def _validate_phase_instrumentation(
    benchmarks: tuple[Mapping[str, Any], ...],
    measurement_metadata: Mapping[str, Any],
) -> None:
    dependencies = _dependency_metadata(measurement_metadata)
    tracelens = dependencies["tracelens"]
    tracelens_root = Path(str(tracelens["root"])).resolve()
    for benchmark, pass_type in zip(
        benchmarks, _PHASE_PASSES, strict=True
    ):
        validate_instrumentation(benchmark, pass_type, tracelens_root)


def _validate_phase_semantics(
    benchmarks: tuple[Mapping[str, Any], ...],
    metadata: tuple[Mapping[str, Any], ...],
    expected: str,
) -> None:
    measurement, diagnostic, replay = benchmarks
    formal = _workload_projection(measurement)
    diagnosed = _workload_projection(diagnostic)
    replayed = _workload_projection(replay)
    if _quality_metadata(metadata[0]).get("kind") == "lm_eval":
        formal_envs = formal.get("envs")
        diagnostic_envs = diagnosed.get("envs")
        if not isinstance(formal_envs, Mapping) or not isinstance(
            diagnostic_envs, dict
        ):
            raise _phase_set_changed()
        diagnostic_envs["RUN_EVAL"] = formal_envs["RUN_EVAL"]
        diagnosed["lm_eval_runtime"] = copy.deepcopy(formal["lm_eval_runtime"])
    if any(sha256_json(value) != expected for value in (formal, diagnosed, replayed)):
        raise _phase_set_changed()


def _workload_projection(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(benchmark))
    for key in ("profiler", "gap_analysis", "docker_image", "run_kind"):
        projected.pop(key, None)
    return projected


def _phase_set_changed() -> IntegrityError:
    return IntegrityError(
        "Resolved benchmark phase set changed workload or phase semantics",
        "benchmark_semantics_changed",
    )


def validate_view_contract(
    document: Mapping[str, Any],
    *,
    pass_type: BenchmarkPass,
    receipt: DependencyReceipt,
    tracelens_root: Path,
    tracelens_commit: str,
    expected_resolved: MagpieConfigContract | None = None,
) -> None:
    benchmark = document["benchmark"]
    metadata = _view_metadata(document)
    _validate_view_kind(metadata, pass_type)
    _validate_dependency_identity(
        metadata, receipt, tracelens_root, tracelens_commit
    )
    _magpie_config_resolution_metadata(metadata, expected_resolved)
    _validate_run_kind(benchmark, pass_type)
    _validate_quality_contract(benchmark, metadata, pass_type)
    _validate_semantics_identity(benchmark, metadata, pass_type, receipt)
    validate_evaluator_policy(benchmark, metadata)
    _validate_lm_eval_runtime(benchmark, metadata, receipt, pass_type)
    validate_instrumentation(benchmark, pass_type, tracelens_root)


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
    dependencies = _dependency_metadata(metadata)
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


def _dependency_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    dependencies = metadata.get("dependencies")
    components = (
        dependencies.get("magpie") if isinstance(dependencies, Mapping) else None,
        dependencies.get("tracelens") if isinstance(dependencies, Mapping) else None,
        dependencies.get("inferencex") if isinstance(dependencies, Mapping) else None,
    )
    valid = (
        isinstance(dependencies, Mapping)
        and isinstance(dependencies.get("receipt_schema"), str)
        and bool(dependencies.get("receipt_schema"))
        and isinstance(dependencies.get("lock_sha256"), str)
        and bool(_SHA256.fullmatch(str(dependencies.get("lock_sha256"))))
        and _absolute_locator(dependencies.get("python"))
        and all(_valid_dependency_component(value) for value in components)
    )
    if not valid:
        raise ConfigurationError(
            "Benchmark view lacks complete dependency receipt metadata",
            "benchmark_dependency_mismatch",
        )
    return dependencies


def _magpie_config_resolution_metadata(
    metadata: Mapping[str, Any],
    expected: MagpieConfigContract | None = None,
) -> Mapping[str, Any]:
    value = metadata.get("magpie_config_resolution")
    fields = {
        "plan_schema", "plan_sha256", "capability_schema",
        "capability_receipt_sha256", "effective_config_sha256",
        "scoring_config_sha256", "phase_views_sha256", "resolution_method_sha256",
    }
    valid = (
        isinstance(value, Mapping)
        and set(value) == fields
        and value.get("plan_schema") == "apex.magpie-main-resolved-plan/v1"
        and value.get("capability_schema")
        == "apex.magpie-main-capability-receipt/v1"
        and all(
            isinstance(value.get(field), str)
            and bool(_SHA256.fullmatch(str(value.get(field))))
            for field in fields - {"plan_schema", "capability_schema"}
        )
    )
    if not valid:
        raise ConfigurationError(
            "Benchmark view lacks its Apex Magpie main resolution binding",
            "magpie_config_resolution_mismatch",
        )
    if expected is not None:
        observed = dict(value)
        wanted = {
            "plan_schema": expected.plan["schema"],
            "plan_sha256": expected.plan["plan_sha256"],
            "capability_schema": expected.capability_receipt["schema"],
            "capability_receipt_sha256": expected.capability_receipt[
                "receipt_sha256"
            ],
            "effective_config_sha256": expected.plan["effective_config_sha256"],
            "scoring_config_sha256": expected.plan["scoring_config_sha256"],
            "phase_views_sha256": expected.plan["phase_views_sha256"],
            "resolution_method_sha256": expected.resolution_method_sha256,
        }
        if observed != wanted:
            raise ConfigurationError(
                "Benchmark view Magpie config resolution binding changed",
                "magpie_config_resolution_mismatch",
            )
    assert isinstance(value, Mapping)
    return value


def _valid_dependency_component(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and _absolute_locator(value.get("root"))
        and isinstance(value.get("commit"), str)
        and _COMMIT.fullmatch(str(value.get("commit")))
    )


def _absolute_locator(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).is_absolute()


def _quality_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    quality = metadata.get("quality_contract")
    if not isinstance(quality, Mapping) or set(quality) != _QUALITY_FIELDS:
        raise ConfigurationError(
            "Benchmark view lacks a complete quality contract",
            "quality_contract_missing",
        )
    return quality


def _validate_quality_contract(
    benchmark: Mapping[str, Any],
    metadata: Mapping[str, Any],
    pass_type: BenchmarkPass,
) -> None:
    quality = _quality_metadata(metadata)
    kind = quality.get("kind")
    if kind == "framework_quality_gate":
        if (
            quality.get("required") is not True
            or quality.get("tasks") != ""
            or quality.get("evaluator_policy") is not None
        ):
            raise ConfigurationError(
                "Framework benchmark view lacks its quality-gate contract",
                "quality_contract_missing",
            )
        return
    envs = benchmark.get("envs")
    if not isinstance(envs, Mapping) or not isinstance(quality, Mapping):
        raise ConfigurationError(
            "Serving benchmark view lacks a typed quality contract",
            "quality_contract_missing",
        )
    tasks = envs.get("MAGPIE_EVAL_TASKS")
    tasks_match = isinstance(tasks, str) and tasks == quality.get("tasks")
    if pass_type is BenchmarkPass.MEASUREMENT:
        valid = (
            _enabled(envs.get("RUN_EVAL"))
            and quality.get("required") is True
            and quality.get("kind") == "lm_eval"
            and tasks_match
        )
    else:
        valid = (
            _disabled(envs.get("RUN_EVAL"))
            and quality.get("required") is False
            and quality.get("kind") == "trace_only"
            and tasks_match
        )
    if not valid:
        lane = "measurement quality" if pass_type is BenchmarkPass.MEASUREMENT else "trace-only diagnostic"
        raise ConfigurationError(
            f"Serving benchmark view violates its {lane} contract",
            "quality_contract_missing",
        )


def _validate_lm_eval_runtime(
    benchmark: Mapping[str, Any],
    metadata: Mapping[str, Any],
    receipt: DependencyReceipt,
    pass_type: BenchmarkPass,
) -> None:
    if _quality_metadata(metadata).get("kind") not in {"lm_eval", "trace_only"}:
        return
    if pass_type is BenchmarkPass.DIAGNOSTIC:
        if "lm_eval_runtime" in benchmark:
            raise ConfigurationError(
                "Trace-only diagnostic cannot carry an lm-eval runtime",
                "benchmark_lm_eval_runtime_mismatch",
            )
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
    benchmark: Mapping[str, Any],
    metadata: Mapping[str, Any],
    pass_type: BenchmarkPass,
    receipt: DependencyReceipt,
) -> None:
    projected = _workload_projection(benchmark)
    if (
        pass_type is BenchmarkPass.DIAGNOSTIC
        and _quality_metadata(metadata).get("kind") == "trace_only"
    ):
        projected["envs"]["RUN_EVAL"] = "true"
        runtime = receipt.lm_eval_runtime
        if runtime is None:
            raise ConfigurationError(
                "Trace-only diagnostic lacks its formal evaluator binding",
                "benchmark_lm_eval_runtime_mismatch",
            )
        projected["lm_eval_runtime"] = {
            "path": str(runtime.root),
            "sha256": runtime.runtime_sha256,
            "identity": dict(runtime.identity),
        }
    observed = sha256_json(projected)
    if metadata.get("workload_semantics_sha256") != observed:
        raise ConfigurationError(
            "Benchmark workload semantics differ from the resolved view receipt",
            "benchmark_semantics_mismatch",
        )


__all__ = ["validate_phase_set_contract", "validate_view_contract"]
