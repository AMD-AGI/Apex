"""Immutable phase-specific views of a Magpie benchmark configuration."""

from __future__ import annotations

import copy
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from apex.core import ConfigurationError, IntegrityError, sha256_bytes, sha256_json
from apex.ports import BenchmarkPass
from apex.runtime import DependencyReceipt, MagpieConfigContract

from .config_validation import validate_phase_set_contract, validate_view_contract
from .evaluator_policy import EvaluatorPolicy, evaluator_policy_from_scoring
from .resolved_view import (
    resolved_scoring_document,
    validate_resolved_binding,
    validated_source_roots,
)
from .runtime_inputs import pin_runtime_inputs


_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_VIEW_SCHEMA = "apex.benchmark-view.v2"


@dataclass(frozen=True, slots=True)
class TraceLensBinding:
    """Exact TraceLens checkout identity embedded in diagnostic views."""

    root: Path
    commit: str
    receipt_schema: str
    lock_sha256: str

    @classmethod
    def from_receipt(cls, receipt: DependencyReceipt) -> "TraceLensBinding":
        root = receipt.root("tracelens").resolve()
        commit = receipt.commits.get("tracelens", "")
        if not root.is_absolute() or not root.is_dir():
            raise ConfigurationError(
                "TraceLens receipt root must be an existing absolute directory",
                "invalid_tracelens_receipt",
            )
        if not _COMMIT.fullmatch(commit):
            raise ConfigurationError(
                "TraceLens receipt must contain a lowercase 40-hex commit",
                "invalid_tracelens_receipt",
            )
        if not _DIGEST.fullmatch(receipt.lock_sha256):
            raise ConfigurationError(
                "Dependency receipt lock digest must be lowercase SHA-256",
                "invalid_dependency_receipt",
            )
        return cls(root, commit, receipt.schema, receipt.lock_sha256)


@dataclass(frozen=True, slots=True)
class BenchmarkConfigViews:
    """Paths and identities for the four benchmark configuration artifacts."""

    original: Path
    measurement: Path
    diagnostic: Path
    replay: Path
    original_sha256: str
    workload_semantics_sha256: str
    quality_tasks: str
    evaluator_policy_sha256: str | None = None


def _load_yaml(content: bytes, *, source: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid benchmark YAML {source}: {error}",
            "invalid_benchmark_config",
        ) from error
    if not isinstance(loaded, dict) or not isinstance(loaded.get("benchmark"), dict):
        raise ConfigurationError(
            "Benchmark config must contain a 'benchmark' mapping",
            "invalid_benchmark_config",
        )
    return loaded


def _enabled(value: Any) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _freeze_quality_contract(
    benchmark: dict[str, Any], policy: EvaluatorPolicy | None
) -> str:
    envs = benchmark.setdefault("envs", {})
    if not isinstance(envs, dict):
        raise ConfigurationError(
            "benchmark.envs must be a mapping", "invalid_benchmark_config"
        )
    if "RUN_EVAL" not in envs:
        return ""

    configured = envs.get("RUN_EVAL")
    if not _enabled(configured):
        raise ConfigurationError(
            "Serving E2E optimization requires RUN_EVAL=true; the source config "
            "explicitly disables it",
            "quality_contract_disabled",
        )
    tasks = envs.get("MAGPIE_EVAL_TASKS")
    if not isinstance(tasks, str) or not tasks.strip():
        raise ConfigurationError(
            "MAGPIE_EVAL_TASKS must be a non-empty comma-separated string",
            "invalid_quality_contract",
        )
    tasks = ",".join(part.strip() for part in tasks.split(",") if part.strip())
    if not tasks:
        raise ConfigurationError(
            "MAGPIE_EVAL_TASKS contains no task", "invalid_quality_contract"
        )
    envs["MAGPIE_EVAL_TASKS"] = tasks
    if policy is not None:
        for name, expected in policy.env().items():
            observed = envs.get(name)
            if observed is not None and str(observed) != expected:
                raise ConfigurationError(
                    f"Evaluator policy field {name} was overridden",
                    "evaluator_policy_override",
                )
            envs[name] = expected
        tasks = policy.tasks
    return tasks


def _disable_instrumentation(benchmark: dict[str, Any]) -> None:
    profiler = benchmark.setdefault("profiler", {})
    if not isinstance(profiler, dict):
        raise ConfigurationError(
            "benchmark.profiler must be a mapping", "invalid_benchmark_config"
        )
    for value in profiler.values():
        if isinstance(value, dict):
            value["enabled"] = False
    for key in ("torch_profiler", "system_profiler", "tracelens", "gpu_monitor", "targeted_trace"):
        value = profiler.setdefault(key, {})
        if not isinstance(value, dict):
            raise ConfigurationError(
                f"benchmark.profiler.{key} must be a mapping",
                "invalid_benchmark_config",
            )
        value["enabled"] = False
    gap = benchmark.setdefault("gap_analysis", {})
    if not isinstance(gap, dict):
        raise ConfigurationError(
            "benchmark.gap_analysis must be a mapping", "invalid_benchmark_config"
        )
    gap["enabled"] = False


def _enable_diagnostics(
    benchmark: dict[str, Any],
    binding: TraceLensBinding,
    source_repository_roots: Sequence[Path],
) -> None:
    _disable_instrumentation(benchmark)
    profiler = benchmark["profiler"]
    profiler["torch_profiler"]["enabled"] = True
    profiler["gpu_monitor"]["enabled"] = True
    tracelens = profiler["tracelens"]
    tracelens.update(
        {
            "enabled": True,
            "analysis_mode": "inference",
            "auto_patch_runtime": True,
            "tracelens_repo_path": str(binding.root),
        }
    )
    profiler["targeted_trace"].update(
        {
            "enabled": True,
            "backend": "torch_profiler",
            "run_seed": "apex-workload-diagnostic-v1",
            "sample_rate": 1.0,
            "max_records_per_shard": 100000,
            "targets": [
                {
                    "target_id": "workload-kernels",
                    "variant_id": "baseline",
                    "name_patterns": ["*"],
                }
            ],
        }
    )
    gap = benchmark["gap_analysis"]
    gap.update(
        {
            "enabled": True,
            "find_kernel_sources": bool(source_repository_roots),
            "kernel_source_repos": [str(path) for path in source_repository_roots],
            "auto_clone_repos": False,
        }
    )


def _workload_projection(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(benchmark))
    projected.pop("profiler", None)
    projected.pop("gap_analysis", None)
    projected.pop("docker_image", None)
    projected.pop("run_kind", None)
    return projected


def _metadata(
    *,
    kind: str,
    original_sha256: str,
    semantics_sha256: str,
    binding: TraceLensBinding,
    receipt: DependencyReceipt,
    quality_tasks: str,
    evaluator_policy: EvaluatorPolicy | None,
    resolved: MagpieConfigContract,
    diagnostic_trace_only: bool = False,
) -> dict[str, Any]:
    return {
        "benchmark_view": {
            "schema": _VIEW_SCHEMA,
            "kind": kind,
            "original_sha256": original_sha256,
            "workload_semantics_sha256": semantics_sha256,
            "magpie_config_resolution": {
                "plan_schema": resolved.plan["schema"],
                "plan_sha256": resolved.plan["plan_sha256"],
                "capability_schema": resolved.capability_receipt["schema"],
                "capability_receipt_sha256": resolved.capability_receipt[
                    "receipt_sha256"
                ],
                "effective_config_sha256": resolved.plan[
                    "effective_config_sha256"
                ],
                "scoring_config_sha256": resolved.plan["scoring_config_sha256"],
                "phase_views_sha256": resolved.plan["phase_views_sha256"],
                "resolution_method_sha256": resolved.resolution_method_sha256,
            },
            "dependencies": {
                "receipt_schema": binding.receipt_schema,
                "lock_sha256": binding.lock_sha256,
                "python": str(receipt.python),
                "magpie": {
                    "root": str(receipt.root("magpie").resolve()),
                    "commit": receipt.commits.get("magpie", ""),
                },
                "tracelens": {
                    "root": str(binding.root),
                    "commit": binding.commit,
                },
                "inferencex": {
                    "root": str(receipt.root("inferencex").resolve()),
                    "commit": receipt.commits.get("inferencex", ""),
                },
            },
            "quality_contract": {
                "required": not diagnostic_trace_only,
                "kind": (
                    "trace_only"
                    if diagnostic_trace_only
                    else "lm_eval" if quality_tasks else "framework_quality_gate"
                ),
                "tasks": quality_tasks,
                "evaluator_policy": (
                    evaluator_policy.to_dict() if evaluator_policy else None
                ),
            },
        }
    }


def _dump(document: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(document), sort_keys=False, default_flow_style=False, allow_unicode=True
    ).encode("utf-8")


def _attach_metadata(document: dict[str, Any], metadata: Mapping[str, Any]) -> None:
    apex_metadata = document.setdefault("apex", {})
    if not isinstance(apex_metadata, dict):
        raise ConfigurationError(
            "Top-level apex metadata must be a mapping",
            "invalid_benchmark_config",
        )
    if "benchmark_view" in apex_metadata:
        raise ConfigurationError(
            "Input benchmark already contains reserved apex.benchmark_view metadata",
            "reserved_benchmark_metadata",
        )
    apex_metadata.update(copy.deepcopy(dict(metadata)))


def _write_immutable(path: Path, content: bytes) -> None:
    if path.is_symlink():
        raise IntegrityError(f"Refusing benchmark view symlink: {path}", "unsafe_path")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise IntegrityError(
                f"Benchmark view already exists with different content: {path}",
                "immutable_benchmark_view",
            )
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _prepare_view_paths(original_config: Path, output_dir: Path) -> tuple[Path, Path]:
    source = original_config.resolve()
    if not source.is_file() or original_config.is_symlink():
        raise ConfigurationError(
            f"Benchmark config is not a regular file: {original_config}",
            "benchmark_config_missing",
        )
    destination = output_dir.resolve()
    if output_dir.exists() and output_dir.is_symlink():
        raise IntegrityError(
            f"Refusing benchmark view directory symlink: {output_dir}", "unsafe_path"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return source, destination


def _measurement_document(
    document: Mapping[str, Any],
    original_sha256: str,
    binding: TraceLensBinding,
    receipt: DependencyReceipt,
    evaluator_policy: EvaluatorPolicy | None,
    resolved: MagpieConfigContract,
) -> tuple[dict[str, Any], str, str]:
    result = copy.deepcopy(dict(document))
    benchmark = result["benchmark"]
    quality_tasks = _freeze_quality_contract(benchmark, evaluator_policy)
    _disable_instrumentation(benchmark)
    benchmark["run_kind"] = "measurement"
    semantics_sha256 = sha256_json(_workload_projection(benchmark))
    _attach_metadata(
        result,
        _metadata(
            kind="measurement",
            original_sha256=original_sha256,
            semantics_sha256=semantics_sha256,
            binding=binding,
            receipt=receipt,
            quality_tasks=quality_tasks,
            evaluator_policy=evaluator_policy,
            resolved=resolved,
        ),
    )
    return result, quality_tasks, semantics_sha256


def _diagnostic_document(
    document: Mapping[str, Any],
    original_sha256: str,
    semantics_sha256: str,
    quality_tasks: str,
    binding: TraceLensBinding,
    receipt: DependencyReceipt,
    source_repository_roots: Sequence[Path],
    evaluator_policy: EvaluatorPolicy | None,
    resolved: MagpieConfigContract,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    _freeze_quality_contract(result["benchmark"], evaluator_policy)
    diagnostic_trace_only = bool(quality_tasks)
    if diagnostic_trace_only:
        result["benchmark"]["envs"]["RUN_EVAL"] = "false"
        result["benchmark"].pop("lm_eval_runtime", None)
    _enable_diagnostics(result["benchmark"], binding, source_repository_roots)
    result["benchmark"]["run_kind"] = "diagnostic"
    _attach_metadata(
        result,
        _metadata(
            kind="diagnostic",
            original_sha256=original_sha256,
            semantics_sha256=semantics_sha256,
            binding=binding,
            receipt=receipt,
            quality_tasks=quality_tasks,
            evaluator_policy=evaluator_policy,
            resolved=resolved,
            diagnostic_trace_only=diagnostic_trace_only,
        ),
    )
    return result


def _replay_document(
    measurement: Mapping[str, Any],
    replay_image: str | None,
    original_sha256: str,
    semantics_sha256: str,
    quality_tasks: str,
    binding: TraceLensBinding,
    receipt: DependencyReceipt,
    evaluator_policy: EvaluatorPolicy | None,
    resolved: MagpieConfigContract,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(measurement))
    if replay_image is not None:
        if not isinstance(replay_image, str) or not replay_image.strip():
            raise ConfigurationError(
                "Replay image locator must be non-empty", "invalid_replay_image"
            )
        run_mode = str(result["benchmark"].get("run_mode", "docker")).lower()
        if run_mode != "docker":
            raise ConfigurationError(
                "A derived replay image applies only to Magpie Docker workloads",
                "replay_image_not_applicable",
            )
        result["benchmark"]["docker_image"] = replay_image.strip()
    metadata = _metadata(
        kind="replay",
        original_sha256=original_sha256,
        semantics_sha256=semantics_sha256,
        binding=binding,
        receipt=receipt,
        quality_tasks=quality_tasks,
        evaluator_policy=evaluator_policy,
        resolved=resolved,
    )
    result["apex"]["benchmark_view"] = metadata["benchmark_view"]
    return result


def _view_paths(
    destination: Path,
    original_sha256: str,
    semantics_sha256: str,
    quality_tasks: str,
    evaluator_policy: EvaluatorPolicy | None,
) -> BenchmarkConfigViews:
    return BenchmarkConfigViews(
        original=destination / "benchmark.original.yaml",
        measurement=destination / "benchmark.measurement.resolved.yaml",
        diagnostic=destination / "benchmark.diagnostic.resolved.yaml",
        replay=destination / "benchmark.replay.yaml",
        original_sha256=original_sha256,
        workload_semantics_sha256=semantics_sha256,
        quality_tasks=quality_tasks,
        evaluator_policy_sha256=(evaluator_policy.digest if evaluator_policy else None),
    )


def _write_views(
    paths: BenchmarkConfigViews,
    original: bytes,
    measurement: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> None:
    _write_immutable(paths.original, original)
    _write_immutable(paths.measurement, _dump(measurement))
    _write_immutable(paths.diagnostic, _dump(diagnostic))
    _write_immutable(paths.replay, _dump(replay))


def build_config_views(
    original_config: Path,
    output_dir: Path,
    *,
    dependency_receipt: DependencyReceipt,
    resolved_contract: MagpieConfigContract,
    replay_image: str | None = None,
    source_repository_roots: Sequence[Path] = (),
    model_revision: str | None = None,
    hf_cache_path: Path | None = None,
    gpu_devices: str | None = None,
    hf_offline: bool = False,
) -> BenchmarkConfigViews:
    """Create immutable phase-specific benchmark configuration views."""

    source, destination = _prepare_view_paths(original_config, output_dir)
    source_roots = validated_source_roots(source_repository_roots)
    binding = TraceLensBinding.from_receipt(dependency_receipt)
    original_bytes = source.read_bytes()
    original_sha256 = sha256_bytes(original_bytes)
    validate_resolved_binding(
        source, original_sha256, dependency_receipt, resolved_contract
    )
    original_document = _load_yaml(original_bytes, source=source)
    document = resolved_scoring_document(original_document, resolved_contract)
    evaluator_policy = evaluator_policy_from_scoring(document["benchmark"])
    pin_runtime_inputs(
        document["benchmark"],
        dependency_receipt,
        model_revision=model_revision,
        hf_cache_path=hf_cache_path,
        gpu_devices=gpu_devices,
        hf_offline=hf_offline,
    )
    measurement, quality_tasks, semantics_sha256 = _measurement_document(
        document, original_sha256, binding, dependency_receipt, evaluator_policy,
        resolved_contract,
    )
    diagnostic = _diagnostic_document(
        document,
        original_sha256,
        semantics_sha256,
        quality_tasks,
        binding,
        dependency_receipt,
        source_roots,
        evaluator_policy,
        resolved_contract,
    )
    replay = _replay_document(
        measurement,
        replay_image,
        original_sha256,
        semantics_sha256,
        quality_tasks,
        binding,
        dependency_receipt,
        evaluator_policy,
        resolved_contract,
    )
    validate_phase_set_contract(
        measurement, diagnostic, replay, semantics_sha256
    )
    paths = _view_paths(
        destination,
        original_sha256,
        semantics_sha256,
        quality_tasks,
        evaluator_policy,
    )
    _write_views(paths, original_bytes, measurement, diagnostic, replay)
    return paths


def validate_resolved_view(
    path: Path,
    *,
    pass_type: BenchmarkPass,
    dependency_receipt: DependencyReceipt,
    expected_resolved: MagpieConfigContract | None = None,
) -> Mapping[str, Any]:
    """Fail before execution when a resolved view violates phase isolation."""

    document = _load_yaml(path.read_bytes(), source=path)
    binding = TraceLensBinding.from_receipt(dependency_receipt)
    validate_view_contract(
        document,
        pass_type=pass_type,
        receipt=dependency_receipt,
        tracelens_root=binding.root,
        tracelens_commit=binding.commit,
        expected_resolved=expected_resolved,
    )
    return document


__all__ = [
    "BenchmarkConfigViews",
    "TraceLensBinding",
    "build_config_views",
    "validate_resolved_view",
]
