"""Pinned Magpie subprocess adapter for benchmark measurement and diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, sha256_file, validate_identifier
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_CREDENTIAL_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)
from apex.ports import BenchmarkPass, BenchmarkRequest, BenchmarkResult
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt

from .config_views import validate_resolved_view
from .results import NormalizedBenchmarkResult, empty_result, parse_benchmark_report

MAGPIE_HOST_RUNTIME_ENVIRONMENT_KEYS = (
    "MAGPIE_PROTECT_BENCHMARK_CONTAINER",
)


@dataclass(frozen=True, slots=True)
class _EvidenceExpectations:
    quality_required: bool
    evaluator_policy: Mapping[str, Any] | None
    config_sha256: str
    execution_mode: str
    requested_image: str | None
    lm_eval_runtime: LmEvalRuntimeReceipt | None
    lm_eval_mode: str | None
    model: str
    model_revision: str | None
    inferencex_tree: str | None
    allow_tracelens_derivation: bool
    tracelens_commit: str | None
    tracelens_tree: str | None


def _lm_eval_expectation(
    benchmark: Mapping[str, object],
    quality_metadata: Mapping[str, object],
    receipt: DependencyReceipt,
):
    required = bool(quality_metadata.get("required", True))
    kind = quality_metadata.get("kind")
    if not required and kind == "trace_only":
        return None, "not_requested"
    if not required or kind != "lm_eval":
        return None, None
    mode = str(benchmark.get("run_mode", "docker")).strip().lower()
    return receipt.lm_eval_runtime, mode


def _dependency_tree(receipt: DependencyReceipt, name: str) -> str | None:
    dependencies = receipt.raw.get("dependencies")
    if not isinstance(dependencies, Mapping):
        return None
    dependency = dependencies.get(name)
    if not isinstance(dependency, Mapping):
        return None
    tree = dependency.get("tree")
    return tree if isinstance(tree, str) else None


def _allows_tracelens_derivation(
    request: BenchmarkRequest,
    document: Mapping[str, Any],
) -> bool:
    benchmark = document.get("benchmark")
    apex = document.get("apex")
    if not isinstance(benchmark, Mapping) or not isinstance(apex, Mapping):
        return False
    view = apex.get("benchmark_view")
    quality = view.get("quality_contract") if isinstance(view, Mapping) else None
    profiler = benchmark.get("profiler")
    tracelens = profiler.get("tracelens") if isinstance(profiler, Mapping) else None
    return bool(
        request.pass_type is BenchmarkPass.DIAGNOSTIC
        and str(benchmark.get("run_mode", "docker")).strip().lower() == "docker"
        and isinstance(view, Mapping)
        and view.get("kind") == "diagnostic"
        and isinstance(quality, Mapping)
        and quality.get("required") is False
        and quality.get("kind") == "trace_only"
        and isinstance(tracelens, Mapping)
        and tracelens.get("enabled") is True
        and tracelens.get("analysis_mode") == "inference"
        and tracelens.get("auto_patch_runtime") is True
    )


class MagpieBenchmarkAdapter:
    """Execute the exact installed Magpie receipt without shell indirection."""

    def __init__(
        self,
        dependency_receipt: DependencyReceipt,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self._receipt = dependency_receipt
        self._magpie_root = dependency_receipt.root("magpie").resolve()
        self._tracelens_root = dependency_receipt.root("tracelens").resolve()
        self._supervisor = supervisor or SubprocessSupervisor()

    def _environment(self, overrides: Mapping[str, str]) -> dict[str, str]:
        return build_subprocess_environment(
            overrides,
            inherit=(
                *GPU_RUNTIME_ENVIRONMENT_KEYS,
                *HF_RUNTIME_ENVIRONMENT_KEYS,
                *DOCKER_RUNTIME_ENVIRONMENT_KEYS,
                *MAGPIE_HOST_RUNTIME_ENVIRONMENT_KEYS,
            ),
            allow_override_secrets=HF_CREDENTIAL_ENVIRONMENT_KEYS,
            fixed={
                "MAGPIE_ROOT": str(self._magpie_root),
                "TRACELENS_REPO_PATH": str(self._tracelens_root),
            },
            reserved=("MAGPIE_RUN_MODE",),
        )

    def _run_root(self, request: BenchmarkRequest) -> Path:
        validate_identifier(request.run_id, field_name="benchmark run_id")
        if not request.output_dir.is_absolute():
            raise ContractError(
                "Benchmark output_dir must be absolute", "invalid_benchmark_output"
            )
        if request.output_dir.exists() and request.output_dir.is_symlink():
            raise ContractError(
                "Benchmark output_dir cannot be a symlink", "invalid_benchmark_output"
            )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        run_root = request.output_dir / request.run_id / request.pass_type.value
        try:
            run_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ContractError(
                f"Immutable benchmark output already exists: {run_root}",
                "benchmark_output_exists",
            ) from error
        return run_root.resolve()

    @staticmethod
    def _find_report(run_root: Path) -> tuple[Path | None, str | None]:
        reports = tuple(sorted(run_root.rglob("benchmark_report.json")))
        if not reports:
            return None, "benchmark_report_missing"
        if len(reports) != 1:
            return None, "ambiguous_benchmark_reports"
        report = reports[0]
        if report.is_symlink():
            return None, "unsafe_benchmark_report"
        return report.resolve(), None

    def _benchmark_argv(
        self, request: BenchmarkRequest, run_root: Path
    ) -> tuple[str, ...]:
        return (
            str(self._receipt.python),
            "-m",
            "Magpie",
            "benchmark",
            "--benchmark-config",
            str(request.config_path.resolve()),
            "--output-dir",
            str(run_root),
        )

    def _expectations(
        self, request: BenchmarkRequest, document: Mapping[str, Any]
    ) -> _EvidenceExpectations:
        quality = document["apex"]["benchmark_view"]["quality_contract"]
        benchmark = document["benchmark"]
        mode = str(benchmark.get("run_mode", "docker")).strip().lower()
        image = benchmark.get("docker_image")
        envs = benchmark.get("envs", {})
        evaluator_policy = quality.get("evaluator_policy")
        lm_eval, lm_eval_mode = _lm_eval_expectation(
            benchmark, quality, self._receipt
        )
        identity = (
            self._receipt.lm_eval_runtime.identity
            if self._receipt.lm_eval_runtime
            else {}
        )
        allow_tracelens_derivation = _allows_tracelens_derivation(
            request, document
        )
        return _EvidenceExpectations(
            quality_required=bool(quality.get("required", True)),
            evaluator_policy=(
                evaluator_policy
                if isinstance(evaluator_policy, Mapping)
                else None
            ),
            config_sha256=sha256_file(request.config_path),
            execution_mode=mode,
            requested_image=(
                str(image)
                if mode == "docker" and isinstance(image, str)
                else None
            ),
            lm_eval_runtime=lm_eval,
            lm_eval_mode=lm_eval_mode,
            model=str(benchmark.get("model", "")),
            model_revision=(
                str(envs.get("MODEL_REVISION"))
                if isinstance(envs, Mapping) and envs.get("MODEL_REVISION")
                else None
            ),
            inferencex_tree=identity.get("inferencex_tree"),
            allow_tracelens_derivation=allow_tracelens_derivation,
            tracelens_commit=(
                self._receipt.commits.get("tracelens")
                if allow_tracelens_derivation
                else None
            ),
            tracelens_tree=(
                _dependency_tree(self._receipt, "tracelens")
                if allow_tracelens_derivation
                else None
            ),
        )

    @staticmethod
    def _empty(
        request: BenchmarkRequest,
        expectations: _EvidenceExpectations,
        process: ProcessResult,
        workspace: Path,
        error: str,
    ) -> NormalizedBenchmarkResult:
        return empty_result(
            run_id=request.run_id,
            pass_type=request.pass_type,
            workspace=workspace,
            error=error,
            command_exit_code=process.exit_code,
            timed_out=process.timed_out,
            expected_lm_eval_runtime=expectations.lm_eval_runtime,
            expected_lm_eval_execution_mode=expectations.lm_eval_mode,
            expected_config_sha256=expectations.config_sha256,
            expected_requested_image=expectations.requested_image,
            expected_execution_mode=expectations.execution_mode,
        )

    def _parse_report(
        self,
        request: BenchmarkRequest,
        expectations: _EvidenceExpectations,
        process: ProcessResult,
        report: Path,
    ) -> NormalizedBenchmarkResult:
        if sha256_file(request.config_path) != expectations.config_sha256:
            return self._empty(
                request,
                expectations,
                process,
                report.parent,
                "benchmark_config_changed_during_execution",
            )
        return parse_benchmark_report(
            report,
            run_id=request.run_id,
            pass_type=request.pass_type,
            quality_required=expectations.quality_required,
            command_exit_code=process.exit_code,
            timed_out=process.timed_out,
            expected_model=expectations.model,
            expected_model_revision=expectations.model_revision,
            expected_inferencex_root=self._receipt.root("inferencex").resolve(),
            expected_inferencex_commit=self._receipt.commits.get("inferencex"),
            expected_inferencex_tree=expectations.inferencex_tree,
            expected_lm_eval_runtime=expectations.lm_eval_runtime,
            expected_lm_eval_execution_mode=expectations.lm_eval_mode,
            expected_evaluator_policy=expectations.evaluator_policy,
            expected_config_sha256=expectations.config_sha256,
            expected_requested_image=expectations.requested_image,
            expected_execution_mode=expectations.execution_mode,
            allow_tracelens_derivation=expectations.allow_tracelens_derivation,
            expected_tracelens_commit=expectations.tracelens_commit,
            expected_tracelens_tree=expectations.tracelens_tree,
        )

    def run_normalized(self, request: BenchmarkRequest) -> NormalizedBenchmarkResult:
        """Run Magpie and return the typed result used by E2E policy."""
        document = validate_resolved_view(
            request.config_path,
            pass_type=request.pass_type,
            dependency_receipt=self._receipt,
        )
        expectations = self._expectations(request, document)
        run_root = self._run_root(request)
        process = self._supervisor.run(
            self._benchmark_argv(request, run_root),
            cwd=self._magpie_root,
            environment=self._environment(request.environment),
            timeout_seconds=request.timeout_seconds,
        )
        report, discovery_error = self._find_report(run_root)
        if report is None:
            error = discovery_error or "benchmark_report_missing"
            if process.timed_out:
                error = "benchmark_process_timeout"
            elif process.exit_code not in (0, None):
                error = f"benchmark_process_exit_{process.exit_code}"
            return self._empty(
                request, expectations, process, run_root, error
            )
        try:
            return self._parse_report(request, expectations, process, report)
        except Exception as error:
            return self._empty(
                request,
                expectations,
                process,
                report.parent,
                f"invalid_benchmark_evidence:{type(error).__name__}:{error}",
            )

    def run(self, request: BenchmarkRequest) -> BenchmarkResult:
        """Implement :class:`BenchmarkPort` using the generic port envelope."""

        result = self.run_normalized(request)
        return BenchmarkResult(
            run_id=result.run_id,
            pass_type=result.pass_type,
            succeeded=result.succeeded,
            report_path=result.report_path,
            workspace_path=result.workspace_path,
            metrics=result.metric_mapping(),
            artifact_paths=result.artifacts,
            error=";".join(result.errors) if result.errors else None,
        )


__all__ = ["MagpieBenchmarkAdapter"]
