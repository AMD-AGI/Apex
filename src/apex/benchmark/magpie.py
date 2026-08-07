"""Pinned Magpie subprocess adapter for benchmark measurement and diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from apex.core import ContractError, validate_identifier
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_CREDENTIAL_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    SubprocessSupervisor,
    build_subprocess_environment,
)
from apex.ports import BenchmarkRequest, BenchmarkResult
from apex.runtime import DependencyReceipt

from .config_views import validate_resolved_view
from .results import NormalizedBenchmarkResult, empty_result, parse_benchmark_report


def _lm_eval_expectation(
    benchmark: Mapping[str, object],
    quality_metadata: Mapping[str, object],
    receipt: DependencyReceipt,
):
    required = bool(quality_metadata.get("required", True))
    if not required or quality_metadata.get("kind") != "lm_eval":
        return None, None
    mode = str(benchmark.get("run_mode", "docker")).strip().lower()
    return receipt.lm_eval_runtime, mode


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

    def run_normalized(self, request: BenchmarkRequest) -> NormalizedBenchmarkResult:
        """Run Magpie and return the typed result used by E2E policy."""
        document = validate_resolved_view(
            request.config_path,
            pass_type=request.pass_type,
            dependency_receipt=self._receipt,
        )
        quality_metadata = document["apex"]["benchmark_view"]["quality_contract"]
        quality_required = bool(quality_metadata.get("required", True))
        benchmark = document["benchmark"]
        expected_lm_eval, expected_lm_eval_mode = _lm_eval_expectation(
            benchmark, quality_metadata, self._receipt)
        expected_model = str(benchmark.get("model", ""))
        envs = benchmark.get("envs", {})
        expected_revision = (
            str(envs.get("MODEL_REVISION")) if isinstance(envs, Mapping)
            and envs.get("MODEL_REVISION") else None
        )
        run_root = self._run_root(request)
        argv = (
            str(self._receipt.python),
            "-m",
            "Magpie",
            "benchmark",
            "--benchmark-config",
            str(request.config_path.resolve()),
            "--output-dir",
            str(run_root),
        )
        process = self._supervisor.run(
            argv,
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
            return empty_result(
                run_id=request.run_id,
                pass_type=request.pass_type,
                workspace=run_root,
                error=error,
                command_exit_code=process.exit_code,
                timed_out=process.timed_out,
                expected_lm_eval_runtime=expected_lm_eval,
                expected_lm_eval_execution_mode=expected_lm_eval_mode,
            )
        try:
            runtime_identity = self._receipt.lm_eval_runtime.identity if self._receipt.lm_eval_runtime else {}
            return parse_benchmark_report(
                report,
                run_id=request.run_id,
                pass_type=request.pass_type,
                quality_required=quality_required,
                command_exit_code=process.exit_code,
                timed_out=process.timed_out,
                expected_model=expected_model,
                expected_model_revision=expected_revision,
                expected_inferencex_root=self._receipt.root("inferencex").resolve(),
                expected_inferencex_commit=self._receipt.commits.get("inferencex"),
                expected_inferencex_tree=runtime_identity.get("inferencex_tree"),
                expected_lm_eval_runtime=expected_lm_eval,
                expected_lm_eval_execution_mode=expected_lm_eval_mode,
            )
        except Exception as error:
            return empty_result(
                run_id=request.run_id,
                pass_type=request.pass_type,
                workspace=report.parent,
                error=f"invalid_benchmark_evidence:{type(error).__name__}:{error}",
                command_exit_code=process.exit_code,
                timed_out=process.timed_out,
                expected_lm_eval_runtime=expected_lm_eval,
                expected_lm_eval_execution_mode=expected_lm_eval_mode,
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
