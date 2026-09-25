"""Pinned Magpie subprocess adapter for benchmark measurement and diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, sha256_file, sha256_json, validate_identifier
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_CREDENTIAL_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)
from apex.ports import (
    BenchmarkPass,
    BenchmarkRequest,
    BenchmarkResult,
    MagpieAttestationRequest,
    MagpieExecutionAttestor,
    MagpieFormalMeasurementSupport,
    RayExecutionContract,
)
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt

from .config_views import validate_resolved_view
from .magpie_execution_lifecycle import (
    prepare_magpie_execution,
    run_magpie_execution,
)
from .magpie_attestation import UnavailableMagpieExecutionAttestor
from .results import NormalizedBenchmarkResult, empty_result, parse_benchmark_report

MAGPIE_HOST_RUNTIME_ENVIRONMENT_KEYS = (
    "MAGPIE_PROTECT_BENCHMARK_CONTAINER",
)


@dataclass(frozen=True, slots=True)
class _EvidenceExpectations:
    quality_required: bool
    quality_kind: str
    evaluator_policy: Mapping[str, Any] | None
    config_sha256: str
    gpu_lease_digest: str | None
    execution_mode: str
    lifecycle: str
    requested_image: str | None
    lm_eval_runtime: LmEvalRuntimeReceipt | None
    lm_eval_mode: str | None
    model: str
    model_revision: str | None
    inferencex_tree: str | None
    allow_tracelens_derivation: bool
    tracelens_commit: str | None
    tracelens_tree: str | None
    ray_contract: RayExecutionContract | None
    evaluator_endpoint_port: int
    evaluator_concurrent_requests: int


def _ray_contract(benchmark: Mapping[str, object]) -> RayExecutionContract | None:
    if str(benchmark.get("run_mode", "docker")).strip().lower() != "ray":
        return None
    raw = benchmark.get("ray_config")
    if not isinstance(raw, Mapping):
        raise ContractError("Ray config is missing", "invalid_magpie_ray_config")
    address = raw.get("cluster_address", "auto")
    shared = raw.get("shared_storage_path")
    multi_node = raw.get("multi_node", False)
    num_nodes = raw.get("num_nodes", 1)
    total_gpus = raw.get("total_num_gpus", 8)
    per_node = raw.get("gpus_per_node", 8)
    integers = (num_nodes, total_gpus, per_node)
    valid_integers = all(
        not isinstance(value, bool) and isinstance(value, int) and value > 0
        for value in integers
    )
    if (
        not isinstance(address, str)
        or not address
        or len(address) > 2048
        or any(character.isspace() for character in address)
        or not isinstance(shared, str)
        or not Path(shared).is_absolute()
        or Path(shared) == Path("/")
        or not isinstance(multi_node, bool)
        or not valid_integers
        or (not multi_node and num_nodes != 1)
    ):
        raise ContractError("Ray config is invalid", "invalid_magpie_ray_config")
    return RayExecutionContract(
        address,
        Path(shared),
        sha256_json(dict(raw)),
        multi_node,
        num_nodes,
        total_gpus,
        per_node,
    )


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
    return receipt.lm_eval_runtime, "local" if mode == "ray" else mode


def _dependency_tree(receipt: DependencyReceipt, name: str) -> str | None:
    dependencies = receipt.raw.get("dependencies")
    if not isinstance(dependencies, Mapping):
        return None
    dependency = dependencies.get(name)
    if not isinstance(dependency, Mapping):
        return None
    tree = dependency.get("tree")
    return tree if isinstance(tree, str) else None


def _enabled(value: object) -> bool:
    return value is True or value == 1 or (
        isinstance(value, str)
        and value.strip().lower() in {"1", "true", "yes", "on"}
    )


def _positive_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ContractError("Evaluator integer input is invalid", "invalid_evaluator_policy")
    text = str(value)
    if not text.isdigit() or text.startswith("0"):
        raise ContractError("Evaluator integer input is invalid", "invalid_evaluator_policy")
    return int(text)


def _lifecycle(benchmark: Mapping[str, object]) -> str:
    value = benchmark.get("server_lifecycle")
    if not isinstance(value, Mapping) or not _enabled(value.get("enabled")):
        return "one_shot"
    return "cleanup" if _enabled(value.get("cleanup")) else "reuse"


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
        execution_attestor: MagpieExecutionAttestor | None = None,
    ) -> None:
        self._receipt = dependency_receipt
        self._magpie_root = dependency_receipt.root("magpie").resolve()
        self._tracelens_root = dependency_receipt.root("tracelens").resolve()
        self._supervisor = supervisor or SubprocessSupervisor()
        self._execution_attestor = (
            execution_attestor or UnavailableMagpieExecutionAttestor()
        )

    @property
    def execution_available(self) -> bool:
        """Whether preflight has a trusted observer that can permit execution."""

        return self._execution_attestor.is_available

    def supports_execution(self, execution_mode: str, lifecycle: str) -> bool:
        supports = getattr(self._execution_attestor, "supports", None)
        return bool(
            self.execution_available
            and (supports(execution_mode, lifecycle) if supports else True)
        )

    def formal_measurement_support(
        self, execution_mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport:
        """Return observer-owned support for quality plus normal measurement."""

        if not self.supports_execution(execution_mode, lifecycle):
            return MagpieFormalMeasurementSupport(
                False,
                "magpie_execution_attestor_unavailable",
                None,
                ("magpie_execution_attestor_unavailable",),
            )
        return self._execution_attestor.formal_measurement_support(
            execution_mode, lifecycle
        )

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

    def _find_report(self, session: object) -> tuple[Path | None, str | None]:
        try:
            location = self._execution_attestor.locate_report(session)
        except Exception as error:
            return (
                None,
                f"magpie_report_location_failed:{type(error).__name__}:{error}",
            )
        return location.path, location.error

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
            quality_kind=str(quality.get("kind", "")),
            evaluator_policy=(
                evaluator_policy
                if isinstance(evaluator_policy, Mapping)
                else None
            ),
            config_sha256=sha256_file(request.config_path),
            gpu_lease_digest=(
                sha256_json(request.gpu_lease)
                if mode in {"local", "ray"} and isinstance(request.gpu_lease, Mapping)
                else None
            ),
            execution_mode=mode,
            lifecycle=_lifecycle(benchmark),
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
            ray_contract=_ray_contract(benchmark),
            evaluator_endpoint_port=_positive_int(
                envs.get("PORT") if isinstance(envs, Mapping) else None,
                default=8888,
            ),
            evaluator_concurrent_requests=_positive_int(
                envs.get("CONC") if isinstance(envs, Mapping) else None,
                default=64,
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
            expected_gpu_lease_digest=expectations.gpu_lease_digest,
            expected_requested_image=expectations.requested_image,
            expected_execution_mode=expectations.execution_mode,
        )

    def _parse_report(
        self,
        request: BenchmarkRequest,
        expectations: _EvidenceExpectations,
        process: ProcessResult,
        report: Path,
        execution_attestation_path: Path,
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
            expected_quality_kind=expectations.quality_kind,
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
            expected_gpu_lease_digest=expectations.gpu_lease_digest,
            expected_requested_image=expectations.requested_image,
            expected_execution_mode=expectations.execution_mode,
            expected_lifecycle=expectations.lifecycle,
            allow_tracelens_derivation=expectations.allow_tracelens_derivation,
            expected_tracelens_commit=expectations.tracelens_commit,
            expected_tracelens_tree=expectations.tracelens_tree,
            execution_attestation_path=execution_attestation_path,
        )

    @staticmethod
    def _not_started() -> ProcessResult:
        return ProcessResult(
            argv=(),
            exit_code=None,
            timed_out=False,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.0,
        )

    def _prepare_attestor(
        self,
        request: BenchmarkRequest,
        expectations: _EvidenceExpectations,
        run_root: Path,
        argv: tuple[str, ...],
    ) -> object:
        return self._execution_attestor.prepare(
            MagpieAttestationRequest(
                run_id=request.run_id,
                pass_type=request.pass_type,
                config_path=request.config_path.resolve(),
                run_root=run_root,
                benchmark_argv=argv,
                config_sha256=expectations.config_sha256,
                execution_mode=expectations.execution_mode,
                lifecycle=expectations.lifecycle,
                requested_image=expectations.requested_image,
                gpu_lease=request.gpu_lease,
                ray_contract=expectations.ray_contract,
                evaluator_policy=expectations.evaluator_policy,
                evaluator_policy_lock=(
                    self._receipt.evaluator_policy.to_dict()
                    if self._receipt.evaluator_policy is not None
                    else None
                ),
                lm_eval_runtime=(
                    expectations.lm_eval_runtime.to_dict()
                    if expectations.lm_eval_runtime is not None
                    else None
                ),
                model=expectations.model,
                evaluator_endpoint_port=expectations.evaluator_endpoint_port,
                evaluator_concurrent_requests=(
                    expectations.evaluator_concurrent_requests
                ),
                evaluator_timeout_seconds=min(request.timeout_seconds, 3600),
            )
        )

    def _execution_inputs(
        self, request: BenchmarkRequest
    ) -> tuple[Path, tuple[str, ...]]:
        run_root = self._run_root(request)
        return run_root, self._benchmark_argv(request, run_root)

    def run_normalized(self, request: BenchmarkRequest) -> NormalizedBenchmarkResult:
        """Run Magpie and return the typed result used by E2E policy."""
        document = validate_resolved_view(
            request.config_path, pass_type=request.pass_type,
            dependency_receipt=self._receipt,
        )
        expectations = self._expectations(request, document)
        if not self._execution_attestor.is_available:
            return self._empty(
                request, expectations, self._not_started(),
                request.output_dir / request.run_id / request.pass_type.value,
                "magpie_execution_attestor_unavailable",
            )
        run_root, argv = self._execution_inputs(request)
        attestor_session, launch_argv, preparation_error = prepare_magpie_execution(
            self._execution_attestor,
            lambda: self._prepare_attestor(request, expectations, run_root, argv),
            argv,
        )
        if preparation_error is not None:
            return self._empty(
                request, expectations, self._not_started(), run_root,
                preparation_error,
            )
        assert attestor_session is not None and launch_argv is not None
        process, launch_error = run_magpie_execution(
            self._execution_attestor,
            self._supervisor,
            attestor_session,
            launch_argv,
            cwd=self._magpie_root,
            environment=self._environment(request.environment),
            timeout_seconds=request.timeout_seconds,
        )
        if process is None:
            assert launch_error is not None
            return self._empty(
                request, expectations, self._not_started(), run_root, launch_error
            )
        report, discovery_error = self._find_report(attestor_session)
        try:
            execution_attestation_path = self._execution_attestor.complete(
                attestor_session,
                report_path=report,
                command_exit_code=process.exit_code,
                timed_out=process.timed_out,
            )
        except Exception as error:
            return self._empty(
                request,
                expectations,
                process,
                report.parent if report else run_root,
                f"magpie_execution_attestor_complete_failed:{type(error).__name__}:{error}",
            )
        if report is None:
            error = discovery_error or "benchmark_report_missing"
            if process.timed_out:
                error = "benchmark_process_timeout"
            elif process.exit_code not in (0, None):
                error = f"benchmark_process_exit_{process.exit_code}"
            return self._empty(request, expectations, process, run_root, error)
        if execution_attestation_path is None:
            return self._empty(
                request, expectations, process, report.parent,
                "magpie_execution_attestation_missing",
            )
        try:
            return self._parse_report(
                request, expectations, process, report, execution_attestation_path,
            )
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
