"""Real primary-build and independent-replay adapters for formal source delivery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.benchmark import NormalizedBenchmarkResult
from apex.core import ContractError, IntegrityError, sha256_file, sha256_json
from apex.delivery import (
    CleanReplayReceipt,
    EngagementRequest,
    LoadedByteEngagementReceipt,
    ReplayArtifactReceipt,
    ReplayRequest,
    SourceBuildReceipt,
    SourceBuildRequest,
)
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EObservation,
    E2EPairedMeasurement,
    E2EPairedWindow,
    evaluate_current_anchor,
    evaluate_no_regression,
    evaluate_paired_current_anchor,
)
from apex.intake import RegressionGates
from apex.ports import BenchmarkPass, BenchmarkRequest

from .benchmarking import BenchmarkAdapter, measurement_from_result
from .overlay_config import derive_overlay_configs
from .source_delivery_models import PrimarySourceBuildOutput, PrimarySourceBuildRequest
from .source_delivery_receipts import (
    PRIMARY_BENCHMARK_SCHEMA,
    acceptance_policy_from_mapping,
    load_primary_benchmark,
    measurement_from_mapping as stored_measurement_from_mapping,
    primary_runtime_identity,
    write_primary_receipt,
)
from .source_image_runtime import SourceImageBuild


class SourceImagePort(Protocol):
    def build(self, *, recipe, repository_roots, source_stack_sha256, output_dir) -> SourceImageBuild: ...

    def engage(
        self, *, bundle_digest, image, source_stack_sha256, artifacts, cwd
    ) -> LoadedByteEngagementReceipt: ...


@dataclass(frozen=True, slots=True)
class FormalMeasurementPolicy:
    acceptance: E2EAcceptancePolicy = E2EAcceptancePolicy(
        RegressionGates(0.0, 5.0, 2.0)
    )
    overlay_parity_noise_pct: float = 1.0


@dataclass(frozen=True, slots=True)
class _PrimaryEvidence:
    build: SourceImageBuild
    engagement: LoadedByteEngagementReceipt
    measurement: E2EObservation
    result: NormalizedBenchmarkResult
    configs: Any
    current_verdict: Any
    parity_verdict: Any
    policy: FormalMeasurementPolicy


class SourceImagePrimaryBuilder:
    """Build and benchmark the first immutable source-baked image."""

    def __init__(
        self,
        images: SourceImagePort,
        benchmark: BenchmarkAdapter,
        policy: FormalMeasurementPolicy | None = None,
    ) -> None:
        self.images = images
        self.benchmark = benchmark
        self.policy = policy or FormalMeasurementPolicy()

    def build_and_validate(
        self, request: PrimarySourceBuildRequest
    ) -> PrimarySourceBuildOutput:
        root = request.artifact_root
        build = self.images.build(
            recipe=request.recipe,
            repository_roots=request.repository_roots,
            source_stack_sha256=request.source_stack_sha256,
            output_dir=root / "image-build",
        )
        primary_binding = sha256_json(
            {"run_id": request.run_id, "source_stack": request.source_stack_sha256}
        )
        engagement = self.images.engage(
            bundle_digest=primary_binding,
            image=build.image,
            source_stack_sha256=request.source_stack_sha256,
            artifacts=build.artifacts,
            cwd=root,
        )
        evidence = self._benchmark_source_image(request, build, engagement, root)
        receipts = _write_primary_receipts(root, request, evidence)
        gates = _primary_gates(evidence)
        environment_id = (
            f"primary-{build.image.image_digest[7:23]}-"
            f"{evidence.measurement.measurement_receipt[:16]}"
        )
        return PrimarySourceBuildOutput(
            environment_id,
            _runtime_identity(evidence.result, build.image.image_digest),
            request.source_stack_sha256,
            build.image,
            build.sbom_path,
            evidence.configs.measurement,
            evidence.configs.diagnostic,
            evidence.configs.replay,
            receipts,
            engagement.verified,
            gates["normal_runtime_measurement"],
            gates["accuracy_passed"],
            gates["latency_gates_passed"],
            gates["objective_improved"],
            gates["overlay_rebuild_parity_passed"],
            False,
        )

    def _benchmark_source_image(
        self,
        request: PrimarySourceBuildRequest,
        build: SourceImageBuild,
        engagement: LoadedByteEngagementReceipt,
        root: Path,
    ) -> _PrimaryEvidence:
        configs = derive_overlay_configs(
            measurement=request.benchmark_measurement,
            diagnostic=request.benchmark_diagnostic,
            replay=request.benchmark_replay,
            output_dir=root / "configs",
            image_id=build.image.image_digest,
            workload_semantics_sha256=request.baseline.protocol_hash,
        )
        result = self.benchmark.run_normalized(
            BenchmarkRequest(
                run_id=f"primary-{request.run_id}",
                config_path=configs.measurement,
                output_dir=root / "benchmarks",
                pass_type=BenchmarkPass.MEASUREMENT,
                timeout_seconds=7200,
            )
        )
        measurement = _measurement(result, request.baseline.protocol_hash)
        policy = FormalMeasurementPolicy(
            request.acceptance_policy,
            self.policy.overlay_parity_noise_pct,
        )
        current = evaluate_current_anchor(request.baseline, measurement, policy.acceptance)
        parity = evaluate_no_regression(
            request.overlay_final,
            measurement,
            policy.acceptance,
            throughput_noise_pct=policy.overlay_parity_noise_pct,
        )
        return _PrimaryEvidence(
            build,
            engagement,
            measurement,
            result,
            configs,
            current,
            parity,
            policy,
        )


class IndependentSourceImageBuild:
    """Rebuild the same source layer from the verifier's fresh worktrees."""

    def __init__(self, images: SourceImagePort) -> None:
        self.images = images

    def build(self, request: SourceBuildRequest) -> SourceBuildReceipt:
        if request.output_dir is None:
            raise ContractError("Verifier build output is missing", "invalid_build_recipe")
        build = self.images.build(
            recipe=request.recipe,
            repository_roots=request.repository_roots,
            source_stack_sha256=request.source_stack_sha256,
            output_dir=request.output_dir,
        )
        return SourceBuildReceipt(
            request.bundle_digest,
            request.recipe.computed_sha256,
            request.expected_image.parent_digest,
            build.image.parent_digest,
            request.expected_image.image_digest,
            build.image.image_digest,
            request.expected_image.sbom_sha256,
            build.image.sbom_sha256,
            request.source_stack_sha256,
            bool(request.repository_receipts)
            and all(item.verified for item in request.repository_receipts),
            build.artifacts,
            build.step_receipts,
        )


class IndependentSourceImageEngagement:
    """Import every changed module from the independently rebuilt image."""

    def __init__(self, images: SourceImagePort, cwd: Path) -> None:
        self.images = images
        self.cwd = cwd

    def verify_loaded_bytes(
        self, request: EngagementRequest
    ) -> LoadedByteEngagementReceipt:
        if request.build_receipt.observed_image_digest != request.expected_image.image_digest:
            raise IntegrityError("Independent image digest changed", "image_identity_mismatch")
        return self.images.engage(
            bundle_digest=request.bundle_digest,
            image=request.expected_image,
            source_stack_sha256=request.source_stack_sha256,
            artifacts=request.build_receipt.artifacts,
            cwd=self.cwd,
        )


class IndependentCleanReplay:
    """Run unchanged Magpie measurement in a second fresh container."""

    def __init__(
        self,
        benchmark: BenchmarkAdapter,
        policy: FormalMeasurementPolicy | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.policy = policy or FormalMeasurementPolicy()

    def replay(self, request: ReplayRequest) -> CleanReplayReceipt:
        if request.output_dir is None or request.primary_receipts is None:
            raise ContractError("Replay evidence roots are missing", "invalid_replay_receipt")
        primary = load_primary_benchmark(
            request.primary_receipts.get("primary_benchmark_receipt")
        )
        first = stored_measurement_from_mapping(primary.get("source_rebuild"))
        acceptance = acceptance_policy_from_mapping(primary.get("acceptance_policy"))
        measurement, results = self._measure_windows(request, acceptance)
        verdict = evaluate_paired_current_anchor(
            measurement, acceptance
        )
        candidates = tuple(
            observation
            for window in measurement.windows
            for observation in (
                window.candidate_forward,
                window.candidate_reverse,
            )
        )
        parity = tuple(
            evaluate_no_regression(
                first,
                candidate,
                acceptance,
                throughput_noise_pct=self.policy.overlay_parity_noise_pct,
            )
            for candidate in candidates
        )
        accuracy = verdict.accuracy_regression_pct <= 0.0 and all(
            item.accuracy_regression_pct <= 0.0 for item in parity
        )
        latency = verdict.reason_code not in {
            "ttft_p99_regression",
            "tpot_p99_regression",
        } and all(_latency_passed(item, item, acceptance) for item in parity)
        all_quality = all(result.quality.passed for result in results)
        all_normal = all(_normal_measurement(result) for result in results)
        runtime_identities = tuple(
            _runtime_identity(
                result,
                request.expected_image.image_digest
                if index % 4 in {1, 2}
                else None,
            )
            for index, result in enumerate(results)
        )
        return CleanReplayReceipt(
            bundle_digest=request.bundle_digest,
            primary_environment_id=request.primary_environment_id,
            replay_environment_id=(
                f"independent-{request.expected_image.image_digest[7:23]}-"
                f"{measurement.digest[:16]}"
            ),
            image_digest=request.expected_image.image_digest,
            replay_config_sha256=sha256_file(request.replay_config),
            benchmark_receipt_sha256=measurement.digest,
            source_stack_sha256=request.source_stack_sha256,
            source_materialization_sha256=sha256_json(
                [item.to_dict() for item in request.repository_receipts]
            ),
            primary_runtime_identity_sha256=primary_runtime_identity(primary),
            replay_runtime_identity_sha256s=runtime_identities,
            normal_runtime_measurement=all_normal,
            quality_passed=all_quality,
            accuracy_passed=accuracy and all_quality,
            latency_gates_passed=latency,
            objective_improved=verdict.keep,
            paired_measurement=measurement.to_dict(),
            paired_verdict=verdict.to_dict(),
            raw_artifacts=_replay_artifacts(request, measurement, results),
        )

    def _measure_windows(
        self, request: ReplayRequest, acceptance: E2EAcceptancePolicy
    ) -> tuple[E2EPairedMeasurement, tuple[NormalizedBenchmarkResult, ...]]:
        observations: list[E2EObservation] = []
        results: list[NormalizedBenchmarkResult] = []
        order = ("anchor", "candidate", "candidate", "anchor")
        slots = ("ab-anchor", "ab-candidate", "ba-candidate", "ba-anchor")
        prefix = f"replay-{request.bundle_digest[:20]}"
        for window in range(acceptance.min_paired_windows):
            for position, side in enumerate(order):
                result = self.benchmark.run_normalized(
                    BenchmarkRequest(
                        run_id=f"{prefix}-{window}-{slots[position]}",
                        config_path=(
                            request.baseline_config
                            if side == "anchor"
                            else request.replay_config
                        ),
                        output_dir=request.output_dir,
                        pass_type=BenchmarkPass.MEASUREMENT,
                        timeout_seconds=7200,
                    )
                )
                results.append(result)
                observations.append(
                    _measurement(result, request.config_receipt.workload_semantics_sha256)
                )
        windows = tuple(
            E2EPairedWindow(
                f"terminal-window-{index}",
                observations[offset],
                observations[offset + 1],
                observations[offset + 2],
                observations[offset + 3],
            )
            for index, offset in enumerate(range(0, len(observations), 4))
        )
        return (
            E2EPairedMeasurement(
                windows,
                acceptance.digest,
                acceptance.min_paired_windows,
            ),
            tuple(results),
        )


def _measurement(result: NormalizedBenchmarkResult, protocol_hash: str) -> E2EObservation:
    if (
        not _normal_measurement(result)
        or result.report_path is None
        or result.report_path.is_symlink()
        or not result.quality.source_paths
    ):
        raise IntegrityError("Formal benchmark failed", "source_delivery_benchmark_failed")
    return measurement_from_result(
        result,
        protocol_hash,
        quality_receipt=sha256_json(
            {"run_id": result.run_id, "sha256": sha256_file(result.quality.source_paths[0])}
        ),
        measurement_receipt=sha256_json(
            {"run_id": result.run_id, "sha256": sha256_file(result.report_path)}
        ),
    )


def _replay_artifacts(
    request: ReplayRequest,
    measurement: E2EPairedMeasurement,
    results: tuple[NormalizedBenchmarkResult, ...],
) -> tuple[ReplayArtifactReceipt, ...]:
    root = request.output_dir
    if root is None:
        raise IntegrityError("Clean replay output root is missing", "invalid_replay_receipt")
    observations = tuple(
        item for window in measurement.windows for item in window.observations
    )
    if len(observations) != len(results):
        raise IntegrityError("Clean replay result count differs", "invalid_replay_receipt")
    artifacts: list[ReplayArtifactReceipt] = []
    for observation, result in zip(observations, results, strict=True):
        report = result.report_path
        quality = result.quality.source_paths[0] if result.quality.source_paths else None
        if report is None or quality is None:
            raise IntegrityError("Clean replay raw evidence is missing", "invalid_replay_receipt")
        if observation.measurement_receipt != sha256_json(
            {"run_id": result.run_id, "sha256": sha256_file(report)}
        ) or observation.quality_receipt != sha256_json(
            {"run_id": result.run_id, "sha256": sha256_file(quality)}
        ):
            raise IntegrityError("Clean replay raw identity differs", "invalid_replay_receipt")
        paths = (
            ("benchmark_report", report),
            ("execution_attestation", _execution_attestation_path(result)),
            *(("quality_result", path) for path in result.quality.source_paths),
            *(
                (
                    "quality_sample"
                    if path.name.startswith("samples")
                    else "quality_raw_artifact",
                    path,
                )
                for path in result.quality.raw_artifact_paths
            ),
        )
        for role, path in paths:
            artifacts.append(
                _replay_artifact(root, result.run_id, observation, role, path)
            )
    return tuple(artifacts)


def _replay_artifact(
    root: Path,
    run_id: str,
    observation: E2EObservation,
    role: str,
    path: Path,
) -> ReplayArtifactReceipt:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise IntegrityError(
            "Clean replay artifact escapes its output root", "invalid_replay_receipt"
        ) from error
    if path.is_symlink() or not resolved.is_file():
        raise IntegrityError("Clean replay artifact is unsafe", "invalid_replay_receipt")
    return ReplayArtifactReceipt(
        role,
        run_id,
        observation.measurement_receipt,
        observation.quality_receipt,
        relative,
        sha256_file(resolved),
        resolved.stat().st_size,
        _media_type(resolved),
    )


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
    }.get(path.suffix.lower(), "application/octet-stream")


def _primary_gates(evidence: _PrimaryEvidence) -> dict[str, bool]:
    policy = evidence.policy.acceptance
    return {
        "normal_runtime_measurement": _normal_measurement(evidence.result),
        "accuracy_passed": _accuracy_passed(
            evidence.current_verdict, evidence.parity_verdict
        )
        and evidence.result.quality.passed,
        "latency_gates_passed": _latency_passed(
            evidence.current_verdict, evidence.parity_verdict, policy
        ),
        "objective_improved": evidence.current_verdict.keep,
        "overlay_rebuild_parity_passed": evidence.parity_verdict.keep,
    }


def _execution_attestation_path(result: NormalizedBenchmarkResult) -> Path:
    matches = tuple(
        path
        for path in result.artifacts
        if path.name == "execution_attestation.json"
    )
    if len(matches) != 1:
        raise IntegrityError(
            "Formal benchmark lacks one protected execution attestation",
            "source_delivery_benchmark_failed",
        )
    return matches[0]


def _runtime_identity(
    result: NormalizedBenchmarkResult, expected_image_id: str | None
) -> str:
    runtime = result.serving_runtime
    attestation = _execution_attestation_path(result)
    if (
        runtime.required is not True
        or runtime.passed is not True
        or not runtime.container_name
        or not runtime.container_spec_sha256
        or not runtime.resolved_image_id
        or expected_image_id is not None
        and runtime.resolved_image_id != expected_image_id
    ):
        raise IntegrityError(
            "Formal benchmark runtime identity is unavailable",
            "source_delivery_runtime_unverified",
        )
    return sha256_json(
        {
            "schema": "apex.e2e-runtime-identity/v1",
            "run_id": result.run_id,
            "container_name": runtime.container_name,
            "container_spec_sha256": runtime.container_spec_sha256,
            "resolved_image_id": runtime.resolved_image_id,
            "execution_attestation_sha256": sha256_file(attestation),
        }
    )


def _normal_measurement(result: NormalizedBenchmarkResult) -> bool:
    return (
        result.succeeded
        and result.pass_type is BenchmarkPass.MEASUREMENT
        and result.run_kind == "measurement"
        and result.reward_eligible
        and not result.profiling_enabled
    )


def _accuracy_passed(current: Any, parity: Any) -> bool:
    return current.accuracy_regression_pct <= 0.0 and parity.accuracy_regression_pct <= 0.0


def _latency_passed(current: Any, parity: Any, policy: E2EAcceptancePolicy) -> bool:
    return all(
        (
            current.ttft_p99_regression_pct <= policy.gates.ttft_p99_regression_pct,
            parity.ttft_p99_regression_pct <= policy.gates.ttft_p99_regression_pct,
            current.tpot_p99_regression_pct <= policy.gates.tpot_p99_regression_pct,
            parity.tpot_p99_regression_pct <= policy.gates.tpot_p99_regression_pct,
        )
    )


def _write_primary_receipts(
    root: Path, request: PrimarySourceBuildRequest, evidence: _PrimaryEvidence
) -> Mapping[str, Path]:
    gates = _primary_gates(evidence)
    documents = {
        "primary_build_receipt": evidence.build.build_document,
        "primary_engagement_receipt": evidence.engagement.to_dict(),
        "primary_benchmark_receipt": {
            "schema": PRIMARY_BENCHMARK_SCHEMA,
            "source_stack_sha256": request.source_stack_sha256,
            "baseline": request.baseline.to_dict(),
            "overlay_final": request.overlay_final.to_dict(),
            "source_rebuild": evidence.measurement.to_dict(),
            "runtime_identity_sha256": _runtime_identity(
                evidence.result, evidence.build.image.image_digest
            ),
            "acceptance_policy": evidence.policy.acceptance.to_dict(),
            "current_verdict": evidence.current_verdict.to_dict(),
            "overlay_parity_verdict": evidence.parity_verdict.to_dict(),
            "gates": gates,
        },
        "primary_safety_receipt": {
            "schema": "apex.advisory-safety/v1",
            "source_stack_sha256": request.source_stack_sha256,
            "status": "advisory_not_certified",
            "safety_certified": False,
        },
    }
    paths = {}
    for role, document in documents.items():
        path = (root / f"{role}.json").resolve()
        write_primary_receipt(path, document)
        paths[role] = path
    return paths


__all__ = [
    "FormalMeasurementPolicy",
    "IndependentCleanReplay",
    "IndependentSourceImageBuild",
    "IndependentSourceImageEngagement",
    "SourceImagePrimaryBuilder",
    "SourceImagePort",
]
