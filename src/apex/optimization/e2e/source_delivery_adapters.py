"""Real primary-build and independent-replay adapters for formal source delivery."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.benchmark import NormalizedBenchmarkResult
from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_file, sha256_json
from apex.delivery import (
    CleanReplayReceipt,
    EngagementRequest,
    LoadedByteEngagementReceipt,
    ReplayRequest,
    SourceBuildReceipt,
    SourceBuildRequest,
)
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EMeasurement,
    evaluate_current_anchor,
    evaluate_no_regression,
)
from apex.intake import RegressionGates
from apex.ports import BenchmarkPass, BenchmarkRequest

from .benchmarking import BenchmarkAdapter, measurement_from_result
from .overlay_config import derive_overlay_configs
from .source_delivery_models import PrimarySourceBuildOutput, PrimarySourceBuildRequest
from .source_image_runtime import SourceImageBuild


_PRIMARY_BENCHMARK_SCHEMA = "apex.qwen-primary-benchmark/v1"


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
    measurement: E2EMeasurement
    result: NormalizedBenchmarkResult
    configs: Any
    current_verdict: Any
    parity_verdict: Any
    policy: FormalMeasurementPolicy


class QwenPrimarySourceBuilder:
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
        current = evaluate_current_anchor(
            request.baseline, measurement, self.policy.acceptance
        )
        parity = evaluate_no_regression(
            request.overlay_final,
            measurement,
            self.policy.acceptance,
            throughput_noise_pct=self.policy.overlay_parity_noise_pct,
        )
        return _PrimaryEvidence(
            build,
            engagement,
            measurement,
            result,
            configs,
            current,
            parity,
            self.policy,
        )


class QwenIndependentSourceBuild:
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


class QwenIndependentEngagement:
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


class QwenIndependentReplay:
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
        primary = _load_primary_benchmark(
            request.primary_receipts.get("primary_benchmark_receipt")
        )
        baseline = _measurement_from_mapping(primary.get("baseline"))
        first = _measurement_from_mapping(primary.get("source_rebuild"))
        result = self.benchmark.run_normalized(
            BenchmarkRequest(
                run_id=f"replay-{request.bundle_digest[:20]}",
                config_path=request.replay_config,
                output_dir=request.output_dir,
                pass_type=BenchmarkPass.MEASUREMENT,
                timeout_seconds=7200,
            )
        )
        replay = _measurement(result, request.config_receipt.workload_semantics_sha256)
        current = evaluate_current_anchor(baseline, replay, self.policy.acceptance)
        parity = evaluate_no_regression(
            first,
            replay,
            self.policy.acceptance,
            throughput_noise_pct=self.policy.overlay_parity_noise_pct,
        )
        accuracy = _accuracy_passed(current, parity) and result.quality.passed
        latency = _latency_passed(current, parity, self.policy.acceptance)
        return CleanReplayReceipt(
            request.bundle_digest,
            request.primary_environment_id,
            f"independent-{request.expected_image.image_digest[7:23]}-"
            f"{replay.measurement_receipt[:16]}",
            request.expected_image.image_digest,
            sha256_file(request.replay_config),
            replay.measurement_receipt,
            request.source_stack_sha256,
            bool(request.repository_receipts)
            and all(item.verified for item in request.repository_receipts),
            True,
            _normal_measurement(result),
            result.quality.passed,
            accuracy,
            latency,
            current.keep,
        )


def _measurement(result: NormalizedBenchmarkResult, protocol_hash: str) -> E2EMeasurement:
    if (
        not _normal_measurement(result)
        or result.report_path is None
        or result.report_path.is_symlink()
    ):
        raise IntegrityError("Formal benchmark failed", "source_delivery_benchmark_failed")
    receipt = sha256_file(result.report_path)
    return measurement_from_result(result, protocol_hash, receipt)


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
            "schema": _PRIMARY_BENCHMARK_SCHEMA,
            "source_stack_sha256": request.source_stack_sha256,
            "baseline": request.baseline.to_dict(),
            "overlay_final": request.overlay_final.to_dict(),
            "source_rebuild": evidence.measurement.to_dict(),
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
        _write_json(path, document)
        paths[role] = path
    return paths


def _load_primary_benchmark(path: Path | None) -> Mapping[str, Any]:
    if path is None or path.is_symlink() or not path.is_file():
        raise IntegrityError("Primary benchmark receipt is missing", "missing_primary_receipt")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError("Primary benchmark receipt is invalid", "missing_primary_receipt") from error
    if not isinstance(value, Mapping) or value.get("schema") != _PRIMARY_BENCHMARK_SCHEMA:
        raise IntegrityError("Primary benchmark schema is invalid", "missing_primary_receipt")
    return value


def _measurement_from_mapping(value: object) -> E2EMeasurement:
    if not isinstance(value, Mapping):
        raise IntegrityError("Stored E2E measurement is invalid", "missing_primary_receipt")
    try:
        return E2EMeasurement(**dict(value))
    except (TypeError, ValueError) as error:
        raise IntegrityError("Stored E2E measurement is invalid", "missing_primary_receipt") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError("Primary receipt already exists", "immutable_delivery_artifact")
    with path.open("xb") as output:
        output.write(canonical_json_bytes(value) + b"\n")
        output.flush()
        os.fsync(output.fileno())


__all__ = [
    "FormalMeasurementPolicy",
    "QwenIndependentEngagement",
    "QwenIndependentReplay",
    "QwenIndependentSourceBuild",
    "QwenPrimarySourceBuilder",
    "SourceImagePort",
]
