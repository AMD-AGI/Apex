"""Read-only E2E config, provenance, and resume validation before GPU lease."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.benchmark import (
    BenchmarkConfigViews,
    build_config_views,
    validate_resolved_view,
)
from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_file,
)
from apex.intake import E2EOptimizeSpec
from apex.ports import BenchmarkPass, MagpieFormalMeasurementSupport
from apex.runtime import DependencyReceipt, MagpieConfigContract, RunProvenance

from .recovery import RecoveredRunRequest


class ProvenancePort(Protocol):
    def resolve(
        self,
        resolved: MagpieConfigContract,
        *,
        gpu_arch: str,
        hints: Mapping[str, Any] | None = None,
    ) -> RunProvenance: ...


class MagpieConfigResolutionPort(Protocol):
    def resolve(self, config_path: Path) -> MagpieConfigContract: ...


def benchmark_execution_available(
    benchmark: object, resolved: MagpieConfigContract
) -> bool:
    """Resolve observer support without acquiring a GPU."""

    supports = getattr(benchmark, "supports_execution", None)
    if supports:
        return bool(
            supports(
                str(resolved.plan["identity"]["run_mode"]),
                str(resolved.plan["lifecycle"]),
            )
        )
    return bool(getattr(benchmark, "execution_available", True))


def benchmark_formal_measurement_support(
    benchmark: object, resolved: MagpieConfigContract
) -> MagpieFormalMeasurementSupport:
    """Resolve quality-and-performance authority before acquiring a GPU."""

    identity = resolved.plan["identity"]
    return _formal_measurement_support(
        benchmark,
        str(identity["run_mode"]),
        str(resolved.plan["lifecycle"]),
    )


def require_benchmark_execution_available(
    benchmark: object, resolved: MagpieConfigContract
) -> None:
    if not benchmark_execution_available(benchmark, resolved):
        raise ContractError(
            "Trusted Magpie execution attestor is unavailable",
            "magpie_execution_attestor_unavailable",
        )


def require_formal_measurement_available(
    benchmark: object, resolved: MagpieConfigContract
) -> None:
    support = benchmark_formal_measurement_support(benchmark, resolved)
    if not support.available:
        raise ContractError(
            "Trusted Magpie quality evaluator authority is unavailable",
            support.reason_code or "magpie_formal_measurement_authority_missing",
        )


@dataclass(frozen=True, slots=True)
class E2EPreflightResult:
    """GPU-free config compatibility and adapter-composition receipt."""

    document: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.document)


def resolve_preflight_provenance(
    resolver: ProvenancePort,
    spec: E2EOptimizeSpec,
    resolved: MagpieConfigContract,
) -> RunProvenance:
    return resolver.resolve(
        resolved,
        gpu_arch=spec.gpu_arch,
        hints=spec.deployment_hints,
    )


def resolve_preflight_contract(
    resolver: MagpieConfigResolutionPort, spec: E2EOptimizeSpec
) -> MagpieConfigContract:
    return resolver.resolve(spec.config_path)


def build_preflight_views(
    receipt: DependencyReceipt,
    spec: E2EOptimizeSpec,
    provenance: RunProvenance,
    resolved: MagpieConfigContract,
    staging_root: Path,
) -> BenchmarkConfigViews:
    return build_config_views(
        spec.config_path,
        staging_root / "configs",
        dependency_receipt=receipt,
        resolved_contract=resolved,
        source_repository_roots=tuple(
            Path(lock.path)
            for lock in provenance.component_sources.locks
            if lock.exact
        ),
        model_revision=provenance.model_revision,
        hf_cache_path=hf_cache_path(spec),
        gpu_devices=gpu_devices(spec),
        hf_offline=hf_offline(spec),
    )


def compose_preflight_result(
    receipt: DependencyReceipt,
    views: BenchmarkConfigViews | None,
    provenance: RunProvenance,
    resolved: MagpieConfigContract,
    *,
    benchmark: object,
    deployment: object,
    micro: object,
    final_delivery: object,
) -> E2EPreflightResult:
    identity = resolved.plan["identity"]
    capability = resolved.capability_receipt
    framework, run_mode = str(identity["framework"]), str(identity["run_mode"])
    lifecycle = str(resolved.plan["lifecycle"])
    document = {
        "schema": "apex.e2e-preflight/v2",
        "status": resolved.status,
        "gpu_acquired": False,
        "config": {
            "path": str(resolved.config_path),
            "raw_sha256": resolved.config_sha256,
            "view_status": (
                "materialized" if views is not None else "capability_upgrade_required"
            ),
            "effective_measurement_sha256": (
                sha256_file(views.measurement) if views is not None else None
            ),
            "workload_semantics_sha256": (
                views.workload_semantics_sha256 if views is not None else None
            ),
            "magpie_effective_config_sha256": resolved.plan[
                "effective_config_sha256"
            ],
            "magpie_scoring_config_sha256": resolved.plan[
                "scoring_config_sha256"
            ],
            "magpie_phase_views_sha256": resolved.plan["phase_views_sha256"],
            "magpie_plan_sha256": resolved.plan["plan_sha256"],
            "magpie_capability_receipt_sha256": capability["receipt_sha256"],
            "magpie_resolution_method_sha256": resolved.resolution_method_sha256,
        },
        "dependency": {
            "lock_sha256": receipt.lock_sha256,
            "magpie_commit": receipt.commits["magpie"],
        },
        "dimensions": {
            "framework": framework,
            "run_mode": run_mode,
            "lifecycle": lifecycle,
            "precision": str(identity["precision"]),
            "model_identity_sha256": identity["model_sha256"],
            "source_components": sorted(provenance.active_components),
        },
        "capabilities": _capabilities(
            compatible=resolved.status == "config_compatible",
            run_mode=run_mode,
            lifecycle=lifecycle,
            provenance=provenance,
            benchmark=benchmark,
            deployment=deployment,
            micro=micro,
            final_delivery=final_delivery,
        ),
        "magpie_contract": {
            "optimization_applicable": capability["optimization_applicable"],
            "reward_contract": dict(capability["reward_contract"]),
            "capabilities": dict(capability["capabilities"]),
            "blockers": list(capability["blockers"]),
            "requirements": dict(resolved.plan["requirements"]),
            "source_runtime": dict(resolved.plan["source_runtime"]),
        },
        "provenance": {
            "digest": provenance.digest,
            "status": provenance.status,
            "missing_evidence": list(provenance.missing_evidence),
            "component_source_locks": provenance.component_sources.to_dict(),
        },
    }
    return E2EPreflightResult(document)


def _capabilities(
    *,
    compatible: bool,
    run_mode: str,
    lifecycle: str,
    provenance: RunProvenance,
    benchmark: object,
    deployment: object,
    micro: object,
    final_delivery: object,
) -> dict[str, Any]:
    supported_components = frozenset(
        str(item) for item in getattr(deployment, "supported_components", ())
    )
    supported_modes = frozenset(
        str(item) for item in getattr(deployment, "supported_run_modes", ())
    )
    components = frozenset(provenance.active_components)
    deployment_routing = _capability_receipt(deployment)
    supports = getattr(benchmark, "supports_execution", None)
    execution_available = bool(supports(run_mode, lifecycle)) if supports else bool(
        getattr(benchmark, "execution_available", True)
    )
    formal = _formal_measurement_support(benchmark, run_mode, lifecycle)
    source_status = "ready"
    if not components.issubset(supported_components) or run_mode not in supported_modes:
        source_status = "capability_upgrade_required"
    elif not provenance.source_delivery_ready:
        source_status = "evidence_pending"
    return {
        "benchmark_execution": {
            "adapter_id": "pinned-magpie-benchmark-v1",
            "available": compatible and execution_available,
            "reason": (
                None
                if compatible and execution_available
                else (
                    "magpie_execution_attestor_unavailable"
                    if compatible
                    else "capability_upgrade_required"
                )
            ),
        },
        "formal_measurement": {
            "available": compatible and formal.available,
            "reason": (
                None
                if compatible and formal.available
                else (
                    formal.reason_code
                    if compatible
                    else "capability_upgrade_required"
                )
            ),
            "evaluator_execution_mode": formal.evaluator_execution_mode,
            "blockers": list(formal.blockers),
        },
        "micro_qualification": {
            "adapter_id": type(micro).__name__,
            "mode": str(getattr(micro, "qualification_mode", "unavailable")),
            "routing": _capability_receipt(micro),
        },
        "source_optimization": {
            "adapter_id": str(
                getattr(deployment, "adapter_id", type(deployment).__name__)
            ),
            "status": source_status,
            "supported_components": sorted(supported_components),
            "supported_run_modes": sorted(supported_modes),
            "missing_evidence": list(provenance.missing_evidence),
            "routing": deployment_routing,
        },
        "formal_delivery": {
            "adapter_id": str(
                getattr(final_delivery, "adapter_id", type(final_delivery).__name__)
            ),
            "qualification": "not_claimed",
        },
        "sanitizer_runtime": "not_implemented",
    }


def _formal_measurement_support(
    benchmark: object, run_mode: str, lifecycle: str
) -> MagpieFormalMeasurementSupport:
    support = getattr(benchmark, "formal_measurement_support", None)
    if not callable(support):
        return MagpieFormalMeasurementSupport(
            False,
            "magpie_formal_measurement_authority_missing",
            None,
            ("magpie_formal_measurement_authority_missing",),
        )
    result = support(run_mode, lifecycle)
    if not isinstance(result, MagpieFormalMeasurementSupport):
        raise IntegrityError(
            "Benchmark formal measurement support is invalid",
            "invalid_magpie_formal_measurement_support",
        )
    return result


def _capability_receipt(adapter: object) -> Mapping[str, Any] | None:
    receipt = getattr(adapter, "capability_receipt", None)
    if not callable(receipt):
        return None
    document = receipt()
    if not isinstance(document, Mapping):
        raise IntegrityError(
            "Adapter capability receipt is not a mapping",
            "invalid_adapter_capability_receipt",
        )
    return dict(document)


def write_preflight_result(result: E2EPreflightResult, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.is_absolute() or output_dir.is_symlink() or output_dir.exists():
        raise ContractError(
            "E2E preflight output must be a new absolute directory",
            "results_exist",
        )
    output_dir.mkdir(parents=True)
    path = output_dir / "preflight.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".preflight.", dir=output_dir)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(result.to_dict()) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def validate_resume_preflight(
    resolver: ProvenancePort,
    resolved_plans: MagpieConfigResolutionPort,
    receipt: DependencyReceipt,
    request: RecoveredRunRequest,
    oracle_policy_sha256: str | None,
) -> RunProvenance:
    resolved = resolve_preflight_contract(resolved_plans, request.spec)
    provenance = resolve_preflight_provenance(resolver, request.spec, resolved)
    if provenance.digest != request.provenance_digest:
        raise ContractError("Run provenance changed", "resume_provenance_mismatch")
    if oracle_policy_sha256 != request.correctness_oracle_policy_sha256:
        raise ContractError("Oracle policy changed", "resume_oracle_policy_mismatch")
    for path, pass_type in (
        (request.views.measurement, BenchmarkPass.MEASUREMENT),
        (request.views.diagnostic, BenchmarkPass.DIAGNOSTIC),
        (request.views.replay, BenchmarkPass.MEASUREMENT),
    ):
        validate_resolved_view(
            path,
            pass_type=pass_type,
            dependency_receipt=receipt,
            expected_resolved=resolved,
        )
    return provenance


def hf_cache_path(spec: E2EOptimizeSpec) -> Path | None:
    raw = spec.deployment_hints.get("hf_cache_path")
    if raw is None:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        raise ContractError(
            "deployment_hints.hf_cache_path must be absolute",
            "invalid_hf_cache_path",
        )
    return path


def gpu_devices(spec: E2EOptimizeSpec) -> str | None:
    raw = spec.deployment_hints.get("gpu_devices")
    return str(raw) if raw is not None else None


def hf_offline(spec: E2EOptimizeSpec) -> bool:
    raw = spec.deployment_hints.get("hf_offline", False)
    if not isinstance(raw, bool):
        raise ContractError(
            "deployment_hints.hf_offline must be a boolean",
            "invalid_hf_offline",
        )
    return raw


__all__ = [
    "E2EPreflightResult",
    "ProvenancePort",
    "MagpieConfigResolutionPort",
    "build_preflight_views",
    "benchmark_formal_measurement_support",
    "compose_preflight_result",
    "gpu_devices",
    "resolve_preflight_provenance",
    "resolve_preflight_contract",
    "require_benchmark_execution_available",
    "require_formal_measurement_available",
    "validate_resume_preflight",
    "write_preflight_result",
]
