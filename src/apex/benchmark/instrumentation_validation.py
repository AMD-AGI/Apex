"""Phase-specific benchmark instrumentation validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError
from apex.ports import BenchmarkPass


def validate_instrumentation(
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
        _validate_measurement(profiler, gap)
    else:
        _validate_diagnostic(profiler, gap, tracelens_root)


def enabled(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "1", "true", "yes", "on"
    }


def disabled(value: Any) -> bool:
    if value is False or value == 0:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "0", "false", "no", "off"
    }


def _validate_measurement(
    profiler: Mapping[str, Any], gap: Mapping[str, Any]
) -> None:
    active = [
        name
        for name, value in profiler.items()
        if isinstance(value, Mapping) and enabled(value.get("enabled"))
    ]
    if active or enabled(gap.get("enabled")):
        raise ConfigurationError(
            f"Measurement view enables instrumentation: {active}",
            "measurement_profiler_enabled",
        )


def _validate_diagnostic(
    profiler: Mapping[str, Any],
    gap: Mapping[str, Any],
    tracelens_root: Path,
) -> None:
    torch = profiler.get("torch_profiler")
    tracelens = profiler.get("tracelens")
    targeted = profiler.get("targeted_trace")
    valid = (
        isinstance(torch, Mapping)
        and enabled(torch.get("enabled"))
        and isinstance(tracelens, Mapping)
        and enabled(tracelens.get("enabled"))
        and Path(str(tracelens.get("tracelens_repo_path", ""))).resolve()
        == tracelens_root
        and isinstance(targeted, Mapping)
        and enabled(targeted.get("enabled"))
        and bool(targeted.get("targets"))
        and enabled(gap.get("enabled"))
    )
    if not valid:
        raise ConfigurationError(
            "Diagnostic view must enable Torch profiler, TraceLens at the pinned "
            "root, TargetedKernelTrace, and gap analysis",
            "invalid_diagnostic_view",
        )


__all__ = ["disabled", "enabled", "validate_instrumentation"]
