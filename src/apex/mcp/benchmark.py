"""GPU-leased Magpie acquisition capabilities with no grading authority."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from apex.core import ContractError, new_identifier, sha256_file
from apex.ports import (
    BenchmarkPass,
    BenchmarkPort,
    BenchmarkRequest,
    CapabilityRequest,
    CapabilityResult,
)
from apex.runtime import GpuLeaseManager, require_gpu_measurement_guard

from .scope import CapabilityScope
from .grants import granted_gpu_selector


class MagpieAcquisitionHandler:
    """Run one explicit normal or diagnostic pass under a physical GPU lease."""

    def __init__(
        self,
        scope: CapabilityScope,
        adapter: Callable[[], BenchmarkPort],
        gpu_leases: GpuLeaseManager,
        *,
        pass_type: BenchmarkPass,
    ) -> None:
        self._scope = scope
        self._adapter = adapter
        self._gpu_leases = gpu_leases
        self._pass_type = pass_type

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        self._validate_mode(request)
        config = self._scope.read_workspace(str(request.arguments["config_path"]))
        if not config.is_file():
            raise ContractError(
                "Magpie config is not a regular file",
                "invalid_capability_arguments",
            )
        run_id = new_identifier("benchmark")
        output = self._scope.claim_output("benchmark-runs", run_id)
        devices = granted_gpu_selector(request)
        with self._gpu_leases.acquire(
            run_id,
            requested_devices=devices,
        ) as lease:
            benchmark_request = BenchmarkRequest(
                run_id=run_id,
                config_path=config,
                output_dir=output.parent,
                pass_type=self._pass_type,
                timeout_seconds=7200,
                gpu_lease=lease.receipt.to_dict(),
            )
            bracket_receipt = None
            if self._pass_type is BenchmarkPass.MEASUREMENT:
                with require_gpu_measurement_guard(lease, run_id) as bracket:
                    result = self._adapter().run(benchmark_request)
                bracket_receipt = bracket.receipt
            else:
                result = self._adapter().run(benchmark_request)
            lease_receipt = lease.receipt
        receipt, artifacts = self._project_result(
            result, lease_receipt, bracket_receipt
        )
        content = {"receipt": receipt}
        if self._pass_type is BenchmarkPass.DIAGNOSTIC:
            content["artifacts"] = artifacts
        return CapabilityResult(
            request.capability_id,
            content,
            artifact_receipts=tuple(artifacts),
            reward_eligible=False,
        )

    def _validate_mode(self, request: CapabilityRequest) -> None:
        if self._pass_type is BenchmarkPass.MEASUREMENT:
            if request.arguments.get("benchmark_pass") != "measurement":
                raise ContractError(
                    "benchmark.run only acquires normal-runtime evidence",
                    "invalid_capability_arguments",
                )
            return
        if request.arguments.get("profile_mode") != "magpie_config":
            raise ContractError(
                "profile.capture only supports the profiler declared by the Magpie config",
                "invalid_capability_arguments",
            )

    def _project_result(
        self, result, lease, bracket
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        if result.pass_type is not self._pass_type:
            raise ContractError(
                "Benchmark adapter returned another pass type",
                "capability_result_mismatch",
            )
        artifacts = [_artifact(self._scope, path) for path in result.artifact_paths]
        report = (
            _artifact(self._scope, result.report_path)
            if result.report_path is not None
            else None
        )
        receipt = {
            "schema": "apex.mcp-magpie-acquisition/v1",
            "run_id": result.run_id,
            "pass_type": result.pass_type.value,
            "succeeded": result.succeeded,
            "error": result.error,
            "metrics": dict(result.metrics),
            "workspace": _locator(self._scope, result.workspace_path),
            "report": report,
            "artifacts": artifacts,
            "gpu_lease": {
                "digest": lease.digest,
                "receipt": lease.to_dict(),
            },
            "gpu_measurement_bracket": (
                {"digest": bracket.digest, "receipt": bracket.to_dict()}
                if bracket is not None
                else None
            ),
            "reward_eligible": False,
        }
        return receipt, artifacts


def _artifact(scope: CapabilityScope, path: Path) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError(
            "Benchmark artifact is not a safe regular file",
            "capability_result_mismatch",
        )
    root, relative = scope.locator(resolved)
    return {
        "root": root,
        "relative_path": relative,
        "sha256": sha256_file(resolved),
        "size": resolved.stat().st_size,
    }


def _locator(scope: CapabilityScope, path: Path) -> dict[str, str]:
    root, relative = scope.locator(Path(path))
    return {"root": root, "relative_path": relative}


__all__ = ["MagpieAcquisitionHandler"]
