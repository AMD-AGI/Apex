from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from apex.core import ContractError, sha256_json
from apex.mcp import (
    CapabilityRegistry,
    CapabilityScope,
    MagpieAcquisitionHandler,
    planned_capability_descriptors,
)
from apex.ports import (
    BenchmarkPass,
    BenchmarkRequest,
    BenchmarkResult,
    CapabilityAuthority,
    CapabilityRequest,
)


@dataclass(frozen=True)
class _LeaseReceipt:
    run_id: str

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {"schema": "test-gpu-lease", "run_id": self.run_id}


class _Lease:
    def __init__(self, run_id: str) -> None:
        self.receipt = _LeaseReceipt(run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def measurement(self, action_id: str):
        return _MeasurementGuard(self.receipt, action_id)


@dataclass(frozen=True)
class _Bracket:
    run_id: str
    action_id: str
    lease_digest: str

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "action_id": self.action_id,
            "lease_digest": self.lease_digest,
            "fixture": True,
        }


class _MeasurementGuard:
    def __init__(self, lease: _LeaseReceipt, action_id: str) -> None:
        self.receipt = _Bracket(lease.run_id, action_id, lease.digest)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _Leases:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str | None]] = []

    def acquire(self, run_id: str, *, requested_devices: str | None = None):
        self.requests.append((run_id, requested_devices))
        return _Lease(run_id)


class _Benchmark:
    def __init__(self) -> None:
        self.requests: list[BenchmarkRequest] = []

    def run(self, request: BenchmarkRequest) -> BenchmarkResult:
        self.requests.append(request)
        workspace = request.output_dir / request.run_id / request.pass_type.value
        workspace.mkdir(parents=True)
        report = workspace / "benchmark_report.json"
        artifact = workspace / "raw.json"
        report.write_text("{}\n", encoding="utf-8")
        artifact.write_text("{\"raw\": true}\n", encoding="utf-8")
        return BenchmarkResult(
            request.run_id,
            request.pass_type,
            True,
            report,
            workspace,
            {"request_throughput": 10.0},
            (artifact,),
            None,
        )


def _registry(
    tmp_path: Path, capability_id: str, benchmark: _Benchmark, leases: _Leases
) -> tuple[CapabilityRegistry, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "benchmark.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    scope = CapabilityScope(workspace, tmp_path / "results")
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == capability_id
    )
    registry = CapabilityRegistry()
    pass_type = (
        BenchmarkPass.MEASUREMENT
        if capability_id == "benchmark.run"
        else BenchmarkPass.DIAGNOSTIC
    )
    registry.register(
        descriptor,
        MagpieAcquisitionHandler(
            scope, lambda: benchmark, leases, pass_type=pass_type
        ),
    )
    return registry, config


@pytest.mark.parametrize(
    ("capability_id", "mode_name", "mode_value", "pass_type"),
    [
        ("benchmark.run", "benchmark_pass", "measurement", BenchmarkPass.MEASUREMENT),
        ("profile.capture", "profile_mode", "magpie_config", BenchmarkPass.DIAGNOSTIC),
    ],
)
def test_magpie_acquisition_is_scoped_gpu_leased_and_reward_ineligible(
    tmp_path: Path,
    capability_id: str,
    mode_name: str,
    mode_value: str,
    pass_type: BenchmarkPass,
) -> None:
    benchmark, leases = _Benchmark(), _Leases()
    registry, _ = _registry(tmp_path, capability_id, benchmark, leases)

    result = registry.invoke(
        CapabilityRequest(
            capability_id,
            {
                "config_path": "benchmark.yaml",
                mode_name: mode_value,
                "gpu_devices": "4",
            },
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    receipt = result.content["receipt"]
    assert receipt["pass_type"] == pass_type.value
    assert receipt["reward_eligible"] is False
    assert receipt["report"]["root"] == "results"
    assert receipt["artifacts"][0]["root"] == "results"
    assert leases.requests == [(benchmark.requests[0].run_id, "4")]
    assert result.reward_eligible is False
    if pass_type is BenchmarkPass.DIAGNOSTIC:
        assert result.content["artifacts"] == receipt["artifacts"]


def test_magpie_acquisition_rejects_unscoped_config_before_gpu(
    tmp_path: Path,
) -> None:
    benchmark, leases = _Benchmark(), _Leases()
    registry, _ = _registry(tmp_path, "benchmark.run", benchmark, leases)

    with pytest.raises(ContractError) as error:
        registry.invoke(
            CapabilityRequest(
                "benchmark.run",
                {"config_path": "../benchmark.yaml", "benchmark_pass": "measurement"},
                frozenset({CapabilityAuthority.WORKSPACE_USER}),
            )
        )

    assert error.value.reason_code == "unsafe_capability_path"
    assert leases.requests == []
    assert benchmark.requests == []
