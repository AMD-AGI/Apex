from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from apex.benchmark.execution_attestor_registry import (
    MagpieExecutionAttestorRegistry,
)
from apex.core import ContractError
from apex.ports import (
    BenchmarkPass,
    MagpieAttestationRequest,
    MagpieFormalMeasurementSupport,
    MagpieReportLocation,
)


@dataclass
class _Attestor:
    mode: str
    lifecycle: str
    is_available: bool = True
    aborted: str | None = None

    def supports(self, mode: str, lifecycle: str) -> bool:
        return (mode, lifecycle) == (self.mode, self.lifecycle)

    def formal_measurement_support(
        self, mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport:
        if not self.supports(mode, lifecycle):
            return MagpieFormalMeasurementSupport(
                False, "unsupported", None, ("unsupported",)
            )
        return MagpieFormalMeasurementSupport(True, None, "test")

    def prepare(self, request: MagpieAttestationRequest) -> object:
        return {"run_id": request.run_id, "mode": self.mode}

    def launch_argv(self, session: object) -> tuple[str, ...]:
        assert session == {"run_id": "run", "mode": self.mode}
        return ("python", "-m", "Magpie")

    def abort(self, session: object, *, reason: str) -> None:
        assert session == {"run_id": "run", "mode": self.mode}
        self.aborted = reason

    def locate_report(self, session: object) -> MagpieReportLocation:
        assert session == {"run_id": "run", "mode": self.mode}
        return MagpieReportLocation(None, "benchmark_report_missing")

    def complete(self, session: object, **kwargs) -> Path | None:
        del kwargs
        assert session == {"run_id": "run", "mode": self.mode}
        return Path(f"/{self.mode}-{self.lifecycle}.json")


def _request(mode: str = "docker", lifecycle: str = "one_shot"):
    return MagpieAttestationRequest(
        run_id="run",
        pass_type=BenchmarkPass.MEASUREMENT,
        config_path=Path("/config.yaml"),
        run_root=Path("/results"),
        benchmark_argv=("python", "-m", "Magpie"),
        config_sha256="1" * 64,
        execution_mode=mode,
        lifecycle=lifecycle,
        requested_image=None,
        gpu_lease=None,
    )


def test_routes_prepare_and_complete_to_exact_lane() -> None:
    local = _Attestor("local", "reuse")
    registry = MagpieExecutionAttestorRegistry(
        (_Attestor("docker", "one_shot"), local)
    )

    session = registry.prepare(_request("local", "reuse"))

    assert registry.supports("local", "reuse")
    assert registry.launch_argv(session) == ("python", "-m", "Magpie")
    assert registry.formal_measurement_support("local", "reuse").available
    assert registry.complete(
        session, report_path=None, command_exit_code=0, timed_out=False
    ) == Path("/local-reuse.json")
    registry.abort(session, reason="launch_failed")
    assert local.aborted == "launch_failed"


def test_rejects_unsupported_lane_without_fallback() -> None:
    registry = MagpieExecutionAttestorRegistry((_Attestor("docker", "one_shot"),))

    with pytest.raises(ContractError) as caught:
        registry.prepare(_request("ray", "one_shot"))

    assert caught.value.reason_code == "magpie_execution_attestor_unavailable"
    support = registry.formal_measurement_support("ray", "one_shot")
    assert not support.available
    assert support.reason_code == "magpie_execution_attestor_unavailable"


def test_rejects_ambiguous_lane() -> None:
    registry = MagpieExecutionAttestorRegistry(
        (_Attestor("docker", "one_shot"), _Attestor("docker", "one_shot"))
    )

    assert not registry.supports("docker", "one_shot")
    with pytest.raises(ContractError) as caught:
        registry.prepare(_request())

    assert caught.value.reason_code == "ambiguous_magpie_execution_attestor"


def test_ignores_unavailable_attestor() -> None:
    registry = MagpieExecutionAttestorRegistry(
        (_Attestor("docker", "one_shot", False), _Attestor("local", "reuse"))
    )

    assert registry.is_available
    assert not registry.supports("docker", "one_shot")
