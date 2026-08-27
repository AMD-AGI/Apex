from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.bootstrap import build_application
from apex.core import ContractError
from apex.evaluation import load_kernel_measurement_report
from apex.execution import (
    MAGPIE_KERNEL_DIAGNOSTICS_ADAPTER_ID,
    STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID,
    STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256,
    StructuredKernelMeasurementAdapter,
)
from apex.ports import KernelMeasurementRequest
from apex.runtime import DependencyReceipt


class _Supervisor:
    def __init__(self, stdout: str, *, exit_code: int = 0) -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.calls: list[dict[str, object]] = []

    def run(self, argv, **kwargs):
        self.calls.append({"argv": tuple(argv), **kwargs})
        return SimpleNamespace(
            exit_code=self.exit_code,
            timed_out=False,
            stdout=self.stdout,
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            process_containment=SimpleNamespace(namespace_empty_verified=True),
            cleanup_succeeded=True,
        )


def _report() -> dict[str, object]:
    health = {
        "device": "gfx950:0",
        "healthy": True,
        "temperature_c": 45.0,
        "clock_mhz": 2100.0,
    }
    order = (
        "reference",
        "optimized",
        "optimized",
        "reference",
        "optimized",
        "reference",
        "reference",
        "optimized",
    )
    return {
        "schema": "apex.kernel-measurement/v1",
        "policy_id": "kernel_invocation_nearest_rank_v1",
        "sample_unit": "kernel_invocation",
        "quantile_method": "nearest_rank_v1",
        "timer": "hip_event",
        "timer_resolution_ns": 1.0,
        "inner_repeats": 1,
        "measurement_method_sha256": STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256,
        "abba_seed": 17,
        "warmup_samples": 20,
        "cases": [
            {
                "case_id": "fixture-case",
                "blocks": [
                    {
                        "block_id": index,
                        "order_position": index,
                        "implementation": implementation,
                        "samples_ms": [
                            10.0 if implementation == "reference" else 8.0
                        ]
                        * 75,
                        "invalid_sample_counts": {},
                        "gpu_health_before": health,
                        "gpu_health_after": health,
                    }
                    for index, implementation in enumerate(order)
                ],
            }
        ],
    }


def _request(tmp_path: Path) -> KernelMeasurementRequest:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    harness = candidate / "harness.py"
    harness.write_text("# protected runner\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    return KernelMeasurementRequest(
        run_id="run-test",
        attempt_id="attempt-test",
        adapter_id=STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID,
        candidate_root=candidate.resolve(),
        report_path=(private / "raw-report.json").resolve(),
        harness_paths=(harness.resolve(),),
        runner_argv=(sys.executable, "harness.py"),
        runner_cwd=candidate.resolve(),
        runner_env={"APEX_FIXTURE": "1"},
        runner_timeout_seconds=30,
        candidate_source_sha256="a" * 64,
        harness_sha256="b" * 64,
        measurement_method_sha256=STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256,
        measurement_policy_sha256="c" * 64,
    )


def test_adapter_captures_stdout_then_parent_writes_private_report(
    tmp_path: Path,
) -> None:
    supervisor = _Supervisor(json.dumps(_report()))
    adapter = StructuredKernelMeasurementAdapter(supervisor)  # type: ignore[arg-type]
    request = _request(tmp_path)

    output = adapter.measure(request)

    assert output.writer_id == STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID
    assert output.report_path == request.report_path
    artifact = load_kernel_measurement_report(output.report_path)
    assert len(artifact.cases[0].reference.values_ms) == 300
    assert len(artifact.cases[0].optimized.values_ms) == 300
    call = supervisor.calls[0]
    assert call["require_pid_namespace"] is True
    assert str(request.report_path) not in " ".join(call["argv"])  # type: ignore[arg-type]
    assert str(request.report_path) not in json.dumps(call["environment"])


def test_adapter_rejects_non_document_stdout_without_publishing(
    tmp_path: Path,
) -> None:
    supervisor = _Supervisor(json.dumps(_report()) + "\ncandidate log")
    adapter = StructuredKernelMeasurementAdapter(supervisor)  # type: ignore[arg-type]
    request = _request(tmp_path)

    with pytest.raises(ContractError) as raised:
        adapter.measure(request)

    assert raised.value.reason_code == "invalid_measurement_runner_output"
    assert not request.report_path.exists()


def test_production_bootstrap_injects_structured_measurement_adapter() -> None:
    application = build_application(knowledge_enabled=False)

    assert (
        application.kernel_optimizer.measurement_adapter_id
        == STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID
    )


def test_production_bootstrap_can_inject_pinned_magpie_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    magpie = tmp_path / "Magpie"
    magpie.mkdir()
    receipt = DependencyReceipt(
        schema="apex.dependency-receipt.v1",
        lock_sha256="a" * 64,
        python=Path(sys.executable),
        roots={"magpie": magpie},
        commits={"magpie": "b" * 40},
        raw={},
    )
    monkeypatch.setattr("apex.bootstrap.verify_runtime_dependencies", lambda: receipt)

    application = build_application(
        knowledge_enabled=False,
    )

    assert (
        application.kernel_optimizer.diagnostics_adapter_id
        == MAGPIE_KERNEL_DIAGNOSTICS_ADAPTER_ID
    )
