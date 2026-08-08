from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from apex.benchmark import (
    InferenceXRuntimeEvidence,
    LatencyDistribution,
    LatencyMetrics,
    ModelRevisionEvidence,
    NormalizedBenchmarkResult,
    QualityEvidence,
    QualityMetric,
    ThroughputMetrics,
)
from apex.core import ConfigurationError, IntegrityError, TaskStatus
from apex.diagnostics import (
    AcquisitionCoverage,
    EvidenceArtifacts,
    KernelEvidence,
    KernelVolume,
    OperationEvidence,
    PerformanceModelEvidence,
    ShapeEvidence,
    TraceEvidence,
    derive_candidate_id,
)
from apex.intake import E2EOptimizeSpec
from apex.optimization.e2e import E2EOptimizeUseCase
from apex.orchestration import RunController, RunPhase
from apex.ports import BenchmarkPass, DiagnosticsResult
from apex.runtime import (
    ContainerIdentity,
    DependencyReceipt,
    GpuDeviceIdentity,
    GpuOwnershipReceipt,
    GpuSelectorRequest,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    LmEvalRuntimeReceipt,
    LocalGpuLeaseManager,
    RsmiDeviceIdentity,
    RunProvenance,
)
from apex.storage import EventJournal, SnapshotStore


class _FakeOwnershipInspector:
    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        unique_id = "GPU-0000000000000001"
        device = GpuDeviceIdentity(0, 2, 0, unique_id, "/dev/dri/renderD128")
        return GpuOwnershipReceipt(
            schema_version=2,
            policy_id="clean_hsa_kfd_rsmi_process_gpu_map_v2",
            selector_inputs=GpuSelectorRequest(requested=("0",)),
            observed_unix_ns=123,
            library_path="/opt/rocm/lib/librocm_smi64.so.7",
            library_sha256="a" * 64,
            topology_root="/sys/class/kfd/kfd/topology/nodes",
            hsa_inventory=HsaInventoryEvidence(
                1,
                "clean_unfiltered_hsa_gpu_inventory_v1",
                "/trusted/helper.py",
                "b" * 64,
                "/opt/rocm/lib/libhsa-runtime64.so.1",
                "c" * 64,
                (HsaGpuIdentity(0, 2, 2, 100, 0, unique_id),),
            ),
            rsmi_monitor_inventory=(
                RsmiDeviceIdentity(0, 2, 100, unique_id, 128),
            ),
            device_inventory=(device,),
            selected_devices=(device,),
            allowed_owners=(),
            foreign_owners=(),
        )


def _gpu_leases(tmp_path: Path) -> LocalGpuLeaseManager:
    return LocalGpuLeaseManager(
        lock_root=tmp_path / "gpu-leases",
        ownership_inspector=_FakeOwnershipInspector(),
    )


class FakeBenchmark:
    def __init__(self, final_throughput: float = 99.5) -> None:
        self.calls = 0
        self.final_throughput = final_throughput

    def run_normalized(self, request):
        self.calls += 1
        workspace = request.output_dir / f"fake-{self.calls}"
        workspace.mkdir(parents=True)
        report = workspace / "benchmark_report.json"
        report.write_text("{}", encoding="utf-8")
        quality_path = workspace / "results.json"
        quality_path.write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.8}}}),
            encoding="utf-8",
        )
        throughput = 100.0 if self.calls < 3 else self.final_throughput
        empty = LatencyDistribution(1.0, 1.0, 1.0, 0.0)
        return NormalizedBenchmarkResult(
            schema_version=1,
            run_id=request.run_id,
            pass_type=request.pass_type,
            succeeded=True,
            framework="vllm",
            model="Qwen/example",
            workspace_path=workspace,
            report_path=report,
            throughput=ThroughputMetrics(1.0, throughput, throughput, 100, 1.0),
            latency=LatencyMetrics(empty, empty, empty, empty),
            quality=QualityEvidence(
                True,
                "lm_eval",
                True,
                (QualityMetric("gsm8k", "exact_match,strict-match", 0.8, True),),
                (quality_path,),
            ),
            profiling_enabled=request.pass_type is BenchmarkPass.DIAGNOSTIC,
            run_kind=request.pass_type.value,
            reward_eligible=request.pass_type is BenchmarkPass.MEASUREMENT,
            model_revision=ModelRevisionEvidence(
                False, True, None, None, None
            ),
            inferencex_runtime=InferenceXRuntimeEvidence(
                False, True, None, None, None, None, None
            ),
            artifacts=(report, quality_path),
            errors=(),
            command_exit_code=0,
        )


class FakeDiagnostics:
    def analyze(self, request):
        kernel = KernelEvidence("unknown_kernel", "triton", "aiter")
        shape = ShapeEvidence(concrete_inputs=("[16, 128]",))
        candidate = derive_candidate_id(
            provenance_hash=request.provenance_hash,
            phase="unknown",
            rank=0,
            kernel=kernel,
            shape=shape,
        )
        evidence = TraceEvidence(
            1,
            candidate,
            request.provenance_hash,
            "unknown",
            0,
            OperationEvidence("unknown", "unknown_kernel"),
            kernel,
            shape,
            KernelVolume(10, 10.0, 10.0),
            PerformanceModelEvidence(),
            EvidenceArtifacts(
                "torch_profiler_summary",
                AcquisitionCoverage(0, 0, 0, 0),
                (),
                "c" * 64,
                ("targeted_launch_metadata_unavailable",),
            ),
        )
        request.output_dir.mkdir(parents=True)
        output = request.output_dir / "trace_evidence.json"
        output.write_text(
            json.dumps({"schema_version": 1, "records": [evidence.to_dict()]}),
            encoding="utf-8",
        )
        return DiagnosticsResult(
            request.run_id,
            True,
            (output,),
            {"record_count": 1, "evidence_path": str(output)},
        )


class CrashDiagnostics:
    def analyze(self, request):
        raise RuntimeError("simulated process loss after diagnostic benchmark")


class FakeProvenance:
    def resolve(self, config_path, *, gpu_arch, hints=None):
        return RunProvenance(
            1,
            str(config_path),
            "a" * 64,
            "vllm",
            "Qwen/example",
            None,
            gpu_arch,
            ContainerIdentity("example:v1", "sha256:" + "d" * 64, (), ()),
            ("vllm", "aiter"),
            (),
            "partial",
            ("model_revision", "source_lock:vllm", "source_lock:aiter"),
        )


def _receipt(tmp_path: Path) -> DependencyReceipt:
    magpie = tmp_path / "Magpie"
    tracelens = tmp_path / "TraceLens"
    inferencex = tmp_path / "InferenceX"
    magpie.mkdir()
    tracelens.mkdir()
    inferencex.mkdir()
    runtime = tmp_path / "lm-eval-runtime"
    runtime.mkdir()
    return DependencyReceipt(
        "apex.dependency-receipt.v1",
        "e" * 64,
        Path("/verified/python"),
        {"magpie": magpie, "tracelens": tracelens, "inferencex": inferencex},
        {"magpie": "1" * 40, "tracelens": "2" * 40, "inferencex": "3" * 40},
        {},
        LmEvalRuntimeReceipt(
            runtime, "4" * 64, "5" * 64, {"lm_eval_commit": "6" * 40}, 1, "7" * 64
        ),
    )


def _spec(tmp_path: Path, results: Path) -> E2EOptimizeSpec:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """benchmark:
  framework: vllm
  model: Qwen/example
  envs: {TP: 1, RUN_EVAL: true}
  docker_image: example:v1
""",
        encoding="utf-8",
    )
    return E2EOptimizeSpec.from_mapping(
        {"config_path": str(config), "results_dir": str(results), "max_kernels": 2}
    )


def test_e2e_vertical_slice_records_trace_and_no_regression(tmp_path: Path) -> None:
    results = tmp_path / "run"
    use_case = E2EOptimizeUseCase(
        dependency_receipt=_receipt(tmp_path),
        benchmark=FakeBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        gpu_leases=_gpu_leases(tmp_path),
    )
    result = use_case.run(_spec(tmp_path, results))
    assert result.status is TaskStatus.NO_GAIN
    assert result.intake_provenance_status == "partial"
    assert result.formal_delivery_verified is False
    assert result.no_regression is True
    assert result.opportunity_count == 1
    assert result.eligible_opportunity_count == 0
    assert Path(result.diagnostic_evidence).is_file()
    assert (results / "result.json").is_file()
    for projection in (
        "report.json",
        "report.md",
        "replication_guide.json",
        "replication_guide.md",
    ):
        assert (results / projection).is_file()
    report = json.loads((results / "report.json").read_text(encoding="utf-8"))
    assert report["run_id"] == result.run_id
    measured = report["headline_measured_results"]
    assert len(measured) == 2
    assert all("total_token_throughput" in item["metrics"] for item in measured)
    assert report["provenance"]["framework"] == "vllm"
    recovered = RunController.recover(
        result.run_id,
        EventJournal(results / "events" / "run.db"),
        SnapshotStore(results / "state.snapshot.json"),
    )
    assert recovered.state.stop_reason == "no_gain"
    assert recovered.state.e2e is not None
    assert recovered.state.e2e.stage.value == "completed"
    terminal = EventJournal(results / "events" / "run.db").get_by_idempotency_key(
        result.run_id, "e2e.terminal_result"
    )
    assert terminal is not None
    assert terminal.payload["artifacts"][0]["role"] == "e2e_terminal_result"
    assert use_case.resume(results).to_dict() == result.to_dict()


def test_intake_config_failure_does_not_leave_a_running_run(tmp_path: Path) -> None:
    results = tmp_path / "invalid-intake-run"
    receipt = replace(_receipt(tmp_path), lm_eval_runtime=None)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        gpu_leases=_gpu_leases(tmp_path),
    )

    with pytest.raises(ConfigurationError) as failure:
        use_case.run(_spec(tmp_path, results))

    assert failure.value.reason_code == "lm_eval_runtime_missing"
    assert not results.exists()
    assert not tuple(tmp_path.glob(".apex-e2e-configs-*"))


def test_resume_recovers_completed_baseline_and_retries_diagnostic(tmp_path: Path) -> None:
    results = tmp_path / "resume-run"
    benchmark = FakeBenchmark()
    receipt = _receipt(tmp_path)
    interrupted = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=benchmark,
        diagnostics=CrashDiagnostics(),
        provenance=FakeProvenance(),
        gpu_leases=_gpu_leases(tmp_path),
    )
    with pytest.raises(RuntimeError, match="simulated process loss"):
        interrupted.run(_spec(tmp_path, results))

    request = json.loads((results / "run.request.json").read_text(encoding="utf-8"))
    assert request["schema"] == "apex.e2e-run-request/v1"
    assert request["spec"]["results_dir"] == str(results)
    assert (results / "action_receipts/baseline-measurement.json").is_file()
    assert (results / "action_receipts/diagnostic-0.json").is_file()

    resumed = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=benchmark,
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        gpu_leases=_gpu_leases(tmp_path),
    ).resume(results)

    assert resumed.status is TaskStatus.NO_GAIN
    assert resumed.no_regression is True
    assert resumed.opportunity_count == 1
    assert (results / "result.json").is_file()
    state = RunController.recover(
        resumed.run_id,
        EventJournal(results / "events/run.db"),
        SnapshotStore(results / "state.snapshot.json"),
    ).state
    assert state.e2e is not None
    assert state.e2e.stage.value == "completed"
    events = EventJournal(results / "events/run.db").iter_events(resumed.run_id)
    plan = next(
        event for event in events
        if event.event_type == "tool_result"
        and event.payload.get("tool") == "kernel_opportunity_planner"
    )
    assert plan.payload["opportunity_count"] == 1


def test_resume_rejects_mutated_run_request_projection(tmp_path: Path) -> None:
    results = tmp_path / "tampered-resume-run"
    receipt = _receipt(tmp_path)
    interrupted = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(),
        diagnostics=CrashDiagnostics(),
        provenance=FakeProvenance(),
        gpu_leases=_gpu_leases(tmp_path),
    )
    with pytest.raises(RuntimeError, match="simulated process loss"):
        interrupted.run(_spec(tmp_path, results))

    request_path = results / "run.request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["spec"]["max_iterations"] = 999
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(IntegrityError) as failure:
        E2EOptimizeUseCase(
            dependency_receipt=receipt,
            benchmark=FakeBenchmark(),
            diagnostics=FakeDiagnostics(),
            provenance=FakeProvenance(),
            gpu_leases=_gpu_leases(tmp_path),
        ).resume(results)
    assert failure.value.reason_code == "run_request_projection_mismatch"


def test_terminal_resume_rejects_unbound_result_projection(tmp_path: Path) -> None:
    results = tmp_path / "tampered-terminal-run"
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        gpu_leases=_gpu_leases(tmp_path),
    )
    use_case.run(_spec(tmp_path, results))
    result_path = results / "result.json"
    value = json.loads(result_path.read_text(encoding="utf-8"))
    value["final_metrics"]["total_tokens_per_second"] = 999999.0
    value["details"]["tampered"] = True
    result_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(IntegrityError) as failure:
        use_case.resume(results)
    assert failure.value.reason_code == "e2e_result_projection_mismatch"


def test_e2e_no_winner_final_replay_drift_remains_observed_evidence(
    tmp_path: Path,
) -> None:
    results = tmp_path / "run-regression"
    result = E2EOptimizeUseCase(
        dependency_receipt=_receipt(tmp_path),
        benchmark=FakeBenchmark(final_throughput=98.0),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        gpu_leases=_gpu_leases(tmp_path),
    ).run(_spec(tmp_path, results))
    assert result.status is TaskStatus.NO_GAIN
    assert result.no_regression is True
    assert result.reason_code == "no_opportunities"
    assert result.accepted_patch_ids == ()
    assert result.formal_delivery_verified is False
    assert result.details["observed_replay_verdict"]["keep"] is False
    assert (
        result.details["observed_replay_verdict"]["reason_code"]
        == "insufficient_throughput_gain"
    )
    assert result.details["search_exit_reason"] == "no_opportunities"
    terminal = result.details["terminal_diagnostics"]
    assert terminal["diagnostic_succeeded"] is True
    assert terminal["comparison"]["reward_eligible"] is False
    events = EventJournal(results / "events" / "run.db").iter_events(result.run_id)
    committed = next(event for event in events if event.event_type == "e2e.final_committed")
    lineage_digest = committed.payload["receipt"]
    lineage = json.loads(
        (
            results
            / "artifacts"
            / "sha256"
            / lineage_digest[:2]
            / lineage_digest
        ).read_text(encoding="utf-8")
    )
    assert len(lineage["final_benchmark_receipt"]) == 64
    assert lineage["observed_replay_verdict"] == result.details[
        "observed_replay_verdict"
    ]
    basis = result.details["final_replay_basis"]
    assert basis == {
        "basis": "no_accepted_or_delivered_source_patch",
        "source_identity_unchanged": True,
        "accepted_candidate_count": 0,
        "delivery_attempted": False,
        "formal_delivery_verified": False,
        "final_clean_replay_verified": False,
    }
    state = RunController.recover(
        result.run_id,
        EventJournal(results / "events" / "run.db"),
        SnapshotStore(results / "state.snapshot.json"),
    ).state
    assert state.e2e is not None
    assert state.e2e.final_clean_replay_verified is False
    assert state.phase is RunPhase.SUCCEEDED
    assert state.stop_reason == "no_gain"
