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
from apex.core import ConfigurationError, ContractError, IntegrityError, TaskStatus
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
from apex.optimization.e2e import E2EOptimizeUseCase, write_preflight_result
from apex.orchestration import RunController, RunPhase
from apex.ports import (
    BenchmarkPass,
    DiagnosticsResult,
    MagpieFormalMeasurementSupport,
)
from apex.runtime import (
    ComponentSourceLockSet,
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
from tests.support.magpie_contract import ResolvedPlanStub
from tests.support.gpu_evidence import StaticGpuDoctorInspector


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


class _ForbiddenLeaseManager:
    def acquire(self, *_args, **_kwargs):
        raise AssertionError("GPU lease must not be requested before preflight passes")


def _gpu_leases(tmp_path: Path) -> LocalGpuLeaseManager:
    return LocalGpuLeaseManager(
        lock_root=tmp_path / "gpu-leases",
        doctor_inspector=StaticGpuDoctorInspector(_FakeOwnershipInspector()),
    )


class FakeBenchmark:
    def __init__(self, final_throughput: float = 99.5) -> None:
        self.calls = 0
        self.final_throughput = final_throughput

    def formal_measurement_support(
        self, execution_mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport:
        del execution_mode, lifecycle
        return MagpieFormalMeasurementSupport(True, None, "test")

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


class UnavailableQualityBenchmark(FakeBenchmark):
    def formal_measurement_support(
        self, execution_mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport:
        del execution_mode, lifecycle
        return MagpieFormalMeasurementSupport(
            False,
            "magpie_local_quality_execution_unavailable",
            None,
            ("magpie_inferencex_eval_argument_mismatch",),
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
            KernelVolume(10, 95.0, 95.0),
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
    def resolve(self, resolved, *, gpu_arch, hints=None):
        return RunProvenance(
            2,
            str(resolved.config_path),
            "a" * 64,
            "vllm",
            "Qwen/example",
            None,
            gpu_arch,
            "docker",
            ContainerIdentity("example:v1", "sha256:" + "d" * 64, (), ()),
            ComponentSourceLockSet(("vllm", "aiter"), ()),
            "partial",
            ("model_revision", "source_lock:vllm", "source_lock:aiter"),
        )


class ForbiddenProvenance:
    def resolve(self, *_args, **_kwargs):
        raise AssertionError("provenance must not run for an unsupported config")


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


def _out_of_scope_spec(
    tmp_path: Path,
    results: Path,
    *,
    run_mode: str = "local",
    lifecycle: str = "one_shot",
) -> E2EOptimizeSpec:
    config = tmp_path / f"benchmark-{run_mode}-{lifecycle}.yaml"
    lifecycle_yaml = ""
    if lifecycle != "one_shot":
        cleanup = "true" if lifecycle == "cleanup" else "false"
        lifecycle_yaml = (
            "  server_lifecycle:\n"
            "    enabled: true\n"
            f"    cleanup: {cleanup}\n"
        )
    config.write_text(
        "benchmark:\n"
        "  framework: vllm\n"
        "  model: Qwen/example\n"
        f"  run_mode: {run_mode}\n"
        "  envs: {TP: 1, RUN_EVAL: true}\n"
        "  docker_image: example:v1\n"
        f"{lifecycle_yaml}",
        encoding="utf-8",
    )
    return E2EOptimizeSpec.from_mapping(
        {"config_path": str(config), "results_dir": str(results)}
    )


def test_e2e_vertical_slice_records_trace_and_no_regression(tmp_path: Path) -> None:
    results = tmp_path / "run"
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
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
        resolved_plans=ResolvedPlanStub(receipt),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    with pytest.raises(ConfigurationError) as failure:
        use_case.run(_spec(tmp_path, results))

    assert failure.value.reason_code == "lm_eval_runtime_missing"
    assert not results.exists()
    assert not tuple(tmp_path.glob(".apex-e2e-configs-*"))


def test_live_results_overlapping_dependency_fail_before_gpu(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    results = receipt.root("magpie") / "ignored-formal-results"
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    with pytest.raises(ContractError) as failure:
        use_case.run(_spec(tmp_path, results))

    assert failure.value.reason_code == "formal_results_overlap"
    assert not results.exists()


def test_e2e_preflight_emits_capability_receipt_without_gpu_lease(
    tmp_path: Path,
) -> None:
    results = tmp_path / "preflight"
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    preview = use_case.preview(_spec(tmp_path, results))

    assert preview.to_dict()["status"] == "config_compatible"
    assert preview.to_dict()["gpu_acquired"] is False
    assert preview.to_dict()["dimensions"]["framework"] == "vllm"
    source = preview.to_dict()["capabilities"]["source_optimization"]
    assert source["status"] == "capability_upgrade_required"
    assert source["routing"] is None
    assert preview.to_dict()["provenance"]["component_source_locks"] == {
        "schema": "apex.component-source-lock-set/v1",
        "required_components": ["vllm", "aiter"],
        "locks": [],
        "exact_components": [],
        "missing_exact_components": ["vllm", "aiter"],
    }
    assert not results.exists()
    path = write_preflight_result(preview, results)
    assert path == results / "preflight.json"
    assert json.loads(path.read_text())["gpu_acquired"] is False
    assert not tuple(tmp_path.glob(".apex-e2e-preflight-*"))


def test_e2e_preflight_reports_default_execution_attestor_unavailable(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    preview = use_case.preview(_spec(tmp_path, tmp_path / "preflight")).to_dict()

    assert preview["capabilities"]["benchmark_execution"] == {
        "adapter_id": "pinned-magpie-benchmark-v1",
        "available": False,
        "reason": "magpie_execution_attestor_unavailable",
    }
    assert preview["capabilities"]["formal_measurement"] == {
        "available": False,
        "reason": "magpie_execution_attestor_unavailable",
        "evaluator_execution_mode": None,
        "blockers": ["magpie_execution_attestor_unavailable"],
    }


def test_e2e_run_rejects_missing_attestor_before_gpu_lease(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    with pytest.raises(ContractError) as caught:
        use_case.run(_spec(tmp_path, tmp_path / "formal-results"))

    assert caught.value.reason_code == "magpie_execution_attestor_unavailable"


def test_e2e_run_rejects_missing_quality_authority_before_gpu_lease(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=UnavailableQualityBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    with pytest.raises(ContractError) as caught:
        use_case.run(_spec(tmp_path, tmp_path / "formal-results"))

    assert caught.value.reason_code == "magpie_local_quality_execution_unavailable"


def test_e2e_preflight_reports_capability_upgrade_without_materializing_views(
    tmp_path: Path,
) -> None:
    results = tmp_path / "upgrade-preflight"
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(
            receipt,
            status="capability_upgrade_required",
            blockers=("framework:future-serving",),
        ),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    preview = use_case.preview(_spec(tmp_path, results)).to_dict()

    assert preview["schema"] == "apex.e2e-preflight/v2"
    assert preview["status"] == "capability_upgrade_required"
    assert preview["config"]["view_status"] == "capability_upgrade_required"
    assert preview["config"]["effective_measurement_sha256"] is None
    assert preview["config"]["workload_semantics_sha256"] is None
    assert preview["magpie_contract"]["blockers"] == [
        "framework:future-serving"
    ]
    assert not results.exists()


def test_e2e_run_rejects_capability_upgrade_before_provenance_or_gpu(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        provenance=ForbiddenProvenance(),
        resolved_plans=ResolvedPlanStub(
            receipt,
            status="capability_upgrade_required",
            blockers=("framework:future-serving",),
        ),
        gpu_leases=_ForbiddenLeaseManager(),
    )

    with pytest.raises(ContractError) as caught:
        use_case.run(_spec(tmp_path, tmp_path / "upgrade-run"))

    assert caught.value.reason_code == "capability_upgrade_required"


@pytest.mark.parametrize(
    ("run_mode", "lifecycle"),
    (("local", "one_shot"), ("ray", "one_shot"), ("docker", "reuse"), ("docker", "cleanup")),
)
def test_e2e_v2_rejects_non_docker_one_shot_before_provenance_or_gpu(
    tmp_path: Path, run_mode: str, lifecycle: str
) -> None:
    receipt = _receipt(tmp_path)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        provenance=ForbiddenProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
        gpu_leases=_ForbiddenLeaseManager(),
    )
    spec = _out_of_scope_spec(
        tmp_path,
        tmp_path / f"unsupported-{run_mode}-{lifecycle}",
        run_mode=run_mode,
        lifecycle=lifecycle,
    )

    with pytest.raises(ContractError) as preview_failure:
        use_case.preview(spec)
    with pytest.raises(ContractError) as run_failure:
        use_case.run(spec)

    assert preview_failure.value.reason_code == "e2e_docker_only"
    assert run_failure.value.reason_code == "e2e_docker_only"
    assert preview_failure.value.details == {
        "run_mode": run_mode,
        "lifecycle": lifecycle,
    }
    assert not spec.results_dir.exists()


def test_resume_recovers_completed_baseline_and_retries_diagnostic(tmp_path: Path) -> None:
    results = tmp_path / "resume-run"
    benchmark = FakeBenchmark()
    receipt = _receipt(tmp_path)
    interrupted = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=benchmark,
        diagnostics=CrashDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
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
        resolved_plans=ResolvedPlanStub(receipt),
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
        resolved_plans=ResolvedPlanStub(receipt),
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
            resolved_plans=ResolvedPlanStub(receipt),
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
        resolved_plans=ResolvedPlanStub(receipt),
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
    receipt = _receipt(tmp_path)
    result = E2EOptimizeUseCase(
        dependency_receipt=receipt,
        benchmark=FakeBenchmark(final_throughput=98.0),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
        resolved_plans=ResolvedPlanStub(receipt),
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
