from __future__ import annotations

import json
from pathlib import Path

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
from apex.core import TaskStatus
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
from apex.orchestration import RunController
from apex.ports import BenchmarkPass, DiagnosticsResult
from apex.runtime import (
    ContainerIdentity,
    DependencyReceipt,
    LmEvalRuntimeReceipt,
    RunProvenance,
)
from apex.storage import EventJournal, SnapshotStore


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


def test_e2e_final_replay_regression_fails_verification(tmp_path: Path) -> None:
    results = tmp_path / "run-regression"
    result = E2EOptimizeUseCase(
        dependency_receipt=_receipt(tmp_path),
        benchmark=FakeBenchmark(final_throughput=98.0),
        diagnostics=FakeDiagnostics(),
        provenance=FakeProvenance(),
    ).run(_spec(tmp_path, results))
    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.no_regression is False
    assert result.reason_code == "insufficient_throughput_gain"
