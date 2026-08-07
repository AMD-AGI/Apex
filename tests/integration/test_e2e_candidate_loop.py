from __future__ import annotations

import json
import shutil
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
from apex.core import AgentBackendName, TaskStatus, ValidationLevel, sha256_json
from apex.diagnostics import (
    AcquisitionCoverage,
    EvidenceArtifactReceipt,
    EvidenceArtifacts,
    KernelEvidence,
    KernelVolume,
    OperationEvidence,
    PerformanceModelEvidence,
    ShapeEvidence,
    TraceEvidence,
    derive_candidate_id,
)
from apex.evaluation import (
    CaseTiming,
    GateVerdict,
    PairedTimingUnit,
    SampleSeries,
    grade_kernel,
)
from apex.intake import E2EOptimizeSpec, MetricGoal, RegressionGates
from apex.optimization.e2e.candidate import E2ECandidate
from apex.optimization.e2e.deferred import E2EDeferredMicroQualifier
from apex.optimization.e2e.services import (
    CandidateDeployment,
    FinalDeliveryResult,
    MicroQualification,
    SafetyQualification,
)
from apex.optimization.e2e.use_case import E2EOptimizeUseCase
from apex.orchestration import RunController
from apex.ports import (
    AgentCost,
    AgentResult,
    AgentSemanticEvent,
    AgentTranscriptEvent,
    AgentUsage,
    BenchmarkPass,
    DiagnosticsResult,
)
from apex.runtime import (
    ContainerIdentity,
    DependencyReceipt,
    GpuLeaseReceipt,
    LmEvalRuntimeReceipt,
    RunProvenance,
)
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal, SnapshotStore


class _LeaseState:
    active = False


class _Lease:
    def __init__(self, run_id: str, state: _LeaseState) -> None:
        self.state = state
        self.receipt = GpuLeaseReceipt(1, run_id, "test-gpu", 1, 1.0, "/tmp/test.lock")

    def __enter__(self):
        assert not self.state.active
        self.state.active = True
        return self

    def __exit__(self, *_args):
        assert self.state.active
        self.state.active = False


class _LeaseManager:
    def __init__(self, state: _LeaseState) -> None:
        self.state = state
        self.calls = 0
        self.requested_devices: str | None = None

    def acquire(
        self, run_id: str, *, requested_devices: str | None = None
    ) -> _Lease:
        self.calls += 1
        self.requested_devices = requested_devices
        return _Lease(run_id, self.state)


class _Benchmark:
    def __init__(self, state: _LeaseState, measurements: list[float]) -> None:
        self.state = state
        self.measurements = iter(measurements)
        self.calls = []
        self.index = 0

    def run_normalized(self, request):
        assert self.state.active
        self.calls.append(request)
        self.index += 1
        workspace = request.output_dir / f"benchmark-{self.index}"
        workspace.mkdir(parents=True)
        report = workspace / "benchmark_report.json"
        quality_path = workspace / "quality.json"
        report.write_text("{}", encoding="utf-8")
        quality_path.write_text("{}", encoding="utf-8")
        throughput = (
            next(self.measurements)
            if request.pass_type is BenchmarkPass.MEASUREMENT
            else 100.0
        )
        latency = LatencyDistribution(1.0, 1.0, 1.0, 0.0)
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
            latency=LatencyMetrics(latency, latency, latency, latency),
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


class _Diagnostics:
    def __init__(self, state: _LeaseState, source_root: Path) -> None:
        self.state = state
        self.source_root = source_root
        self.calls = 0

    def analyze(self, request):
        assert self.state.active
        self.calls += 1
        source_name = "kernel.py" if self.calls == 1 else "kernel2.py"
        evidence = _eligible_evidence(
            self.source_root,
            request.provenance_hash,
            source_name=source_name,
        )
        request.output_dir.mkdir(parents=True)
        path = request.output_dir / "trace_evidence.json"
        path.write_text(
            json.dumps({"schema_version": 1, "records": [evidence.to_dict()]}),
            encoding="utf-8",
        )
        return DiagnosticsResult(
            request.run_id,
            True,
            (path,),
            {"record_count": 1, "evidence_path": str(path)},
        )


class _Provenance:
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
            ("vllm",),
            (),
            "partial",
            ("model_revision", "source_lock:vllm"),
        )


class _Worker:
    def __init__(self, state: _LeaseState) -> None:
        self.state = state
        self.requests = []

    def generate(self, request):
        assert self.state.active
        self.requests.append(request)
        relative = request.opportunity.source_path.relative_to(
            request.opportunity.source_root
        )
        source = request.destination / relative
        source.parent.mkdir(parents=True)
        source.write_text(f"value = {len(self.requests) + 1}\n", encoding="utf-8")
        digest = sha256_json({"attempt": request.attempt_id, "value": source.read_text()})
        events = (
            AgentTranscriptEvent(
                "assistant",
                metadata={"type": "assistant", "message": {"role": "assistant"}},
            ),
            AgentTranscriptEvent(
                "tool.called",
                metadata={"type": "tool.called", "tool_name": "profile"},
            ),
            AgentTranscriptEvent(
                "tool.result",
                metadata={"type": "tool.result", "tool_name": "profile"},
            ),
            AgentTranscriptEvent(
                "result",
                metadata={
                    "type": "result",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "total_cost_usd": 0.01,
                },
            ),
        )
        semantic = (
            AgentSemanticEvent(0, 0, "assistant", "agent_message", "assistant", "done"),
            AgentSemanticEvent(1, 1, "tool.called", "tool_called", tool_name="profile", tool_call_id="tool-1"),
            AgentSemanticEvent(2, 2, "tool.result", "tool_result", tool_name="profile", tool_call_id="tool-1", succeeded=True),
        )
        result = AgentResult(
            backend=AgentBackendName.CODEX,
            model=request.model,
            exit_code=0,
            timed_out=False,
            events=events,
            stdout="{}\n",
            stderr="",
            duration_seconds=0.01,
            semantic_events=semantic,
            usage=AgentUsage(10, None, None, 5, None, 15, 1, 1, (0, 1, 2, 3)),
            cost=AgentCost("0.01", "USD", 3, "total_cost_usd"),
        )
        return E2ECandidate(
            request.attempt_id,
            f"candidate-{request.attempt_id}",
            True,
            "candidate_frozen",
            request.destination,
            (relative.as_posix(),),
            (relative.as_posix(),),
            "b" * 64,
            digest,
            result,
        )


class _Micro:
    def __init__(self, state: _LeaseState, outcomes: list[bool]) -> None:
        self.state = state
        self.outcomes = iter(outcomes)
        self.calls = 0

    def supports(self, _opportunity) -> bool:
        return True

    def verify(self, request):
        assert self.state.active
        self.calls += 1
        passed = next(self.outcomes)
        return MicroQualification(
            candidate_id=str(request.candidate.candidate_id),
            grade=_micro_grade(passed=passed),
            evidence={"independent": True},
        )


def _micro_grade(*, passed: bool = True, speedup: float = 1.08, samples: int = 300):
    gates = GateVerdict(True, passed, True, True)
    if not passed:
        return grade_kernel(gates, ())
    counts = (samples // 2 + samples % 2, samples // 2)
    reference = (10.0,) * samples
    optimized = (10.0 / speedup,) * samples
    units = tuple(
        PairedTimingUnit(
            index,
            (10.0,) * count,
            (10.0 / speedup,) * count,
        )
        for index, count in enumerate(counts)
    )
    return grade_kernel(
        gates,
        (CaseTiming("micro", SampleSeries(reference), SampleSeries(optimized), paired_units=units),),
    )


class _Safety:
    def __init__(self, state: _LeaseState, *, finding: bool = False) -> None:
        self.state = state
        self.finding = finding
        self.calls = 0

    def verify(self, request):
        assert self.state.active
        self.calls += 1
        return SafetyQualification(
            candidate_id=str(request.candidate.candidate_id),
            allowed_to_measure=not self.finding,
            promotion_eligible=not self.finding,
            safety_certified=False,
            finding=self.finding,
            reason_codes=("finding" if self.finding else "no_tools_configured",),
            evidence={"isolation": True},
        )


class _Deployments:
    def __init__(self, state: _LeaseState, *, succeed: bool = True) -> None:
        self.state = state
        self.succeed = succeed
        self.requests = []
        self.results = []
        self.rollbacks = []

    def supports(self, _opportunity, _provenance) -> bool:
        return True

    def deploy(self, request):
        assert self.state.active
        self.requests.append(request)
        request.artifact_root.mkdir(parents=True)
        configs = []
        replay = request.benchmark_replay or request.benchmark_measurement
        for name, source in (
            ("measurement.yaml", request.benchmark_measurement),
            ("diagnostic.yaml", request.benchmark_diagnostic),
            ("replay.yaml", replay),
        ):
            destination = request.artifact_root / name
            shutil.copyfile(source, destination)
            configs.append(destination)
        result = CandidateDeployment(
            candidate_id=str(request.candidate.candidate_id),
            deployed=self.succeed,
            reason_code="deployed" if self.succeed else "source_build_failed",
            measurement_config=configs[0],
            diagnostic_config=configs[1],
            replay_config=configs[2],
            workload_semantics_sha256=request.workload_semantics_sha256,
            deployed_source_sha256=str(request.candidate.candidate_source_sha256),
            validation_level=(
                ValidationLevel.RUNTIME_OVERLAY_VERIFIED
                if self.succeed
                else ValidationLevel.NONE
            ),
            engagement_verified=self.succeed,
            evidence={"loaded_bytes": self.succeed},
        )
        self.results.append(result)
        return result

    def rollback(self, deployment) -> None:
        assert self.state.active
        self.rollbacks.append(deployment.candidate_id)


class _FinalDelivery:
    def __init__(self, state: _LeaseState, *, succeed: bool = True) -> None:
        self.state = state
        self.succeed = succeed
        self.requests = []

    def finalize(self, request):
        assert self.state.active
        self.requests.append(request)
        if not self.succeed:
            return FinalDeliveryResult(
                False,
                TaskStatus.PROVENANCE_UNRESOLVED,
                "source_provenance_unresolved",
                ValidationLevel.RUNTIME_OVERLAY_VERIFIED,
                False,
                None,
                None,
                {"second_environment": False},
            )
        request.artifact_root.mkdir(parents=True)
        bundle = request.artifact_root / "bundle.json"
        bundle.write_text("{}\n", encoding="utf-8")
        return FinalDeliveryResult(
            True,
            TaskStatus.SUCCEEDED,
            "verified",
            ValidationLevel.SOURCE_REBUILD_VERIFIED,
            True,
            str(bundle),
            "f" * 64,
            {"second_environment": True},
        )


def _eligible_evidence(
    root: Path,
    provenance_hash: str,
    *,
    source_name: str = "kernel.py",
) -> TraceEvidence:
    suffix = "2" if source_name == "kernel2.py" else ""
    kernel = KernelEvidence(
        runtime_name=f"kernel{suffix}",
        language="triton",
        origin_library="aiter",
        source_path=str(root / source_name),
        source_confidence="active_finder",
        patchable=True,
        source_root=str(root),
        test_file=str(root / f"test_kernel{suffix}.py"),
        test_command=f"pytest test_kernel{suffix}.py",
    )
    shape = ShapeEvidence(dtypes=("float16",), concrete_inputs=("[16, 128]",))
    candidate = derive_candidate_id(
        provenance_hash=provenance_hash,
        phase="decode",
        rank=0,
        kernel=kernel,
        shape=shape,
    )
    return TraceEvidence(
        1,
        candidate,
        provenance_hash,
        "decode",
        0,
        OperationEvidence("attention", "kernel"),
        kernel,
        shape,
        KernelVolume(100, 10.0, 10.0),
        PerformanceModelEvidence(),
        EvidenceArtifacts(
            "TargetedKernelTrace",
            AcquisitionCoverage(10, 10, 10, 0),
            (
                EvidenceArtifactReceipt("targeted_manifest", "manifest.json", "c" * 64, 1, "application/json"),
                EvidenceArtifactReceipt("targeted_shard", "shard.jsonl", "d" * 64, 1, "application/x-ndjson"),
            ),
            "e" * 64,
        ),
    )


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    (root / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    (root / "kernel2.py").write_text("value = 1\n", encoding="utf-8")
    (root / "test_kernel2.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    return root


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


def _spec(tmp_path: Path, *, iterations: int) -> E2EOptimizeSpec:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "benchmark:\n  framework: vllm\n  model: Qwen/example\n  envs: {TP: 1, RUN_EVAL: true}\n  docker_image: example:v1\n",
        encoding="utf-8",
    )
    return E2EOptimizeSpec.from_mapping(
        {
            "config_path": str(config),
            "results_dir": str(tmp_path / "results"),
            "max_kernels": 1,
            "max_iterations": iterations,
        }
    )


def _system(
    tmp_path: Path,
    measurements: list[float],
    micro_outcomes: list[bool],
    *,
    safety_finding: bool = False,
    deployment_succeeds: bool = True,
    final_succeeds: bool = True,
    deferred_micro: bool = False,
):
    state = _LeaseState()
    source = _source(tmp_path)
    lease = _LeaseManager(state)
    benchmark = _Benchmark(state, measurements)
    diagnostics = _Diagnostics(state, source)
    worker = _Worker(state)
    micro = E2EDeferredMicroQualifier() if deferred_micro else _Micro(state, micro_outcomes)
    safety = _Safety(state, finding=safety_finding)
    deployments = _Deployments(state, succeed=deployment_succeeds)
    final = _FinalDelivery(state, succeed=final_succeeds)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=_receipt(tmp_path),
        benchmark=benchmark,
        diagnostics=diagnostics,
        provenance=_Provenance(),
        candidate_worker=worker,
        micro=micro,
        safety=safety,
        deployments=deployments,
        final_delivery=final,
        gpu_leases=lease,
    )
    return use_case, state, lease, benchmark, diagnostics, worker, micro, safety, deployments, final


def test_two_keeps_use_current_overlay_and_require_clean_delivery(tmp_path: Path) -> None:
    system = _system(tmp_path, [100.0, 102.0, 104.0, 104.0], [True, True])
    use_case, state, lease, _, diagnostics, worker, _, _, deployments, final = system

    spec = replace(
        _spec(tmp_path, iterations=2),
        deployment_hints={"gpu_devices": "2"},
    )
    result = use_case.run(spec)

    assert result.status is TaskStatus.SUCCEEDED
    assert result.validation_level is ValidationLevel.SOURCE_REBUILD_VERIFIED
    assert result.intake_provenance_status == "partial"
    assert result.intake_missing_evidence == ("model_revision", "source_lock:vllm")
    assert result.formal_delivery_verified is True
    assert len(result.accepted_patch_ids) == 2
    assert lease.calls == 1 and lease.requested_devices == "2" and not state.active
    assert len(worker.requests) == 2 and diagnostics.calls == 3
    assert deployments.requests[1].benchmark_measurement == deployments.results[0].measurement_config
    assert deployments.requests[1].benchmark_replay == deployments.results[0].replay_config
    assert final.requests[0].benchmark_replay == deployments.results[1].replay_config
    recovered = RunController.recover(
        result.run_id,
        EventJournal(tmp_path / "results" / "events" / "run.db"),
        SnapshotStore(tmp_path / "results" / "state.snapshot.json"),
    )
    assert recovered.state.e2e is not None
    assert recovered.state.e2e.final_clean_replay_verified is True
    events = EventJournal(tmp_path / "results" / "events" / "run.db").iter_events(
        result.run_id
    )
    micro_measurements = tuple(
        event
        for event in events
        if event.event_type == "measurement_result"
        and event.payload.get("attempt_id")
    )
    assert len(micro_measurements) == 2
    assert all(event.payload["promotion_eligible"] is True for event in micro_measurements)
    assert all(event.payload["srobust_ci_lower"] > 1.0 for event in micro_measurements)
    assert all(event.payload["bootstrap_seed"] == 1729 for event in micro_measurements)


def test_agent_transcript_usage_and_cost_are_attempt_scoped(tmp_path: Path) -> None:
    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])

    result = system[0].run(_spec(tmp_path, iterations=1))

    journal = EventJournal(tmp_path / "results" / "events" / "run.db")
    events = tuple(journal.iter_events(result.run_id))
    attempt = tuple(
        event for event in events if event.payload.get("attempt_id") == "attempt-1"
    )
    types = [event.event_type for event in attempt]
    expected = (
        "agent_message",
        "tool_called",
        "tool_result",
        "usage_recorded",
        "cost_recorded",
        "agent_completed",
        "candidate_frozen",
    )
    assert all(types.index(first) < types.index(second) for first, second in zip(expected, expected[1:]))
    assert all(event.payload["candidate_id"] == "candidate-attempt-1" for event in attempt if event.event_type in expected)
    assert all(
        event.payload["backend"] == "codex"
        for event in attempt
        if event.event_type in expected[:-1]
    )
    usage = next(event for event in attempt if event.event_type == "usage_recorded")
    assert usage.payload["input_tokens"] == 10
    assert usage.payload["output_tokens"] == 5
    assert usage.payload["tool_call_count"] == 1
    cost = next(event for event in attempt if event.event_type == "cost_recorded")
    assert cost.payload["amount"] == "0.01"
    assert cost.payload["currency"] == "USD"
    completed = next(event for event in attempt if event.event_type == "agent_completed")
    binding = next(
        item for item in completed.payload["artifacts"] if item["role"] == "agent_transcript"
    )
    receipt = ArtifactReceipt.from_dict(binding["receipt"])
    transcript = json.loads(
        ArtifactStore(tmp_path / "results" / "artifacts").read_bytes(receipt)
    )
    assert transcript["schema"] == "apex.agent-transcript/v1"
    assert [event["kind"] for event in transcript["semantic_events"]] == [
        "agent_message",
        "tool_called",
        "tool_result",
    ]
    assert transcript["usage"]["total_tokens"] == 15
    assert transcript["cost"]["amount"] == "0.01"


def test_e2e_revert_rolls_back_and_returns_no_gain(tmp_path: Path) -> None:
    system = _system(tmp_path, [100.0, 100.0, 100.0], [True])
    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.NO_GAIN
    assert result.accepted_patch_ids == ()
    assert system[8].rollbacks == ["candidate-attempt-1"]
    assert system[9].requests == []


def test_final_no_regression_uses_the_exact_requested_gates(
    tmp_path: Path, monkeypatch
) -> None:
    import apex.optimization.e2e.finalization as finalization

    observed = []
    original = finalization.evaluate_no_regression

    def capture_policy(baseline, replay, policy=None, **kwargs):
        observed.append(policy)
        return original(baseline, replay, policy=policy, **kwargs)

    monkeypatch.setattr(finalization, "evaluate_no_regression", capture_policy)
    system = _system(tmp_path, [100.0, 100.0, 100.0], [True])
    spec = replace(
        _spec(tmp_path, iterations=1),
        goal=MetricGoal(
            gates=RegressionGates(
                accuracy_regression_pct=0.0,
                ttft_p99_regression_pct=0.25,
                tpot_p99_regression_pct=0.5,
            )
        ),
    )

    result = system[0].run(spec)

    assert result.status is TaskStatus.NO_GAIN
    assert len(observed) == 1
    assert observed[0].gates == spec.goal.gates


def test_micro_failure_retries_with_fresh_context_history(tmp_path: Path) -> None:
    system = _system(tmp_path, [100.0, 100.0], [False, False])
    result = system[0].run(_spec(tmp_path, iterations=2))

    assert result.status is TaskStatus.NO_GAIN
    assert len(system[5].requests) == 2
    assert system[5].requests[0].prompt != system[5].requests[1].prompt
    assert "correctness_or_integrity_gate" in system[5].requests[1].prompt
    assert system[8].requests == []


def test_safety_finding_blocks_deployment(tmp_path: Path) -> None:
    system = _system(
        tmp_path,
        [100.0, 100.0],
        [True],
        safety_finding=True,
    )
    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.NO_GAIN
    assert system[7].calls == 1
    assert system[8].requests == []


def test_deployment_failure_never_runs_candidate_e2e(tmp_path: Path) -> None:
    system = _system(
        tmp_path,
        [100.0, 100.0],
        [True],
        deployment_succeeds=False,
    )
    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.NO_GAIN
    assert len(system[3].calls) == 3  # baseline, diagnostic, unchanged final
    assert system[8].rollbacks == ["candidate-attempt-1"]


def test_formal_success_is_denied_when_second_delivery_is_unresolved(tmp_path: Path) -> None:
    system = _system(
        tmp_path,
        [100.0, 102.0, 102.0],
        [True],
        final_succeeds=False,
    )
    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.PROVENANCE_UNRESOLVED
    assert result.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    assert result.accepted_patch_ids
    assert result.details["final_delivery"]["clean_replay_verified"] is False


def test_strict_micro_uses_canonical_sample_and_strict_threshold_gates() -> None:
    short = MicroQualification("candidate-1", _micro_grade(samples=299), {})
    boundary = MicroQualification("candidate-1", _micro_grade(speedup=1.05), {})
    accepted = MicroQualification("candidate-1", _micro_grade(speedup=1.051), {})

    assert short.qualified is False
    assert short.reason_code == "insufficient_samples"
    assert boundary.qualified is False
    assert boundary.reason_code == "srobust_threshold_not_met"
    assert accepted.qualified is True


def test_deferred_micro_journal_makes_no_compile_correctness_or_timing_claim(
    tmp_path: Path,
) -> None:
    system = _system(
        tmp_path,
        [100.0, 102.0, 102.0],
        [],
        deferred_micro=True,
    )
    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.SUCCEEDED
    events = tuple(
        EventJournal(tmp_path / "results" / "events" / "run.db").iter_events(
            result.run_id
        )
    )
    attempt_events = tuple(
        event for event in events if event.payload.get("attempt_id") == "attempt-1"
    )
    assert not {
        "compile_result",
        "correctness_result",
        "measurement_result",
    }.intersection(event.event_type for event in attempt_events)
    deferred = [
        event
        for event in attempt_events
        if event.event_type == "tool_result"
        and event.payload.get("qualification_mode") == "e2e_quality_deferred"
    ]
    assert len(deferred) == 1
    assert deferred[0].payload["kernel_reward_available"] is False
