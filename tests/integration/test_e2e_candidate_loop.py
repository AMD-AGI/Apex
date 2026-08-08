from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import apex.optimization.e2e.candidate_snapshot as candidate_snapshot
import apex.optimization.e2e.search as search_module
from apex.benchmark import (
    InferenceXRuntimeEvidence,
    LatencyDistribution,
    LatencyMetrics,
    ModelRevisionEvidence,
    NormalizedBenchmarkResult,
    QualityEvidence,
    QualityMetric,
    ServingRuntimeEvidence,
    ThroughputMetrics,
)
from apex.core import (
    AgentBackendName,
    ContractError,
    DependencyError,
    IntegrityError,
    TaskStatus,
    ValidationLevel,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
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
    E2EAcceptancePolicy,
    GateVerdict,
    PairedTimingUnit,
    SampleSeries,
    grade_kernel,
)
from apex.intake import E2EOptimizeSpec, MetricGoal, RegressionGates
from apex.knowledge import KnowledgeCard, KnowledgeRetriever
from apex.optimization.e2e.candidate import E2ECandidate, FrozenCandidateSource
from apex.optimization.e2e.context import E2EContextBuilder
from apex.optimization.e2e.deferred import E2EDeferredMicroQualifier
from apex.optimization.e2e.promotion_recovery import recover_matched_promotion
from apex.optimization.e2e.recovery import load_run_request, recover_record
from apex.optimization.e2e.run_record import E2ERunRecord
from apex.optimization.e2e.services import (
    CandidateDeployment,
    DeploymentConfigDigests,
    FinalDeliveryResult,
    MicroQualification,
    NoToolSafetyVerifier,
    SafetyQualification,
)
from apex.optimization.e2e.use_case import E2EOptimizeUseCase
from apex.orchestration import RunController, SearchStage
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentCaptureStatus,
    AgentCost,
    AgentProcessContainmentReceipt,
    AgentResult,
    AgentSemanticEvent,
    AgentTranscriptEvent,
    AgentUsage,
    BenchmarkPass,
    DiagnosticsResult,
)
from apex.rl import (
    EpisodeGraphMaterializer,
    SemanticRole,
)


def _agent_containment() -> AgentProcessContainmentReceipt:
    return AgentProcessContainmentReceipt(
        policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        launcher_path="/usr/bin/bwrap",
        launcher_sha256="b" * 64,
        namespace_init_host_pid=100,
        namespace_init_starttime=200,
        namespace_init_inner_pid=1,
        pid_namespace_inode=300,
        mount_namespace_inode=301,
        ipc_namespace_inode=302,
        user_namespace_inode=303,
        private_procfs_verified=True,
        pidfd_opened=True,
        termination_reason="natural_exit",
        teardown_mode="natural_exit",
        pidfd_sigkill_sent=False,
        namespace_init_exit_verified=True,
        wrapper_exit_verified=True,
        wrapper_force_killed=False,
        terminal_status_verified=True,
        terminal_status_absent_after_sigkill=False,
        status_eof_verified=True,
        namespace_membership_scan_complete=True,
        live_namespace_members_after=(),
    )
from apex.runtime import (
    ContainerIdentity,
    DependencyReceipt,
    GpuDeviceIdentity,
    GpuLeaseReceipt,
    GpuOwnershipReceipt,
    LmEvalRuntimeReceipt,
    RunProvenance,
)
from apex.storage import (
    ArtifactReceipt,
    ArtifactStore,
    EventJournal,
    EventRecord,
    SnapshotStore,
)


_CANDIDATE_IMAGE_ID = "sha256:" + "f" * 64
_BASELINE_IMAGE_ID = "sha256:" + "e" * 64


class _LeaseState:
    active = False


class _Lease:
    def __init__(
        self, run_id: str, state: _LeaseState, generation: int, unique_id: str
    ) -> None:
        self.state = state
        ownership = GpuOwnershipReceipt(
            1,
            "rocm_smi_process_gpu_map_v1",
            "amd-gpu-set=0",
            123,
            "/opt/rocm/lib/librocm_smi64.so.7",
            "a" * 64,
            (GpuDeviceIdentity(0, unique_id, "/dev/dri/renderD128"),),
            (),
            (),
        )
        self.receipt = GpuLeaseReceipt(
            1,
            run_id,
            ownership.physical_scope,
            1,
            float(generation),
            "/tmp/test.lock",
            ownership,
        )

    def __enter__(self):
        assert not self.state.active
        self.state.active = True
        return self

    def __exit__(self, *_args):
        assert self.state.active
        self.state.active = False


class _LeaseManager:
    def __init__(
        self,
        state: _LeaseState,
        unique_ids: tuple[str, ...] = ("0x0000000000000001",),
    ) -> None:
        self.state = state
        self.unique_ids = unique_ids
        self.calls = 0
        self.requested_devices: str | None = None

    def acquire(
        self, run_id: str, *, requested_devices: str | None = None
    ) -> _Lease:
        self.calls += 1
        self.requested_devices = requested_devices
        unique_id = self.unique_ids[min(self.calls - 1, len(self.unique_ids) - 1)]
        return _Lease(run_id, self.state, self.calls, unique_id)


class _Benchmark:
    def __init__(
        self,
        state: _LeaseState,
        measurements: list[float],
        *,
        resolved_candidate_image_id: str = _CANDIDATE_IMAGE_ID,
        mutate_candidate_config: bool = False,
        matched_measurements: list[float] | None = None,
    ) -> None:
        self.state = state
        self.measurements = iter(measurements)
        self.anchor_throughput: float | None = None
        self.candidate_throughput: dict[str, float] = {}
        self.resolved_candidate_image_id = resolved_candidate_image_id
        self.mutate_candidate_config = mutate_candidate_config
        self.matched_measurements = (
            iter(matched_measurements) if matched_measurements is not None else None
        )
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
        throughput = self._throughput(request)
        candidate_run = "promotion-" in request.run_id and "-candidate" in request.run_id
        promotion_run = request.run_id.startswith("promotion-")
        if candidate_run and self.mutate_candidate_config:
            request.config_path.write_text(
                request.config_path.read_text(encoding="utf-8") + "\n# drift\n",
                encoding="utf-8",
            )
        image = (
            _CANDIDATE_IMAGE_ID
            if candidate_run or "/delivery/" in str(request.config_path)
            else _BASELINE_IMAGE_ID
        )
        serving_runtime = ServingRuntimeEvidence(
            required=promotion_run,
            passed=True,
            input_config_sha256=(
                sha256_file(request.config_path) if promotion_run else None
            ),
            requested_image=image if promotion_run else None,
            resolved_image_id=(
                (
                    self.resolved_candidate_image_id
                    if image == _CANDIDATE_IMAGE_ID
                    else image
                )
                if promotion_run
                else None
            ),
            container_name="magpie-benchmark-test" if promotion_run else None,
            docker_argv_sha256="b" * 64 if promotion_run else None,
            process_succeeded=True if promotion_run else None,
            error=None,
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
            serving_runtime=serving_runtime,
        )

    def _throughput(self, request) -> float:
        if request.pass_type is not BenchmarkPass.MEASUREMENT:
            return 100.0
        action = request.run_id
        if action == "baseline-measurement" or action.startswith("baseline-resume-"):
            value = float(next(self.measurements))
            self.anchor_throughput = value
            return value
        if action.startswith("promotion-"):
            if self.matched_measurements is not None:
                return float(next(self.matched_measurements))
            if "-candidate" in action:
                attempt = action.split("-window-", 1)[0]
                if attempt not in self.candidate_throughput:
                    self.candidate_throughput[attempt] = float(next(self.measurements))
                return self.candidate_throughput[attempt]
            assert self.anchor_throughput is not None
            if action.endswith("ba-anchor"):
                attempt = action.split("-window-", 1)[0]
                candidate = self.candidate_throughput.get(attempt)
                if candidate is not None and candidate >= self.anchor_throughput * 1.005:
                    prior = self.anchor_throughput
                    self.anchor_throughput = candidate
                    return prior
            return self.anchor_throughput
        return float(next(self.measurements))


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
    def __init__(
        self,
        state: _LeaseState,
        *,
        infrastructure_error: Exception | None = None,
        returned_failure: str | None = None,
    ) -> None:
        self.state = state
        self.requests = []
        self.infrastructure_error = infrastructure_error
        self.returned_failure = returned_failure

    def generate(self, request):
        assert self.state.active
        self.requests.append(request)
        if self.infrastructure_error is not None:
            raise self.infrastructure_error
        relative = request.opportunity.source_path.relative_to(
            request.opportunity.source_root
        )
        source = request.destination / relative
        source.parent.mkdir(parents=True)
        source.write_text(f"value = {len(self.requests) + 1}\n", encoding="utf-8")
        content = source.read_bytes()
        source_sha256 = sha256_bytes(content)
        source_mode = source.stat().st_mode & 0o777
        digest = sha256_json(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": relative.as_posix(),
                        "sha256": source_sha256,
                        "mode": source_mode,
                    }
                ],
            }
        )
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
            process_containment=_agent_containment(),
        )
        if self.returned_failure == "agent_process_cleanup_failed":
            result = replace(result, capture_status=AgentCaptureStatus.CLEANUP_FAILED)
        elif self.returned_failure == "agent_process_containment_unverified":
            result = replace(result, process_containment=None)
        candidate = E2ECandidate(
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
            (
                FrozenCandidateSource(
                    relative.as_posix(), source_sha256, source_mode, content
                ),
            ),
        )
        if self.returned_failure is None:
            return candidate
        return replace(
            candidate,
            candidate_id=None,
            succeeded=False,
            reason_code=self.returned_failure,
            changed_files=(),
            candidate_source_sha256=None,
            frozen_sources=(),
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
    def __init__(
        self,
        state: _LeaseState,
        *,
        succeed: bool = True,
        infrastructure_failure: bool = False,
    ) -> None:
        self.state = state
        self.succeed = succeed
        self.infrastructure_failure = infrastructure_failure
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
        config_sha256 = (
            DeploymentConfigDigests.capture(*configs) if self.succeed else None
        )
        result = CandidateDeployment(
            candidate_id=str(request.candidate.candidate_id),
            deployed=self.succeed,
            reason_code=(
                "deployed"
                if self.succeed
                else (
                    "container_command_failed"
                    if self.infrastructure_failure
                    else "invalid_frozen_candidate"
                )
            ),
            measurement_config=configs[0],
            diagnostic_config=configs[1],
            replay_config=configs[2],
            workload_semantics_sha256=request.workload_semantics_sha256,
            deployed_source_sha256=str(request.candidate.candidate_source_sha256),
            deployed_image_id=_CANDIDATE_IMAGE_ID if self.succeed else None,
            validation_level=(
                ValidationLevel.RUNTIME_OVERLAY_VERIFIED
                if self.succeed
                else ValidationLevel.NONE
            ),
            engagement_verified=self.succeed,
            evidence={
                "loaded_bytes": self.succeed,
                **(
                    {
                        "derived_image": {"image_id": _CANDIDATE_IMAGE_ID},
                        "config_sha256": config_sha256.to_dict(),
                    }
                    if self.succeed
                    else {}
                ),
            },
            infrastructure_failure=not self.succeed and self.infrastructure_failure,
            config_sha256=config_sha256,
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


def _knowledge_card(claim: str, kind: str) -> KnowledgeCard:
    return KnowledgeCard.from_mapping(
        {
            "kind": kind,
            "status": "imported_unverified",
            "scope": {
                "operator": [],
                "gpu_arch": [],
                "dtype": [],
                "regime": [],
                "language": [],
                "framework": [],
                "versions": {},
            },
            "claim": claim,
            "apply": f"Consider {claim}",
            "verify": "Measure under the frozen workload.",
            "caution": "Advisory only.",
            "source": {
                "repository": "https://example.invalid/geak",
                "git_sha": "1" * 40,
                "path": f"perf_knowledge/{sha256_json(claim)[:8]}.md",
                "license": "Apache-2.0",
                "content_sha256": sha256_bytes(claim.encode()),
                "transform_version": "test_v1",
            },
        }
    )


def _system(
    tmp_path: Path,
    measurements: list[float],
    micro_outcomes: list[bool],
    *,
    safety_finding: bool = False,
    deployment_succeeds: bool = True,
    deployment_infrastructure_failure: bool = False,
    final_succeeds: bool = True,
    deferred_micro: bool = False,
    worker_infrastructure_error: Exception | None = None,
    worker_returned_failure: str | None = None,
    safety_override=None,
    resolved_candidate_image_id: str = _CANDIDATE_IMAGE_ID,
    mutate_candidate_config: bool = False,
    matched_measurements: list[float] | None = None,
    contexts: E2EContextBuilder | None = None,
):
    state = _LeaseState()
    source = _source(tmp_path)
    lease = _LeaseManager(state)
    benchmark = _Benchmark(
        state,
        measurements,
        resolved_candidate_image_id=resolved_candidate_image_id,
        mutate_candidate_config=mutate_candidate_config,
        matched_measurements=matched_measurements,
    )
    diagnostics = _Diagnostics(state, source)
    worker = _Worker(
        state,
        infrastructure_error=worker_infrastructure_error,
        returned_failure=worker_returned_failure,
    )
    micro = E2EDeferredMicroQualifier() if deferred_micro else _Micro(state, micro_outcomes)
    safety = safety_override or _Safety(state, finding=safety_finding)
    deployments = _Deployments(
        state,
        succeed=deployment_succeeds,
        infrastructure_failure=deployment_infrastructure_failure,
    )
    final = _FinalDelivery(state, succeed=final_succeeds)
    use_case = E2EOptimizeUseCase(
        dependency_receipt=_receipt(tmp_path),
        benchmark=benchmark,
        diagnostics=diagnostics,
        provenance=_Provenance(),
        candidate_worker=worker,
        contexts=contexts,
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
    assert len(worker.requests) == 2 and diagnostics.calls == 4
    terminal = result.details["terminal_diagnostics"]
    assert terminal["comparison"] == {
        "status": "unavailable",
        "reason_code": "tracelens_perf_report_comparison_api_unavailable",
        "receipt": terminal["comparison"]["receipt"],
        "reward_eligible": False,
    }
    assert deployments.requests[0].accepted_stack == ()
    assert len(deployments.requests[1].accepted_stack) == 1
    assert (
        deployments.requests[1].accepted_stack[0].candidate.candidate_id
        == "candidate-attempt-1"
    )
    assert deployments.requests[1].anchor_generation == 1
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
    terminal_measurement = next(
        event
        for event in events
        if event.event_type == "measurement_result"
        and event.payload.get("action_id") == "final-diagnostic"
    )
    assert terminal_measurement.payload["evidence_class"] == "diagnostic"
    assert terminal_measurement.payload["reward_eligible"] is False
    comparison = next(
        event
        for event in events
        if event.event_type == "tool_result"
        and event.payload.get("tool") == "tracelens_terminal_comparison"
    )
    assert comparison.payload["status"] == "unavailable"
    assert comparison.payload["reward_eligible"] is False
    assert comparison.payload["artifacts"][0]["role"] == (
        "terminal_trace_comparison"
    )
    graph = EpisodeGraphMaterializer(
        EventJournal(tmp_path / "results" / "events" / "run.db"),
        ArtifactStore(tmp_path / "results" / "artifacts"),
    ).materialize(result.run_id, workload_state=recovered.state)
    projected_comparison = next(
        event
        for event in graph.parent.events
        if event.payload.get("tool") == "tracelens_terminal_comparison"
    )
    assert projected_comparison.payload["reward_eligible"] is False
    assert projected_comparison.artifacts[0].role == "terminal_trace_comparison"
    promotion_measurements = tuple(
        event
        for event in events
        if event.event_type == "measurement_result"
        and str(event.payload.get("action_id", "")).startswith(
            "promotion-"
        )
    )
    assert len(promotion_measurements) == 8
    expected_measurement_roles = {
        "benchmark_config",
        "normalized_benchmark",
        "quality_evidence",
        "benchmark_report",
        "quality_result",
    }
    for event in promotion_measurements:
        roles = {item["role"] for item in event.payload["artifacts"]}
        assert expected_measurement_roles <= roles
        assert "raw_measurement" not in roles
        assert event.payload["config_sha256"] == next(
            item["receipt"]["digest"]
            for item in event.payload["artifacts"]
            if item["role"] == "benchmark_config"
        )
    pairs = tuple(
        event
        for event in events
        if event.event_type == "measurement_result"
        and event.payload.get("measurement_kind") == "matched_promotion_ab_ba"
    )
    assert len(pairs) == 2
    assert all(
        event.payload["order"] == ["anchor", "candidate", "candidate", "anchor"]
        for event in pairs
    )
    assert all(
        {"matched_promotion_pair", "promotion_gpu_lease"}
        <= {item["role"] for item in event.payload["artifacts"]}
        for event in pairs
    )
    deliveries = tuple(
        event
        for event in events
        if event.event_type == "delivery_result"
        and event.payload.get("attempt_id")
    )
    assert len(deliveries) == 2
    assert all(
        {
            "primary_delivery",
            "delivery_measurement_config",
            "delivery_diagnostic_config",
            "delivery_replay_config",
        }
        <= {item["role"] for item in event.payload["artifacts"]}
        for event in deliveries
    )
    micro_measurements = tuple(
        event
        for event in events
        if event.event_type == "measurement_result"
        and event.payload.get("attempt_id")
        and event.payload.get("grade_policy_id")
    )
    assert len(micro_measurements) == 2
    assert all(event.payload["promotion_eligible"] is True for event in micro_measurements)
    assert all(event.payload["srobust_ci_lower"] > 1.0 for event in micro_measurements)
    assert all(event.payload["bootstrap_seed"] == 1729 for event in micro_measurements)
    decisions = tuple(
        event for event in events if event.event_type == "e2e.candidate_decided"
    )
    rewards = tuple(event for event in events if event.event_type == "reward_committed")
    assert len(decisions) == len(rewards) == 2
    assert all(
        decision.transaction_id == reward.transaction_id
        for decision, reward in zip(decisions, rewards, strict=True)
    )
    assert all(reward.payload["scalar_reward"] > 100.0 for reward in rewards)
    experiences = tuple(
        event for event in events if event.event_type == "experience.measured"
    )
    assert [event.payload["outcome"] for event in experiences] == ["success", "success"]
    assert [event.payload["evidence_receipts"] for event in experiences] == [
        [decision.payload["receipt"]] for decision in decisions
    ]
    context_events = tuple(
        event for event in events if event.event_type == "context_packet_created"
    )
    second_packet = next(
        item["receipt"]
        for item in context_events[1].payload["artifacts"]
        if item["role"] == "context_packet"
    )
    packet = json.loads(
        ArtifactStore(tmp_path / "results" / "artifacts").read_bytes(
            ArtifactReceipt.from_dict(second_packet)
        )
    )
    # The second diagnosis selects kernel2.py; the kernel.py experience must not leak
    # across the exact source/harness identity boundary.
    assert packet["relevant_history"]["attempts"] == []


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
    assert transcript["schema"] == "apex.agent-transcript/v3"
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
    assert result.no_regression is True
    assert result.details["observed_replay_verdict"]["keep"] is True
    assert result.details["final_replay_basis"]["source_identity_unchanged"] is True
    assert result.details["final_replay_basis"]["delivery_attempted"] is False
    assert system[8].rollbacks == ["candidate-attempt-1"]
    assert system[9].requests == []
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    decision = next(
        event for event in events if event.event_type == "e2e.candidate_decided"
    )
    reward = next(event for event in events if event.event_type == "reward_committed")
    assert decision.transaction_id == reward.transaction_id
    assert decision.payload["attempt_id"] == reward.payload["attempt_id"] == "attempt-1"
    assert reward.payload["scalar_reward"] == -10.0
    experience = next(
        event for event in events if event.event_type == "experience.measured"
    )
    assert experience.payload["outcome"] == "no_gain"
    assert experience.payload["evidence_receipts"] == [decision.payload["receipt"]]


def test_e2e_throughput_loss_records_regression_experience(tmp_path: Path) -> None:
    result = _system(tmp_path, [100.0, 99.0, 100.0], [True])[0].run(
        _spec(tmp_path, iterations=1)
    )

    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    experience = next(
        event for event in events if event.event_type == "experience.measured"
    )

    assert experience.payload["outcome"] == "regression"
    assert experience.payload["failure_reason"] == "insufficient_throughput_gain"


def test_e2e_knowledge_links_are_inconclusive_and_exported(tmp_path: Path) -> None:
    cards = (
        _knowledge_card("Fuse repeated loads", "fact"),
        _knowledge_card("Avoid oversized launch geometry", "anti_pattern"),
    )
    contexts = E2EContextBuilder(KnowledgeRetriever(cards))
    result = _system(
        tmp_path,
        [100.0, 102.0, 102.0],
        [True],
        contexts=contexts,
    )[0].run(_spec(tmp_path, iterations=1))
    journal = EventJournal(tmp_path / "results/events/run.db")
    artifacts = ArtifactStore(tmp_path / "results/artifacts")
    events = journal.iter_events(result.run_id)
    read = next(event for event in events if event.event_type == "knowledge_read")
    decision = next(
        event for event in events if event.event_type == "e2e.candidate_decided"
    )
    links = tuple(
        event for event in events if event.event_type == "knowledge_outcome_linked"
    )

    assert {event.payload["card_id"] for event in links} == set(read.payload["card_ids"])
    assert all(event.payload["outcome"] == "inconclusive" for event in links)
    assert all(event.payload["verdict"] == "keep" for event in links)
    assert all(
        event.payload["evidence_receipt"] == decision.payload["receipt"]
        for event in links
    )

    graph = EpisodeGraphMaterializer(journal, artifacts).materialize(result.run_id)
    child = graph.children[0]
    assert next(
        event for event in child.events if event.event_type == "experience.measured"
    ).semantic_role is SemanticRole.OUTCOME
    assert all(
        event.semantic_role is SemanticRole.OBSERVATION
        for event in child.events
        if event.event_type == "knowledge_outcome_linked"
    )


def test_accepted_winner_cumulative_replay_regression_fails_verification(
    tmp_path: Path,
) -> None:
    system = _system(tmp_path, [100.0, 102.0, 99.0], [True])

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "insufficient_throughput_gain"
    assert result.no_regression is False
    assert result.accepted_patch_ids
    assert result.formal_delivery_verified is False
    assert result.details["cumulative_verdict"]["keep"] is False
    assert system[9].requests == []
    recovered = RunController.recover(
        result.run_id,
        EventJournal(tmp_path / "results" / "events" / "run.db"),
        SnapshotStore(tmp_path / "results" / "state.snapshot.json"),
    )
    assert recovered.state.e2e is not None
    assert recovered.state.e2e.final_clean_replay_verified is False


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
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    assert [
        event.payload["outcome"]
        for event in events
        if event.event_type == "experience.measured"
    ] == ["failure", "failure"]
    second_context = tuple(
        event for event in events if event.event_type == "context_packet_created"
    )[1]
    packet_receipt = next(
        ArtifactReceipt.from_dict(item["receipt"])
        for item in second_context.payload["artifacts"]
        if item["role"] == "context_packet"
    )
    packet = json.loads(
        ArtifactStore(tmp_path / "results/artifacts").read_bytes(packet_receipt)
    )
    assert packet["relevant_history"]["attempts"][0]["candidate_id"] == (
        "candidate-attempt-1"
    )


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
    assert len(system[3].calls) == 4  # plus reward-ineligible terminal diagnostic
    assert system[8].rollbacks == ["candidate-attempt-1"]


def test_candidate_worker_infrastructure_failure_terminates_without_rejection(
    tmp_path: Path,
) -> None:
    system = _system(
        tmp_path,
        [100.0],
        [],
        worker_infrastructure_error=DependencyError(
            "PID namespace setup failed",
            "agent_process_containment_failed",
            {"stage": "candidate_generation"},
        ),
    )

    result = system[0].run(_spec(tmp_path, iterations=2))

    assert result.status is TaskStatus.INFRASTRUCTURE_ERROR
    assert result.reason_code == "agent_process_containment_failed"
    assert len(system[3].calls) == 2  # baseline and diagnostic; no final replay
    assert len(system[5].requests) == 1
    assert system[6].calls == 0
    assert system[8].requests == []
    assert system[9].requests == []
    events = EventJournal(tmp_path / "results" / "events" / "run.db").iter_events(
        result.run_id
    )
    assert not any(event.event_type == "e2e.execution_rejected" for event in events)
    assert not any(event.event_type == "decision" for event in events)
    recovered = RunController.recover(
        result.run_id,
        EventJournal(tmp_path / "results" / "events" / "run.db"),
        SnapshotStore(tmp_path / "results" / "state.snapshot.json"),
    )
    assert recovered.state.e2e is not None
    assert recovered.state.e2e.decisions == ()


def test_untyped_candidate_worker_exception_is_normalized_as_infrastructure(
    tmp_path: Path,
) -> None:
    system = _system(
        tmp_path,
        [100.0],
        [],
        worker_infrastructure_error=OSError("simulated exec failure"),
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.INFRASTRUCTURE_ERROR
    assert result.reason_code == "candidate_generation_infrastructure_failed"
    assert result.details["failure"]["evidence"] == {"error_type": "OSError"}
    assert len(system[3].calls) == 2  # baseline and diagnostic; no final replay
    events = EventJournal(tmp_path / "results" / "events" / "run.db").iter_events(
        result.run_id
    )
    assert not any(event.event_type == "e2e.execution_rejected" for event in events)
    assert not any(event.event_type == "decision" for event in events)


@pytest.mark.parametrize(
    ("reason", "capture_status", "has_containment"),
    (
        ("agent_process_cleanup_failed", "cleanup_failed", True),
        ("agent_process_containment_unverified", "complete", False),
    ),
)
def test_unverified_agent_teardown_is_recorded_then_terminates_infrastructure(
    tmp_path: Path,
    reason: str,
    capture_status: str,
    has_containment: bool,
) -> None:
    system = _system(
        tmp_path,
        [100.0],
        [],
        worker_returned_failure=reason,
    )

    result = system[0].run(_spec(tmp_path, iterations=2))

    assert result.status is TaskStatus.INFRASTRUCTURE_ERROR
    assert result.reason_code == reason
    assert len(system[3].calls) == 2  # baseline and diagnostic; no final replay
    evidence = result.details["failure"]["evidence"]
    assert evidence["capture_status"] == capture_status
    assert (evidence["process_containment"] is not None) is has_containment
    digest = evidence["candidate_manifest_receipt"]
    manifest_path = tmp_path / "results" / "artifacts" / "sha256" / digest[:2] / digest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_receipts"] == []
    assert manifest["frozen_sources"] == []
    events = EventJournal(tmp_path / "results" / "events" / "run.db").iter_events(
        result.run_id
    )
    event_types = [event.event_type for event in events]
    assert "agent_failed" in event_types
    assert "candidate_frozen" in event_types
    assert "e2e.execution_rejected" not in event_types
    assert "decision" not in event_types


def test_safety_snapshot_io_failure_returns_terminal_infrastructure_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_write(_destination, _source):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(candidate_snapshot, "_write_frozen_source", fail_write)
    system = _system(
        tmp_path,
        [100.0],
        [True],
        safety_override=NoToolSafetyVerifier(),
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.INFRASTRUCTURE_ERROR
    assert result.reason_code == "candidate_snapshot_materialization_failed"
    assert result.details["failure"]["evidence"] == {"error_type": "OSError"}
    assert len(system[3].calls) == 2  # baseline and diagnostic; no final replay
    assert system[8].requests == []
    assert system[9].requests == []
    events = EventJournal(tmp_path / "results" / "events" / "run.db").iter_events(
        result.run_id
    )
    assert not any(event.event_type == "e2e.execution_rejected" for event in events)
    assert not any(event.event_type == "decision" for event in events)


def test_returned_candidate_failure_remains_a_search_rejection(tmp_path: Path) -> None:
    system = _system(
        tmp_path,
        [100.0, 100.0],
        [],
        worker_returned_failure="undeclared_agent_edit",
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.NO_GAIN
    assert len(system[3].calls) == 4  # plus reward-ineligible terminal diagnostic
    assert system[6].calls == 0
    assert system[8].requests == []
    events = EventJournal(tmp_path / "results" / "events" / "run.db").iter_events(
        result.run_id
    )
    rejected = [event for event in events if event.event_type == "e2e.execution_rejected"]
    assert len(rejected) == 1
    assert rejected[0].payload["reason"] == "undeclared_agent_edit"
    recovered = RunController.recover(
        result.run_id,
        EventJournal(tmp_path / "results" / "events" / "run.db"),
        SnapshotStore(tmp_path / "results" / "state.snapshot.json"),
    )
    assert recovered.state.e2e is not None
    assert len(recovered.state.e2e.decisions) == 1
    assert recovered.state.e2e.decisions[0].verdict == "reject"


def test_source_free_agent_attempt_commits_explicit_no_source_reward(
    tmp_path: Path,
) -> None:
    system = _system(
        tmp_path,
        [100.0, 100.0],
        [],
        worker_returned_failure="agent_made_no_source_change",
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.NO_GAIN
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    decision = next(
        event for event in events if event.event_type == "e2e.candidate_decided"
    )
    reward = next(event for event in events if event.event_type == "reward_committed")
    assert decision.transaction_id == reward.transaction_id
    assert decision.payload["verdict"] == "reject"
    assert "candidate_id" not in decision.payload
    assert reward.payload["reward_vector"]["outcome_class"] == "no_source"
    assert reward.payload["scalar_reward"] == -20.0


def test_deployment_infrastructure_failure_is_not_candidate_no_gain(
    tmp_path: Path,
) -> None:
    system = _system(
        tmp_path,
        [100.0],
        [True],
        deployment_succeeds=False,
        deployment_infrastructure_failure=True,
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.INFRASTRUCTURE_ERROR
    assert result.reason_code == "deployment_infrastructure_failed"
    assert len(system[3].calls) == 2  # baseline and diagnostic; no final replay
    assert len(system[5].requests) == 1
    assert len(system[8].requests) == 1
    assert system[9].requests == []
    assert result.details["failure"]["evidence"]["deployment_reason_code"] == (
        "container_command_failed"
    )
    recovered = RunController.recover(
        result.run_id,
        EventJournal(tmp_path / "results" / "events" / "run.db"),
        SnapshotStore(tmp_path / "results" / "state.snapshot.json"),
    )
    assert recovered.state.phase.value == "failed"
    assert recovered.state.e2e is not None
    assert recovered.state.e2e.decisions == ()
    delivery_events = [
        event
        for event in EventJournal(
            tmp_path / "results" / "events" / "run.db"
        ).iter_events(result.run_id)
        if event.event_type == "delivery_result"
        and event.payload.get("attempt_id") == "attempt-1"
    ]
    assert len(delivery_events) == 1
    assert delivery_events[0].payload["infrastructure_failure"] is True


def test_candidate_measurement_rejects_serving_runtime_image_drift(
    tmp_path: Path,
) -> None:
    system = _system(
        tmp_path,
        [100.0, 102.0],
        [True],
        resolved_candidate_image_id="sha256:" + "0" * 64,
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.INFRASTRUCTURE_ERROR
    assert result.reason_code == "candidate_runtime_image_mismatch"
    assert system[8].rollbacks == ["candidate-attempt-1"]
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    assert not any(event.event_type == "e2e.candidate_decided" for event in events)
    assert not any(event.event_type == "reward_committed" for event in events)
    measurement = next(
        event
        for event in events
        if event.event_type == "measurement_result"
        and event.payload.get("attempt_id") == "attempt-1"
        and "-candidate" in str(event.payload.get("action_id", ""))
    )
    assert measurement.payload["resolved_image_id"] == "sha256:" + "0" * 64


def test_candidate_measurement_rejects_post_deployment_config_drift(
    tmp_path: Path,
) -> None:
    system = _system(
        tmp_path,
        [100.0, 102.0],
        [True],
        mutate_candidate_config=True,
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.INFRASTRUCTURE_ERROR
    assert result.reason_code == "candidate_runtime_config_mismatch"
    assert system[8].rollbacks == ["candidate-attempt-1"]
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    assert not any(event.event_type == "e2e.candidate_decided" for event in events)
    assert not any(event.event_type == "reward_committed" for event in events)


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
    assert not {"compile_result", "correctness_result"}.intersection(
        event.event_type for event in attempt_events
    )
    assert not any(
        event.event_type == "measurement_result"
        and event.payload.get("grade_policy_id")
        for event in attempt_events
    )
    deferred = [
        event
        for event in attempt_events
        if event.event_type == "tool_result"
        and event.payload.get("qualification_mode") == "e2e_quality_deferred"
    ]
    assert len(deferred) == 1
    assert deferred[0].payload["kernel_reward_available"] is False


@pytest.mark.parametrize(
    "transition",
    (
        "freeze_e2e_candidate",
        "commit_e2e_micro_verification",
        "commit_e2e_safety_verification",
        "commit_e2e_delivery_verification",
    ),
)
def test_resume_reconciles_recorded_candidate_gate_without_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, _, _, _, worker, micro, safety, deployments, _ = system
    original = getattr(RunController, transition)

    def crash(*_args, **_kwargs):
        raise RuntimeError(f"crash-before-{transition}")

    monkeypatch.setattr(RunController, transition, crash)
    with pytest.raises(RuntimeError, match="crash-before"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(RunController, transition, original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    assert len(worker.requests) == 1
    assert micro.calls == 1
    assert safety.calls == 1
    assert len(deployments.requests) == 1
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_rejects_interrupted_generation_without_reusing_agent_text(
    tmp_path: Path,
) -> None:
    class SimulatedProcessLoss(BaseException):
        pass

    system = _system(
        tmp_path,
        [100.0, 100.0],
        [],
        worker_infrastructure_error=SimulatedProcessLoss("agent-process-lost"),
    )
    use_case, _, _, _, _, worker, micro, safety, deployments, _ = system
    with pytest.raises(SimulatedProcessLoss, match="agent-process-lost"):
        use_case.run(_spec(tmp_path, iterations=1))
    worker.infrastructure_error = None

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.NO_GAIN
    assert len(worker.requests) == 1
    assert micro.calls == 0
    assert safety.calls == 0
    assert not deployments.requests
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(
        result.run_id
    )
    decision = next(
        event for event in events if event.event_type == "e2e.candidate_decided"
    )
    assert decision.payload["reason"] == "interrupted_candidate_generation"
    assert decision.payload.get("candidate_id") is None
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_reuses_recorded_candidate_measurement_and_commits_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, _, benchmark, diagnostics, *_ = system
    original = search_module.commit_measured_e2e_outcome

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash-after-candidate-measurement")

    monkeypatch.setattr(search_module, "commit_measured_e2e_outcome", crash)
    with pytest.raises(RuntimeError, match="candidate-measurement"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(search_module, "commit_measured_e2e_outcome", original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    assert len(
        [call for call in benchmark.calls if call.run_id.startswith("promotion-")]
    ) == 4
    assert diagnostics.calls == 3
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_after_atomic_keep_does_not_duplicate_reward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case = system[0]
    original = RunController.decide_e2e_candidate

    def commit_then_crash(controller, *args, **kwargs):
        original(controller, *args, **kwargs)
        raise RuntimeError("crash-after-atomic-decision")

    monkeypatch.setattr(RunController, "decide_e2e_candidate", commit_then_crash)
    with pytest.raises(RuntimeError, match="atomic-decision"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(RunController, "decide_e2e_candidate", original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    assert len(result.accepted_patch_ids) == 1
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_uses_uncommitted_reprofile_plan_from_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, _, _, diagnostics, *_ = system
    original = RunController.commit_e2e_reprofile

    def crash(*_args, **_kwargs):
        raise RuntimeError("crash-after-reprofile-plan")

    monkeypatch.setattr(RunController, "commit_e2e_reprofile", crash)
    with pytest.raises(RuntimeError, match="reprofile-plan"):
        use_case.run(_spec(tmp_path, iterations=1))
    assert diagnostics.calls == 2
    monkeypatch.setattr(RunController, "commit_e2e_reprofile", original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    assert diagnostics.calls == 3
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_completes_keep_update_after_reprofile_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, _, _, diagnostics, *_ = system
    original = RunController.complete_e2e_update

    def crash_keep_update(controller, *args, **kwargs):
        search = controller.state.e2e
        if (
            search is not None
            and search.stage is SearchStage.UPDATING
            and search.decisions
            and search.decisions[-1].verdict == "keep"
        ):
            raise RuntimeError("crash-after-reprofile-commit")
        return original(controller, *args, **kwargs)

    monkeypatch.setattr(
        RunController, "complete_e2e_update", crash_keep_update
    )
    with pytest.raises(RuntimeError, match="reprofile-commit"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(RunController, "complete_e2e_update", original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    assert diagnostics.calls == 3
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_retries_final_measurement_with_a_fresh_action_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedFinalProcessLoss(BaseException):
        pass

    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, _, benchmark, *_ = system
    original = benchmark.run_normalized

    def crash_final(request):
        if request.run_id == "final-measurement":
            raise SimulatedFinalProcessLoss("final-process-lost")
        return original(request)

    monkeypatch.setattr(benchmark, "run_normalized", crash_final)
    with pytest.raises(SimulatedFinalProcessLoss, match="final-process-lost"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(benchmark, "run_normalized", original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    assert any(
        call.run_id.startswith("final-measurement-resume-")
        for call in benchmark.calls
    )
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_from_planning_preserves_keep_chain_and_continues_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(
        tmp_path,
        [100.0, 102.0, 104.0, 104.0],
        [True, True],
    )
    use_case, _, _, _, diagnostics, worker, _, _, deployments, _ = system
    original = RunController.complete_e2e_update
    crashed = False

    def commit_then_crash(controller, *args, **kwargs):
        nonlocal crashed
        state = original(controller, *args, **kwargs)
        search = state.e2e
        if (
            not crashed
            and search is not None
            and search.stage is SearchStage.PLANNING
            and search.budget.candidates_used == 1
        ):
            crashed = True
            raise RuntimeError("crash-after-first-keep-update")
        return state

    monkeypatch.setattr(
        RunController, "complete_e2e_update", commit_then_crash
    )
    with pytest.raises(RuntimeError, match="first-keep-update"):
        use_case.run(_spec(tmp_path, iterations=2))
    monkeypatch.setattr(RunController, "complete_e2e_update", original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    assert len(result.accepted_patch_ids) == 2
    assert len(worker.requests) == 2
    assert diagnostics.calls == 4
    assert deployments.requests[0].accepted_stack == ()
    assert len(deployments.requests[1].accepted_stack) == 1
    assert deployments.requests[1].anchor_generation == 1
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(
        result.run_id
    )
    decisions = tuple(
        event for event in events if event.event_type == "e2e.candidate_decided"
    )
    rewards = tuple(
        event for event in events if event.event_type == "reward_committed"
    )
    assert len(decisions) == len(rewards) == 2
    assert {item.transaction_id for item in decisions} == {
        item.transaction_id for item in rewards
    }


def test_resume_rejects_mutated_accepted_deployment_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, _, _, _, _, _, _, deployments, _ = system
    original = RunController.decide_e2e_candidate

    def commit_then_crash(controller, *args, **kwargs):
        original(controller, *args, **kwargs)
        raise RuntimeError("crash-before-config-tamper")

    monkeypatch.setattr(RunController, "decide_e2e_candidate", commit_then_crash)
    with pytest.raises(RuntimeError, match="config-tamper"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(RunController, "decide_e2e_candidate", original)
    deployments.results[0].measurement_config.write_text(
        "benchmark: {docker_image: attacker:latest}\n", encoding="utf-8"
    )

    with pytest.raises(IntegrityError, match="config drifted"):
        use_case.resume(tmp_path / "results")


@pytest.mark.parametrize("crash_after", [1, 2, 3])
def test_resume_discards_partial_matched_window_and_starts_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after: int,
) -> None:
    class SimulatedPromotionLoss(BaseException):
        pass

    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, _, benchmark, *_ = system
    original = E2ERunRecord.record_benchmark
    completed = 0

    def commit_then_crash(record, action_id, *args, **kwargs):
        nonlocal completed
        evidence = original(record, action_id, *args, **kwargs)
        if action_id.startswith("promotion-"):
            completed += 1
            if completed == crash_after:
                raise SimulatedPromotionLoss("partial-matched-window")
        return evidence

    monkeypatch.setattr(E2ERunRecord, "record_benchmark", commit_then_crash)
    with pytest.raises(SimulatedPromotionLoss, match="partial-matched-window"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(E2ERunRecord, "record_benchmark", original)

    result = use_case.resume(tmp_path / "results")

    assert result.status is TaskStatus.SUCCEEDED
    promotion_calls = [
        call for call in benchmark.calls if call.run_id.startswith("promotion-")
    ]
    assert len(promotion_calls) == crash_after + 4
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    pairs = [
        event
        for event in events
        if event.payload.get("measurement_kind") == "matched_promotion_ab_ba"
    ]
    assert len(pairs) == 1
    window = pairs[0].payload["window_id"]
    assert sum(window in call.run_id for call in promotion_calls) == 4
    resume_lease = next(
        event
        for event in events
        if event.payload.get("kind") == "gpu_lease_resume"
    )
    assert pairs[0].payload["gpu_lease_digest"] == resume_lease.payload["lease_digest"]
    _assert_one_decision_reward_pair(tmp_path / "results", result.run_id)


def test_resume_rejects_changed_physical_gpu_before_recording_new_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessLoss(BaseException):
        pass

    system = _system(tmp_path, [100.0, 102.0, 102.0], [True])
    use_case, _, lease, benchmark, *_ = system
    lease.unique_ids = ("0x0000000000000001", "0x0000000000000002")
    original = benchmark.run_normalized

    def crash(request):
        if request.run_id.startswith("promotion-"):
            raise SimulatedProcessLoss("promotion-process-lost")
        return original(request)

    monkeypatch.setattr(benchmark, "run_normalized", crash)
    with pytest.raises(SimulatedProcessLoss, match="promotion-process-lost"):
        use_case.run(_spec(tmp_path, iterations=1))
    monkeypatch.setattr(benchmark, "run_normalized", original)

    with pytest.raises(ContractError) as raised:
        use_case.resume(tmp_path / "results")

    assert raised.value.reason_code == "resume_gpu_scope_mismatch"
    request = load_run_request(tmp_path / "results")
    assert request.gpu_device_scope.endswith("0000000000000001")
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(request.run_id)
    assert not any(event.payload.get("kind") == "gpu_lease_resume" for event in events)


def test_keep_requires_both_ab_and_ba_comparisons(tmp_path: Path) -> None:
    system = _system(
        tmp_path,
        [100.0, 100.0],
        [True],
        matched_measurements=[100.0, 102.0, 99.0, 100.0],
    )

    result = system[0].run(_spec(tmp_path, iterations=1))

    assert result.status is TaskStatus.NO_GAIN
    assert result.accepted_patch_ids == ()
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    pair = next(
        event
        for event in events
        if event.payload.get("measurement_kind") == "matched_promotion_ab_ba"
    )
    receipt = ArtifactReceipt.from_dict(
        next(
            item["receipt"]
            for item in pair.payload["artifacts"]
            if item["role"] == "matched_promotion_pair"
        )
    )
    document = json.loads(ArtifactStore(tmp_path / "results/artifacts").read_bytes(receipt))
    assert [item["keep"] for item in document["comparisons"]] == [True, False]
    assert document["verdict"]["keep"] is False
    decision = next(event for event in events if event.event_type == "e2e.candidate_decided")
    assert decision.payload["verdict"] == "revert"


def test_recovery_rejects_malicious_anchor_candidate_receipt_swap(
    tmp_path: Path,
) -> None:
    result = _system(tmp_path, [100.0, 102.0, 102.0], [True])[0].run(
        _spec(tmp_path, iterations=1)
    )
    request = load_run_request(tmp_path / "results")
    record = recover_record(request)
    events = EventJournal(tmp_path / "results/events/run.db").iter_events(result.run_id)
    pair = next(
        event
        for event in events
        if event.payload.get("measurement_kind") == "matched_promotion_ab_ba"
    )
    pair_receipt = ArtifactReceipt.from_dict(
        next(
            item["receipt"]
            for item in pair.payload["artifacts"]
            if item["role"] == "matched_promotion_pair"
        )
    )
    document = json.loads(record.artifacts.read_bytes(pair_receipt))
    observations = document["observations"]
    observations[0]["normalized_receipt"], observations[1]["normalized_receipt"] = (
        observations[1]["normalized_receipt"],
        observations[0]["normalized_receipt"],
    )
    forged_receipt = record.put_json(document)
    payload = dict(pair.payload)
    payload["artifacts"] = [
        {
            **item,
            "receipt": (
                forged_receipt.to_dict()
                if item["role"] == "matched_promotion_pair"
                else item["receipt"]
            ),
        }
        for item in pair.payload["artifacts"]
    ]
    forged_event = replace(pair, payload=payload)
    by_key: dict[str, EventRecord] = {event.idempotency_key: event for event in events}

    with pytest.raises(IntegrityError) as raised:
        recover_matched_promotion(
            record,
            pair_event=forged_event,
            events_by_key=by_key,
            protocol_hash=request.views.workload_semantics_sha256,
            policy=E2EAcceptancePolicy(request.spec.goal.gates),
            attempt_id=document["attempt_id"],
            candidate_id=document["candidate_id"],
            opportunity_id=document["opportunity_id"],
        )

    assert raised.value.reason_code == "promotion_receipt_mismatch"


def _assert_one_decision_reward_pair(root: Path, run_id: str) -> None:
    events = EventJournal(root / "events/run.db").iter_events(run_id)
    decisions = tuple(
        event for event in events if event.event_type == "e2e.candidate_decided"
    )
    rewards = tuple(
        event for event in events if event.event_type == "reward_committed"
    )
    assert len(decisions) == len(rewards) == 1
    assert decisions[0].transaction_id == rewards[0].transaction_id
