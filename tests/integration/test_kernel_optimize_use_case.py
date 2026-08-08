from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

import pytest

from apex.core import AgentBackendName, TaskStatus, sha256_bytes, sha256_json
from apex.delivery import load_and_verify_kernel_bundle
from apex.evaluation.safety import (
    CapabilityCheck,
    CapabilityStatus,
    ExecutionStatus,
    FindingStatus,
    RESULT_SCHEMA_VERSION,
    SafetyGateRequest,
    SafetyGateResult,
    SafetyRequirement,
    ToolCapability,
    ToolEvaluation,
    ToolPolicy,
    ToolRuntimeIdentity,
    ToolVerificationPlan,
    VerificationPolicy,
    decide_safety,
)
from apex.execution import AgentRegistry
from apex.intake import TaskSpec
from apex.optimization.kernel import KernelOptimizeRequest, KernelOptimizeUseCase
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentCaptureStatus,
    AgentCost,
    AgentInvocationReceipt,
    AgentProcessContainmentReceipt,
    AgentRequest,
    AgentResult,
    AgentSemanticEvent,
    AgentTranscriptEvent,
    AgentTerminationKind,
    AgentUsage,
    KernelMeasurementOutput,
    KernelMeasurementRequest,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)
from apex.runtime import GpuDeviceIdentity, GpuOwnershipReceipt, LocalGpuLeaseManager


def _agent_containment(*, stopped: bool = False) -> AgentProcessContainmentReceipt:
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
        termination_reason="stdout_budget_boundary" if stopped else "natural_exit",
        teardown_mode="pidfd_sigkill" if stopped else "natural_exit",
        pidfd_sigkill_sent=stopped,
        namespace_init_exit_verified=True,
        wrapper_exit_verified=True,
        wrapper_force_killed=False,
        terminal_status_verified=True,
        terminal_status_absent_after_sigkill=False,
        status_eof_verified=True,
        namespace_membership_scan_complete=True,
        live_namespace_members_after=(),
    )
from apex.rl import DatasetExportConfig, DatasetExporter, EpisodeGraphMaterializer
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal


class _FakeOwnershipInspector:
    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        return GpuOwnershipReceipt(
            1,
            "rocm_smi_process_gpu_map_v1",
            selector_scope,
            123,
            "/opt/rocm/lib/librocm_smi64.so.7",
            "a" * 64,
            (GpuDeviceIdentity(0, "0x0000000000000001", "/dev/dri/renderD128"),),
            (),
            (),
        )


def _gpu_leases(tmp_path: Path) -> LocalGpuLeaseManager:
    return LocalGpuLeaseManager(
        lock_root=tmp_path / "gpu-leases",
        ownership_inspector=_FakeOwnershipInspector(),
    )


class EditingAgent:
    name = AgentBackendName.CODEX

    def __init__(
        self,
        *,
        edit_harness: bool = False,
        make_change: bool = True,
        termination_kind: AgentTerminationKind = AgentTerminationKind.COMPLETED,
        capture_status: AgentCaptureStatus = AgentCaptureStatus.COMPLETE,
        ignored_artifact_marker: Path | None = None,
    ) -> None:
        self.edit_harness = edit_harness
        self.make_change = make_change
        self.termination_kind = termination_kind
        self.capture_status = capture_status
        self.ignored_artifact_marker = ignored_artifact_marker

    def run(self, request: AgentRequest) -> AgentResult:
        if self.make_change:
            (request.workspace / "source" / "kernel.py").write_text(
                "def kernel(x):\n    return x + 0\n", encoding="utf-8"
            )
        if self.edit_harness:
            (request.workspace / "harness.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
        if self.ignored_artifact_marker is not None:
            poison = request.workspace / "sitecustomize.py"
            poison.write_text(
                "from pathlib import Path\n"
                f"Path({str(self.ignored_artifact_marker)!r}).write_text('executed')\n",
                encoding="utf-8",
            )
            py_compile.compile(
                str(poison),
                cfile=str(request.workspace / "sitecustomize.pyc"),
                doraise=True,
            )
            poison.unlink()
        boundary = self.termination_kind is AgentTerminationKind.EXACT_TURN_BOUNDARY
        overrun = self.termination_kind is AgentTerminationKind.TURN_OVERRUN
        return AgentResult(
            backend=self.name,
            model=request.model,
            exit_code=137 if boundary or overrun else 0,
            timed_out=False,
            events=(),
            stdout='{"type":"turn.completed"}\n',
            stderr="",
            duration_seconds=0.1,
            invocation=_agent_invocation(request),
            termination_kind=self.termination_kind,
            capture_status=self.capture_status,
            termination_reason=(
                "max_turns_exact_boundary"
                if boundary
                else "max_turns_overrun" if overrun else None
            ),
            observed_turns=(
                request.max_turns if boundary else request.max_turns + 1 if overrun else 1
            ),
            observer_stop_sent=boundary or overrun,
            process_containment=_agent_containment(stopped=boundary or overrun),
        )


def _agent_invocation(request: AgentRequest) -> AgentInvocationReceipt:
    return AgentInvocationReceipt(
        cli_name="codex",
        cli_version="test",
        executable_path="/usr/bin/codex",
        resolved_executable_path="/usr/bin/codex",
        entrypoint_sha256="a" * 64,
        argv=("codex", "exec"),
        workspace=str(request.workspace),
        prompt_transport="stdin",
        requested_allowed_files=request.allowed_files,
        allowed_files_enforced_by_cli=False,
        max_turns=request.max_turns,
        turn_policy=STRUCTURED_TURN_CHECKPOINT_POLICY,
        process_containment_policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        isolation=(("sandbox", "workspace-write"),),
    )


class SequencedEditingAgent:
    name = AgentBackendName.CODEX

    def __init__(self, sources: tuple[str, ...]) -> None:
        self.sources = sources
        self.requests: list[AgentRequest] = []
        self.report_visible_before_edit: list[bool] = []

    def run(self, request: AgentRequest) -> AgentResult:
        index = len(self.requests)
        self.requests.append(request)
        self.report_visible_before_edit.append(
            (request.workspace / "build" / "timings.json").exists()
        )
        (request.workspace / "source" / "kernel.py").write_text(
            self.sources[index], encoding="utf-8"
        )
        return AgentResult(
            backend=self.name,
            model=request.model,
            exit_code=0,
            timed_out=False,
            events=(
                AgentTranscriptEvent(
                    kind="turn.completed",
                    metadata={
                        "type": "turn.completed",
                        "usage": {"input_tokens": index + 1},
                    },
                ),
            ),
            stdout='{"type":"turn.completed"}\n',
            stderr="",
            duration_seconds=0.1,
            semantic_events=(
                AgentSemanticEvent(
                    index=0,
                    source_event_index=0,
                    source_kind="turn.completed",
                    kind="agent_message",
                    role="assistant",
                    text="candidate written",
                ),
                AgentSemanticEvent(
                    index=1,
                    source_event_index=0,
                    source_kind="turn.completed",
                    kind="tool_called",
                    tool_name="edit",
                    tool_call_id="tool-1",
                ),
                AgentSemanticEvent(
                    index=2,
                    source_event_index=0,
                    source_kind="turn.completed",
                    kind="tool_result",
                    tool_name="edit",
                    tool_call_id="tool-1",
                    succeeded=True,
                ),
            ),
            usage=AgentUsage(
                input_tokens=index + 1,
                output_tokens=2,
                total_tokens=index + 3,
                turn_count=1,
                tool_call_count=1,
                source_event_indices=(0,),
            ),
            cost=AgentCost(
                amount="0.125",
                currency="USD",
                source_event_index=0,
                source_key="total_cost_usd",
            ),
            invocation=_agent_invocation(request),
            process_containment=_agent_containment(),
        )


def _digest(label: str) -> str:
    return sha256_bytes(label.encode())


def _measurement_report(
    reference: float,
    optimized: float,
    samples: int,
    *,
    method_sha256: str = "1" * 64,
) -> dict[str, object]:
    per_block = tuple(
        samples // 4 + (1 if index < samples % 4 else 0) for index in range(4)
    )
    health = {
        "device": "gfx950:0",
        "healthy": True,
        "temperature_c": 45.0,
        "clock_mhz": 2100.0,
    }
    implementations = (
        "reference", "optimized", "optimized", "reference",
        "optimized", "reference", "reference", "optimized",
    )
    seen = {"reference": 0, "optimized": 0}
    order: list[tuple[str, float, int]] = []
    for implementation in implementations:
        index = seen[implementation]
        seen[implementation] += 1
        latency = reference if implementation == "reference" else optimized
        order.append((implementation, latency, per_block[index]))
    return {
        "schema": "apex.kernel-measurement/v1",
        "policy_id": "kernel_invocation_nearest_rank_v1",
        "sample_unit": "kernel_invocation",
        "quantile_method": "nearest_rank_v1",
        "timer": "hip_event",
        "timer_resolution_ns": 1.0,
        "inner_repeats": 1,
        "measurement_method_sha256": method_sha256,
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
                        "samples_ms": [latency] * count,
                        "invalid_sample_counts": {},
                        "gpu_health_before": health,
                        "gpu_health_after": health,
                    }
                    for index, (implementation, latency, count) in enumerate(order)
                ],
            }
        ],
    }


class FixtureMeasurementEvaluator:
    adapter_id = "fixture-evaluator-v1"

    def __init__(
        self,
        values: tuple[float, float, int] | None = None,
        *,
        dynamic: bool = False,
        writer_id: str | None = None,
        measurement_method_sha256: str = "1" * 64,
        report_method_sha256: str | None = None,
        mutate_harness: bool = False,
    ) -> None:
        self.values = values
        self.dynamic = dynamic
        self.writer_id = writer_id or self.adapter_id
        self.measurement_method_sha256 = measurement_method_sha256
        self.report_method_sha256 = report_method_sha256
        self.mutate_harness = mutate_harness
        self.requests: list[KernelMeasurementRequest] = []

    def measure(self, request: KernelMeasurementRequest) -> KernelMeasurementOutput:
        self.requests.append(request)
        if self.mutate_harness:
            request.harness_paths[0].write_text("raise SystemExit(0)\n", encoding="utf-8")
        if self.dynamic:
            source = (request.candidate_root / "source" / "kernel.py").read_text()
            marker = "SPEED_MS = "
            optimized = float(source.split(marker, 1)[1].splitlines()[0])
            values = (10.0, optimized, 300)
        else:
            assert self.values is not None
            values = self.values
        report = _measurement_report(
            *values,
            method_sha256=(
                self.report_method_sha256 or self.measurement_method_sha256
            ),
        )
        request.report_path.write_text(json.dumps(report), encoding="utf-8")
        return KernelMeasurementOutput(self.writer_id, request.report_path)


def _task(
    tmp_path: Path,
    *,
    performance_marker: Path | None = None,
    measurement_values: tuple[float, float, int] | None = None,
    dynamic_measurement: bool = False,
    candidate_forges_report: bool = False,
    max_iterations: int = 1,
    external_evaluator: bool = True,
) -> TaskSpec:
    workspace = tmp_path / "workspace"
    (workspace / "source").mkdir(parents=True)
    (workspace / "source" / "kernel.py").write_text(
        "def kernel(x):\n    return x\n", encoding="utf-8"
    )
    (workspace / "harness.py").write_text("assert True\n", encoding="utf-8")
    success = [sys.executable, "-c", "print('ok')"]
    compile_command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            "source=Path('source/kernel.py').read_text(); "
            "raise SystemExit(1 if 'COMPILE_FAIL' in source else 0)"
        ),
    ]
    performance = success
    if performance_marker is not None:
        performance = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                f"Path({str(performance_marker)!r}).write_text('normal-runtime')"
            ),
        ]
    if candidate_forges_report:
        report = _measurement_report(100.0, 0.1, 300)
        performance = [
            sys.executable,
            "-c",
            (
                "import json; from pathlib import Path; "
                "Path('build').mkdir(); "
                f"Path('build/timings.json').write_text(json.dumps({report!r}))"
            ),
        ]
    data = {
            "task_id": "kernel-task",
            "workspace": str(workspace),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize kernel without changing its result",
            "language": "triton",
            "editable_files": ["source/kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                "compile": {"argv": compile_command},
                "correctness": {"argv": success},
                "performance": {"argv": performance},
            },
        "budget": {"max_iterations": max_iterations},
    }
    if external_evaluator:
        data["recipe"] = {
            "kind": "python_triton",
            "recipe_id": "external-central-evaluator-v1",
            "sha256": "e" * 64,
            "provenance": "external_evaluator",
        }
    if measurement_values is not None or dynamic_measurement or candidate_forges_report:
        data["measurement"] = {
            "schema": "apex.kernel-measurement/v1",
            "adapter_id": "fixture-evaluator-v1",
            "harness_files": ["harness.py"],
            "measurement_method_sha256": "1" * 64,
            "runner": {"argv": [sys.executable, "harness.py"]},
            "aggregation": "equal_case",
        }
    return TaskSpec.from_mapping(data)


def _run(
    tmp_path: Path,
    agent: EditingAgent,
    *,
    safety_gate=None,
    safety_policy: VerificationPolicy | None = None,
    safety_tools: tuple[ToolVerificationPlan, ...] = (),
    performance_marker: Path | None = None,
    measurement_values: tuple[float, float, int] | None = None,
):
    task = _task(
        tmp_path,
        performance_marker=performance_marker,
        measurement_values=measurement_values,
    )
    result_json = tmp_path / "machine" / "result.json"
    use_case = KernelOptimizeUseCase(
        agents=AgentRegistry([agent], default=AgentBackendName.CODEX),
        safety_gate=safety_gate,
        safety_policy=safety_policy,
        safety_tools=safety_tools,
        gpu_leases=_gpu_leases(tmp_path),
        measurement_evaluator=(
            FixtureMeasurementEvaluator(measurement_values)
            if measurement_values is not None
            else None
        ),
    )
    result = use_case.run(KernelOptimizeRequest(task=task, result_json=result_json))
    return task, result, result_json


def _run_sequence(
    tmp_path: Path,
    sources: tuple[str, ...],
    *,
    max_iterations: int,
    dynamic_measurement: bool = True,
):
    agent = SequencedEditingAgent(sources)
    task = _task(
        tmp_path,
        dynamic_measurement=dynamic_measurement,
        max_iterations=max_iterations,
    )
    result_json = tmp_path / "machine" / "result.json"
    use_case = KernelOptimizeUseCase(
        agents=AgentRegistry([agent], default=AgentBackendName.CODEX),
        measurement_evaluator=(
            FixtureMeasurementEvaluator(dynamic=True)
            if dynamic_measurement
            else None
        ),
        gpu_leases=_gpu_leases(tmp_path),
    )
    result = use_case.run(KernelOptimizeRequest(task=task, result_json=result_json))
    run_root = next((task.results_dir / "runs").iterdir())
    events = EventJournal(run_root / "events" / "run.db").iter_events(run_root.name)
    return agent, task, result, run_root, events


def test_use_case_never_modifies_input_and_emits_verified_source_bundle(tmp_path: Path) -> None:
    task, result, result_json = _run(tmp_path, EditingAgent())

    assert result.status is TaskStatus.CANDIDATE_READY
    assert result.reason_code == "candidate_deferred_to_external_evaluator"
    assert result.reward is None
    assert result.applied is False
    assert result.external_verification_required is True
    assert (task.workspace / "source" / "kernel.py").read_text().endswith("return x\n")
    bundle = load_and_verify_kernel_bundle(Path(result.bundle_path), expected_digest=result.bundle_digest)
    assert bundle.changed_files == ("source/kernel.py",)
    assert json.loads(result_json.read_text())["status"] == "candidate_ready"
    assert result.safety_status == "not_configured"
    assert result.safety_certified is False
    assert result.safety_result_fingerprint is not None
    assert result.safety_receipt_digest is not None
    assert result.run_id is not None
    assert result.baseline_lock is not None
    assert result.baseline_lock["resolution_hash"] == bundle.manifest["baseline"]["resolution_hash"]
    assert result.baseline_lock["file_hashes"] == bundle.manifest["baseline"]["file_hashes"]
    assert result.internal_verdict == "keep"
    assert result.internal_verdict_ref is not None
    assert len(result.verification_summary_refs) == 1
    assert result.event_journal_ref is not None
    assert result.artifact_store_ref is not None
    assert result.gpu_lease is not None
    assert result.gpu_lease["run_id"] == result.run_id
    assert result.gpu_lease_receipt_digest == sha256_json(result.gpu_lease)
    assert result.gpu_lease_receipt_digest in result.artifact_store_ref["receipt_digests"]
    assert result.error is None
    serialized = json.loads(result_json.read_text())
    assert serialized["run_id"] == result.run_id
    assert serialized["internal_verdict_ref"] == result.internal_verdict_ref
    assert serialized["gpu_lease"] == json.loads(json.dumps(result.gpu_lease))
    assert serialized["gpu_lease_receipt_digest"] == result.gpu_lease_receipt_digest

    run_root = next((task.results_dir / "runs").iterdir())
    for projection in (
        "report.json",
        "report.md",
        "replication_guide.json",
        "replication_guide.md",
    ):
        assert (run_root / projection).is_file()
    journal = EventJournal(run_root / "events" / "run.db")
    events = journal.iter_events(run_root.name)
    assert not any(item.event_type == "experience.measured" for item in events)
    deferred = [item for item in events if item.event_type == "experience.deferred"]
    assert len(deferred) == 1
    assert deferred[0].payload["evidence_class"] == "derived"
    assert deferred[0].payload["status"] == "pending_external_evaluator"
    assert deferred[0].payload["external_verification_required"] is True
    assert "outcome" not in deferred[0].payload
    graph = EpisodeGraphMaterializer(
        journal, ArtifactStore(run_root / "artifacts")
    ).materialize(run_root.name)
    child = graph.children[0]
    assert child.context_packet_id is not None
    assert child.trainability == "truncated"
    assert child.validation_reasons == ("external_evaluation_pending",)
    assert child.verdict == "keep"
    event_types = [item.event_type for item in child.events]
    assert event_types.index("context_packet_created") < event_types.index("prompt_sent")
    assert event_types.index("prompt_sent") < event_types.index("agent_completed")
    assert event_types.index("agent_completed") < event_types.index("candidate_frozen")
    assert event_types.index("candidate_frozen") < event_types.index("compile_result")
    assert event_types.index("compile_result") < event_types.index("correctness_result")
    assert event_types.index("correctness_result") < event_types.index("safety_result")
    assert event_types.index("safety_result") < event_types.index("performance_command_result")


def test_command_success_without_measurement_authority_never_becomes_candidate_ready(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, external_evaluator=False)
    result = KernelOptimizeUseCase(
        agents=AgentRegistry([EditingAgent()], default=AgentBackendName.CODEX),
        gpu_leases=_gpu_leases(tmp_path),
    ).run(
        KernelOptimizeRequest(
            task=task,
            result_json=tmp_path / "machine" / "result.json",
        )
    )

    assert result.status is TaskStatus.NO_MEASUREMENT
    assert result.reason_code == "measurement_contract_missing"
    assert result.bundle_path is None
    assert result.reward is None


def test_use_case_rejects_harness_tampering_without_bundle(tmp_path: Path) -> None:
    _, result, _ = _run(tmp_path, EditingAgent(edit_harness=True))

    assert result.status is TaskStatus.REJECTED
    assert result.reason_code == "undeclared_agent_edit"
    assert result.bundle_path is None
    assert result.run_id is not None
    assert result.baseline_lock is not None
    assert result.internal_verdict == "reject"
    assert result.internal_verdict_ref is not None
    assert result.event_journal_ref is not None
    assert result.artifact_store_ref is not None
    assert result.error is not None
    assert result.error["reason_code"] == "undeclared_agent_edit"
    assert result.error["details"] == {"paths": ["harness.py"]}


def test_use_case_reports_no_gain_for_unchanged_candidate(tmp_path: Path) -> None:
    _, result, _ = _run(tmp_path, EditingAgent(make_change=False))

    assert result.status is TaskStatus.NO_GAIN
    assert result.bundle_path is None


def test_use_case_rejects_agent_turn_overrun(tmp_path: Path) -> None:
    task, result, _ = _run(
        tmp_path,
        EditingAgent(termination_kind=AgentTerminationKind.TURN_OVERRUN),
    )

    assert result.status is TaskStatus.BUDGET_EXHAUSTED
    assert result.reason_code == "agent_turn_budget_overrun"
    assert result.bundle_path is None
    run_root = next((task.results_dir / "runs").iterdir())
    events = EventJournal(run_root / "events" / "run.db").iter_events(run_root.name)
    failed = next(event for event in events if event.event_type == "agent_failed")
    assert failed.payload["termination_kind"] == "turn_overrun"
    assert failed.payload["candidate_capture_allowed"] is False
    assert failed.payload["observed_turns"] == 26


def test_exact_turn_boundary_candidate_runs_all_trusted_gates(tmp_path: Path) -> None:
    task, result, _ = _run(
        tmp_path,
        EditingAgent(termination_kind=AgentTerminationKind.EXACT_TURN_BOUNDARY),
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert result.bundle_path is not None
    run_root = next((task.results_dir / "runs").iterdir())
    events = EventJournal(run_root / "events" / "run.db").iter_events(run_root.name)
    event_types = [event.event_type for event in events]
    completed = next(event for event in events if event.event_type == "agent_completed")
    assert completed.payload["termination_kind"] == "exact_turn_boundary"
    assert completed.payload["capture_status"] == "complete"
    assert completed.payload["candidate_capture_allowed"] is True
    assert completed.payload["observed_turns"] == 25
    binding = next(
        item
        for item in completed.payload["artifacts"]
        if item["role"] == "agent_transcript"
    )
    receipt = ArtifactReceipt.from_dict(binding["receipt"])
    transcript = json.loads(ArtifactStore(run_root / "artifacts").read_bytes(receipt))
    assert transcript["schema"] == "apex.agent-transcript/v3"
    termination = transcript["termination"]
    assert termination["kind"] == "exact_turn_boundary"
    assert termination["reason"] == "max_turns_exact_boundary"
    assert termination["candidate_capture_allowed"] is True
    assert termination["observer_stop_sent"] is True
    assert termination["process_containment"]["policy_id"] == (
        AGENT_PROCESS_CONTAINMENT_POLICY
    )
    assert termination["process_containment"]["namespace_empty_verified"] is True
    assert termination["discarded_stdout_tail"] == {
        "lines": 0,
        "bytes": 0,
        "sha256": None,
    }
    assert termination["observed_turns"] == 25
    assert termination["max_turns"] == 25
    assert termination["turn_policy"] == STRUCTURED_TURN_CHECKPOINT_POLICY
    assert event_types.index("agent_completed") < event_types.index("candidate_frozen")
    assert event_types.index("candidate_frozen") < event_types.index("compile_result")
    assert event_types.index("compile_result") < event_types.index("correctness_result")
    assert event_types.index("correctness_result") < event_types.index("safety_result")
    assert event_types.index("safety_result") < event_types.index(
        "performance_command_result"
    )


def test_exact_turn_boundary_without_source_change_is_not_delivered(tmp_path: Path) -> None:
    _, result, _ = _run(
        tmp_path,
        EditingAgent(
            make_change=False,
            termination_kind=AgentTerminationKind.EXACT_TURN_BOUNDARY,
        ),
    )

    assert result.status is TaskStatus.NO_GAIN
    assert result.reason_code == "agent_made_no_source_change"
    assert result.bundle_path is None


def test_evaluator_never_executes_agent_generated_ignored_bytecode(tmp_path: Path) -> None:
    marker = tmp_path / "poison-executed"
    task, result, _ = _run(
        tmp_path,
        EditingAgent(
            termination_kind=AgentTerminationKind.EXACT_TURN_BOUNDARY,
            ignored_artifact_marker=marker,
        ),
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert not marker.exists()
    run_root = next((task.results_dir / "runs").iterdir())
    projection = next((run_root / "projections").iterdir())
    assert not (projection / "sitecustomize.pyc").exists()


class _SafetyOutcomePort:
    def __init__(self, finding: FindingStatus) -> None:
        self.finding = finding
        self.calls: list[SafetyGateRequest] = []

    def evaluate(self, request: SafetyGateRequest) -> SafetyGateResult:
        self.calls.append(request)
        execution = (
            ExecutionStatus.TOOL_ERROR
            if self.finding is FindingStatus.INCONCLUSIVE
            else ExecutionStatus.COMPLETED
        )
        evaluation = ToolEvaluation(
            tool="gpu_asan",
            capability=CapabilityStatus.READY,
            execution=execution,
            finding=self.finding,
            reason_codes=(
                (
                    "fixture_finding"
                    if self.finding is FindingStatus.FOUND
                    else "fixture_inconclusive"
                ),
            ),
        )
        evaluations = (evaluation,)
        decision = decide_safety(evaluations, policy=request.policy)
        plan = request.plan
        return SafetyGateResult(
            schema_version=RESULT_SCHEMA_VERSION,
            run_id=plan.run_id,
            candidate_id=plan.candidate_id,
            anchor_generation=plan.anchor_generation,
            plan_fingerprint=plan.fingerprint,
            policy_fingerprint=plan.policy_fingerprint,
            source_digest=plan.source_digest,
            candidate_digest=plan.candidate_digest,
            deployed_digest=plan.deployed_digest,
            isolation_receipt_fingerprint=request.isolation_receipt.fingerprint,
            evaluations=evaluations,
            decision=decision,
        )


def _configured_safety(
    requirement: SafetyRequirement = SafetyRequirement.ADVISORY,
) -> tuple[VerificationPolicy, tuple[ToolVerificationPlan, ...]]:
    policy = VerificationPolicy(
        rules=(
            ToolPolicy(
                tool="gpu_asan",
                requirement=requirement,
                qualified=requirement is SafetyRequirement.REQUIRED,
                qualification_digest=(
                    _digest("gpu-asan-qualification")
                    if requirement is SafetyRequirement.REQUIRED
                    else None
                ),
            ),
        )
    )
    ready = CapabilityCheck(CapabilityStatus.READY)
    tool = ToolVerificationPlan(
        identity=ToolRuntimeIdentity(
            tool="gpu_asan",
            version="fixture-1",
            plugin_digest=_digest("plugin"),
            runtime_image_id=f"sha256:{_digest('image')}",
            helper_digest=_digest("helper"),
            dispatch_digest=_digest("dispatch"),
        ),
        capability=ToolCapability(
            tool="gpu_asan",
            engine=ready,
            adapter=ready,
            runtime=ready,
        ),
        argv=(sys.executable, "-m", "trusted_fixture"),
        cases=("case-1",),
        positive_control_digest=_digest("positive-control"),
    )
    return policy, (tool,)


def test_confirmed_safety_finding_skips_normal_performance(tmp_path: Path) -> None:
    policy, tools = _configured_safety()
    safety = _SafetyOutcomePort(FindingStatus.FOUND)
    marker = tmp_path / "performance-ran"

    task, result, _ = _run(
        tmp_path,
        EditingAgent(),
        safety_gate=safety,
        safety_policy=policy,
        safety_tools=tools,
        performance_marker=marker,
    )

    assert len(safety.calls) == 1
    assert result.status is TaskStatus.REJECTED
    assert result.reason_code == "confirmed_safety_finding"
    assert result.safety_status == "rejected_finding"
    assert not result.safety_certified
    assert not marker.exists()
    run_root = next((task.results_dir / "runs").iterdir())
    event_types = [
        item.event_type
        for item in EventJournal(run_root / "events" / "run.db").iter_events(
            run_root.name
        )
    ]
    assert event_types.index("correctness_result") < event_types.index("safety_result")
    assert "performance_command_result" not in event_types
    assert "measurement_result" not in event_types


def test_advisory_inconclusive_runs_normal_performance_but_is_uncertified(
    tmp_path: Path,
) -> None:
    policy, tools = _configured_safety()
    safety = _SafetyOutcomePort(FindingStatus.INCONCLUSIVE)
    marker = tmp_path / "performance-ran"

    task, result, result_json = _run(
        tmp_path,
        EditingAgent(),
        safety_gate=safety,
        safety_policy=policy,
        safety_tools=tools,
        performance_marker=marker,
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert result.safety_status == "advisory_incomplete"
    assert not result.safety_certified
    assert marker.read_text() == "normal-runtime"
    machine = json.loads(result_json.read_text())
    assert machine["safety_status"] == "advisory_incomplete"
    assert machine["safety_certified"] is False
    run_root = next((task.results_dir / "runs").iterdir())
    event_types = [
        item.event_type
        for item in EventJournal(run_root / "events" / "run.db").iter_events(
            run_root.name
        )
    ]
    assert event_types.index("safety_result") < event_types.index("performance_command_result")


def test_raw_measurement_commits_robust_reward_and_rl_receipts(tmp_path: Path) -> None:
    task, result, _ = _run(
        tmp_path,
        EditingAgent(),
        measurement_values=(10.0, 8.0, 300),
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert result.measurement_status == "valid"
    assert result.s50 == 1.25
    assert result.s99 == 1.25
    assert result.srobust == 1.25
    assert result.reward == 170.0
    run_root = next((task.results_dir / "runs").iterdir())
    journal = EventJournal(run_root / "events" / "run.db")
    events = journal.iter_events(run_root.name)
    event_types = [item.event_type for item in events]
    assert event_types.index("performance_command_result") < event_types.index(
        "measurement_result"
    )
    assert event_types.index("measurement_result") < event_types.index(
        "reward_committed"
    )
    graph = EpisodeGraphMaterializer(
        journal,
        ArtifactStore(run_root / "artifacts"),
    ).materialize(run_root.name)
    child = graph.children[0]
    assert child.scalar_reward == 170.0
    assert child.policy_ids == ("kernel_robust_v1",)
    assert child.trainability == "complete"
    measured = next(event for event in events if event.event_type == "measurement_result")
    bindings = {item["role"]: item["receipt"] for item in measured.payload["artifacts"]}
    assert {"raw_measurement", "measurement_execution", "harness", "kernel_grade"} <= set(bindings)
    execution = json.loads(
        ArtifactStore(run_root / "artifacts").read_bytes(
            ArtifactReceipt.from_dict(bindings["measurement_execution"])
        )
    )
    assert execution["schema"] == "apex.kernel-measurement-execution/v1"
    assert execution["writer_kind"] == "trusted_evaluator_adapter"
    assert execution["writer_id"] == "fixture-evaluator-v1"
    assert execution["phase"] == "measurement"
    assert execution["harness_sha256"] == measured.payload[
        "measurement_harness_sha256"
    ]
    assert execution["phase_started_monotonic_ns"] <= execution[
        "adapter_returned_monotonic_ns"
    ] <= execution["output_observed_monotonic_ns"] <= execution[
        "phase_completed_monotonic_ns"
    ]
    exported = DatasetExporter(ArtifactStore(run_root / "artifacts")).export(
        graph,
        tmp_path / "rl-export",
        config=DatasetExportConfig(include_sft=False),
    )
    assert exported.record_count == 1
    transition = json.loads(
        (tmp_path / "rl-export" / "dataset.jsonl").read_text(encoding="utf-8")
    )
    assert transition["reward"]["scalar"] == 170.0
    assert transition["reward"]["vector"]["kernel_srobust"] == 1.25


def test_candidate_written_measurement_cannot_create_tampering_pass_or_reward(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        candidate_forges_report=True,
    )
    result_json = tmp_path / "machine" / "result.json"
    use_case = KernelOptimizeUseCase(
        agents=AgentRegistry([EditingAgent()], default=AgentBackendName.CODEX),
        gpu_leases=_gpu_leases(tmp_path),
    )

    result = use_case.run(KernelOptimizeRequest(task=task, result_json=result_json))

    assert result.status is TaskStatus.NO_MEASUREMENT
    assert result.reason_code == "measurement_evaluator_unavailable"
    assert result.reward is None
    assert result.bundle_path is None
    run_root = next((task.results_dir / "runs").iterdir())
    events = EventJournal(run_root / "events" / "run.db").iter_events(run_root.name)
    assert any(event.event_type == "performance_command_result" for event in events)
    assert not any(event.event_type == "reward_committed" for event in events)
    error = next(event for event in events if event.event_type == "measurement_result")
    assert error.payload["reason_code"] == "measurement_evaluator_unavailable"
    assert "tampering_passed" not in error.payload
    forged = next((run_root / "projections").rglob("timings.json"))
    assert forged.is_file()


def test_candidate_report_is_ignored_when_trusted_evaluator_is_bound(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        measurement_values=(10.0, 8.0, 300),
        candidate_forges_report=True,
    )
    evaluator = FixtureMeasurementEvaluator((10.0, 8.0, 300))
    use_case = KernelOptimizeUseCase(
        agents=AgentRegistry([EditingAgent()], default=AgentBackendName.CODEX),
        measurement_evaluator=evaluator,
        gpu_leases=_gpu_leases(tmp_path),
    )

    result = use_case.run(
        KernelOptimizeRequest(task=task, result_json=tmp_path / "machine" / "result.json")
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert result.srobust == 1.25
    assert result.reward == 170.0
    assert len(evaluator.requests) == 1
    request = evaluator.requests[0]
    assert not request.report_path.is_relative_to(request.candidate_root)


@pytest.mark.parametrize(
    ("evaluator", "reason_code"),
    [
        (
            FixtureMeasurementEvaluator(
                (10.0, 8.0, 300), writer_id="candidate-self-report"
            ),
            "measurement_writer_mismatch",
        ),
        (
            FixtureMeasurementEvaluator(
                (10.0, 8.0, 300), measurement_method_sha256="2" * 64
            ),
            "measurement_method_mismatch",
        ),
        (
            FixtureMeasurementEvaluator(
                (10.0, 8.0, 300), report_method_sha256="2" * 64
            ),
            "measurement_method_mismatch",
        ),
        (
            FixtureMeasurementEvaluator(
                (10.0, 8.0, 300), mutate_harness=True
            ),
            "measurement_harness_changed",
        ),
    ],
)
def test_measurement_authority_mismatch_never_commits_reward(
    tmp_path: Path,
    evaluator: FixtureMeasurementEvaluator,
    reason_code: str,
) -> None:
    task = _task(tmp_path, measurement_values=(10.0, 8.0, 300))
    use_case = KernelOptimizeUseCase(
        agents=AgentRegistry([EditingAgent()], default=AgentBackendName.CODEX),
        measurement_evaluator=evaluator,
        gpu_leases=_gpu_leases(tmp_path),
    )

    result = use_case.run(
        KernelOptimizeRequest(task=task, result_json=tmp_path / "machine" / "result.json")
    )

    assert result.status is TaskStatus.NO_MEASUREMENT
    assert result.reason_code == reason_code
    assert result.reward is None
    run_root = next((task.results_dir / "runs").iterdir())
    events = EventJournal(run_root / "events" / "run.db").iter_events(run_root.name)
    assert not any(event.event_type == "reward_committed" for event in events)
    if evaluator.measurement_method_sha256 != "1" * 64:
        assert evaluator.requests == []


def test_299_samples_return_no_measurement_without_reward(tmp_path: Path) -> None:
    task, result, _ = _run(
        tmp_path,
        EditingAgent(),
        measurement_values=(10.0, 8.0, 299),
    )

    assert result.status is TaskStatus.NO_MEASUREMENT
    assert result.measurement_status == "insufficient_samples"
    assert result.reward is None
    assert result.bundle_path is None
    run_root = next((task.results_dir / "runs").iterdir())
    event_types = [
        item.event_type
        for item in EventJournal(run_root / "events" / "run.db").iter_events(
            run_root.name
        )
    ]
    assert "measurement_result" in event_types
    assert "reward_committed" not in event_types


def test_valid_but_slower_candidate_is_no_gain_with_training_reward(tmp_path: Path) -> None:
    _, result, _ = _run(
        tmp_path,
        EditingAgent(),
        measurement_values=(8.0, 10.0, 300),
    )

    assert result.status is TaskStatus.NO_GAIN
    assert result.measurement_status == "valid"
    assert result.srobust == 0.8
    assert result.reward == 80.0
    assert result.bundle_path is None


def test_required_inconclusive_safety_skips_normal_performance(tmp_path: Path) -> None:
    policy, tools = _configured_safety(SafetyRequirement.REQUIRED)
    safety = _SafetyOutcomePort(FindingStatus.INCONCLUSIVE)
    marker = tmp_path / "performance-ran"

    _, result, _ = _run(
        tmp_path,
        EditingAgent(),
        safety_gate=safety,
        safety_policy=policy,
        safety_tools=tools,
        performance_marker=marker,
    )

    assert result.status is TaskStatus.REJECTED
    assert result.reason_code == "required_safety_incomplete"
    assert result.safety_status == "required_incomplete"
    assert result.safety_certified is False
    assert not marker.exists()


def test_three_fresh_attempts_deliver_robust_best_not_last_candidate(
    tmp_path: Path,
) -> None:
    sources = (
        "SPEED_MS = 8.0\ndef kernel(x):\n    return x + 1\n",
        "SPEED_MS = 6.0\ndef kernel(x):\n    return x + 2\n",
        "SPEED_MS = 7.0\ndef kernel(x):\n    return x + 3\n",
    )

    agent, task, result, run_root, events = _run_sequence(
        tmp_path, sources, max_iterations=3
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert result.srobust == pytest.approx(10.0 / 6.0)
    assert result.reward == pytest.approx(253.33333333333334)
    assert len(agent.requests) == 3
    assert len({item.attempt_id for item in agent.requests}) == 3
    assert len({item.workspace for item in agent.requests}) == 3
    assert agent.report_visible_before_edit == [False, False, False]
    assert (task.workspace / "source" / "kernel.py").read_text().endswith("return x\n")

    bundle = load_and_verify_kernel_bundle(
        Path(result.bundle_path), expected_digest=result.bundle_digest
    )
    patch_path = bundle.path / bundle.manifest["patches"][0]["path"]
    patch = patch_path.read_text(encoding="utf-8")
    assert "SPEED_MS = 6.0" in patch
    assert "SPEED_MS = 7.0" not in patch

    decisions = [item for item in events if item.event_type == "decision"]
    rewards = [item for item in events if item.event_type == "reward_committed"]
    assert len(decisions) == 3
    assert len({item.payload["attempt_id"] for item in decisions}) == 3
    assert [item.payload["verdict"] for item in decisions].count("keep") == 1
    keep = next(item for item in decisions if item.payload["verdict"] == "keep")
    assert keep.payload["attempt_id"] == agent.requests[1].attempt_id
    assert keep.payload["srobust"] == pytest.approx(10.0 / 6.0)
    assert len(rewards) == 3
    assert len({item.payload["attempt_id"] for item in rewards}) == 3
    graph = EpisodeGraphMaterializer(
        EventJournal(run_root / "events" / "run.db"),
        ArtifactStore(run_root / "artifacts"),
    ).materialize(run_root.name)
    assert len(graph.children) == 3
    assert all(child.trainability == "complete" for child in graph.children)
    assert [child.verdict for child in graph.children].count("keep") == 1


def test_compile_failure_is_typed_history_and_retried_in_fresh_attempt(
    tmp_path: Path,
) -> None:
    sources = (
        "COMPILE_FAIL = True\nSPEED_MS = 5.0\ndef kernel(x):\n    return x + 1\n",
        "SPEED_MS = 7.0\ndef kernel(x):\n    return x + 2\n",
    )

    agent, _, result, _, events = _run_sequence(
        tmp_path, sources, max_iterations=2
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert len(agent.requests) == 2
    assert agent.requests[0].workspace != agent.requests[1].workspace
    assert "compile_failed" in agent.requests[1].prompt
    assert agent.requests[0].attempt_id in agent.requests[1].prompt
    compile_events = [item for item in events if item.event_type == "compile_result"]
    assert [item.payload["passed"] for item in compile_events] == [False, True]
    assert not any(
        item.event_type == "performance_command_result"
        and item.payload["attempt_id"] == agent.requests[0].attempt_id
        for item in events
    )
    decisions = [item for item in events if item.event_type == "decision"]
    assert [item.payload["verdict"] for item in decisions].count("reject") == 1
    assert [item.payload["verdict"] for item in decisions].count("keep") == 1


def test_iteration_bound_and_context_history_are_exact_and_fresh(tmp_path: Path) -> None:
    sources = tuple(
        f"SPEED_MS = {value}.0\ndef kernel(x):\n    return x + {index}\n"
        for index, value in enumerate((9, 8, 7), start=1)
    )

    agent, _, result, _, events = _run_sequence(
        tmp_path, sources, max_iterations=3
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert len(agent.requests) == 3
    assert agent.requests[0].attempt_id in agent.requests[1].prompt
    assert agent.requests[0].attempt_id in agent.requests[2].prompt
    assert agent.requests[1].attempt_id in agent.requests[2].prompt
    contexts = [item for item in events if item.event_type == "context_packet_created"]
    assert len(contexts) == 3
    assert len({item.payload["context_packet_id"] for item in contexts}) == 3
    assert [item.payload["cycle"] for item in contexts] == [0, 1, 2]
    experiences = [item for item in events if item.event_type == "experience.measured"]
    assert len(experiences) == 3


def test_max_iterations_one_emits_canonical_agent_transcript(tmp_path: Path) -> None:
    source = "SPEED_MS = 8.0\ndef kernel(x):\n    return x + 1\n"

    agent, _, result, run_root, events = _run_sequence(
        tmp_path, (source,), max_iterations=1
    )

    assert result.status is TaskStatus.CANDIDATE_READY
    assert len(agent.requests) == 1
    agent_event = next(item for item in events if item.event_type == "agent_completed")
    assert agent_event.payload["turn_count"] == 1
    assert agent_event.payload["tool_call_count"] == 1
    assert agent_event.payload["message_event_count"] == 1
    assert agent_event.payload["usage"]["input_tokens"] == 1
    assert agent_event.payload["cost"] == {
        "amount": "0.125",
        "currency": "USD",
        "source_event_index": 0,
        "source_key": "total_cost_usd",
    }
    binding = next(
        item
        for item in agent_event.payload["artifacts"]
        if item["role"] == "agent_transcript"
    )
    receipt = ArtifactReceipt.from_dict(binding["receipt"])
    raw = ArtifactStore(run_root / "artifacts").read_bytes(receipt)
    transcript = json.loads(raw)
    assert transcript["schema"] == "apex.agent-transcript/v3"
    assert transcript["events"] == [
        {
            "kind": "turn.completed",
            "metadata": {
                "type": "turn.completed",
                "usage": {"input_tokens": 1},
            },
            "text": "",
        }
    ]
    assert [item["kind"] for item in transcript["semantic_events"]] == [
        "agent_message",
        "tool_called",
        "tool_result",
    ]
    assert transcript["usage"]["total_tokens"] == 3
    assert transcript["cost"]["amount"] == "0.125"

    attempt_events = tuple(
        event for event in events if event.payload.get("attempt_id") == agent.requests[0].attempt_id
    )
    event_types = [event.event_type for event in attempt_events]
    canonical = (
        "agent_message",
        "tool_called",
        "tool_result",
        "usage_recorded",
        "cost_recorded",
        "agent_completed",
    )
    assert all(
        event_types.index(first) < event_types.index(second)
        for first, second in zip(canonical, canonical[1:])
    )
    for event in attempt_events:
        if event.event_type not in canonical[:-1]:
            continue
        assert event.payload["evidence_class"] == "self_reported"
        assert any(
            item["role"] == "agent_transcript"
            for item in event.payload["artifacts"]
        )

    graph = EpisodeGraphMaterializer(
        EventJournal(run_root / "events" / "run.db"),
        ArtifactStore(run_root / "artifacts"),
    ).materialize(run_root.name)
    exported = DatasetExporter(ArtifactStore(run_root / "artifacts")).export(
        graph,
        tmp_path / "standalone-agent-rl",
        config=DatasetExportConfig(include_sft=False),
    )
    assert exported.record_count == 1
    transition = json.loads(
        (tmp_path / "standalone-agent-rl" / "dataset.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert "agent_message" in {
        event["event_type"] for event in transition["actions"]
    }
    assert {event["event_type"] for event in transition["tools"]} == {
        "tool_called",
        "tool_result",
    }
    assert {event["event_type"] for event in transition["costs"]["events"]} >= {
        "usage_recorded",
        "cost_recorded",
    }
    assert "agent_transcript" in transition["artifacts_by_role"]
