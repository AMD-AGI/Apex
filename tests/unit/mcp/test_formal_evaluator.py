from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from apex.core import ContractError, canonical_json_bytes, sha256_json
from apex.evaluation import user_confirmed_evaluation_authorizer
from apex.mcp import (
    CampaignStartHandler,
    CampaignStopHandler,
    CapabilityRegistry,
    CapabilityScope,
    KernelEvaluatorHandler,
    planned_capability_descriptors,
)
from apex.optimization.kernel import (
    CommandEvidence,
    ExecutableIdentity,
    KernelCampaignDraftUseCase,
    FormalKernelCampaign,
    KernelFormalCapabilityUseCase,
    KernelFormalEvaluator,
    stop_formal_campaign,
)
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentProcessContainmentReceipt,
    CapabilityAuthority,
    CapabilityRequest,
    KernelMeasurementOutput,
)
from apex.reporting import resolve_run_source
from apex.storage import ArtifactReceipt
from tests.support.gpu_evidence import (
    synthetic_gpu_heartbeat,
    synthetic_gpu_lease,
)


def _git(workspace: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments), cwd=workspace, check=True, capture_output=True
    )


def _containment() -> AgentProcessContainmentReceipt:
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


class _Verifier:
    def compile(self, resolved, *, candidate_root, expected_source_digest):
        del resolved, candidate_root, expected_source_digest
        return _command("compile")

    def correctness(self, resolved, *, candidate_root, expected_source_digest):
        del resolved, candidate_root, expected_source_digest
        return _command("correctness")

    def performance(self, resolved, *, candidate_root, expected_source_digest):
        del resolved, candidate_root, expected_source_digest
        return _command("performance")


class _CompileFailingVerifier(_Verifier):
    def compile(self, resolved, *, candidate_root, expected_source_digest):
        del resolved, candidate_root, expected_source_digest
        return CommandEvidence(
            phase="compile",
            argv=("/usr/bin/fixture", "compile"),
            executable_identity=_executable(),
            executable_identity_reverified=True,
            exit_code=1,
            timed_out=False,
            stdout="",
            stderr="compile failed",
            duration_seconds=0.01,
            process_containment=_containment(),
        )


def _command(phase: str) -> CommandEvidence:
    return CommandEvidence(
        phase=phase,
        argv=("/usr/bin/fixture", phase),
        executable_identity=_executable(),
        executable_identity_reverified=True,
        exit_code=0,
        timed_out=False,
        stdout="",
        stderr="",
        duration_seconds=0.01,
        process_containment=_containment(),
    )


def _executable() -> ExecutableIdentity:
    return ExecutableIdentity(
        path="/usr/bin/fixture",
        size=100,
        sha256="e" * 64,
        device=1,
        inode=2,
        mode=0o755,
        mtime_ns=3,
        ctime_ns=4,
    )


@dataclass(frozen=True)
class _LeaseReceipt:
    run_id: str

    @property
    def execution_scope(self) -> str:
        return "amd-gpu-set=fixture"

    @property
    def physical_scope(self) -> str:
        return "amd-gpu-unique-id-set=fixture"

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "run_id": self.run_id,
            "execution_scope": self.execution_scope,
            "physical_scope": self.physical_scope,
            "fixture": True,
        }


class _Lease:
    def __init__(self, run_id: str) -> None:
        self.receipt = synthetic_gpu_lease(run_id)
        self._heartbeat_sequence = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def measurement(self, action_id: str):
        return _MeasurementGuard(self.receipt, action_id)

    def heartbeat(self, reason: str = "manual"):
        self._heartbeat_sequence += 1
        return synthetic_gpu_heartbeat(
            self.receipt,
            reason=reason,
            sequence=self._heartbeat_sequence,
        )


@dataclass(frozen=True)
class _BracketReceipt:
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
    def __init__(self, lease, action_id: str) -> None:
        self.receipt = _BracketReceipt(lease.run_id, action_id, lease.digest)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


class _Leases:
    def __init__(self) -> None:
        self.calls = 0

    def acquire(self, run_id: str, *, requested_devices=None):
        del requested_devices
        self.calls += 1
        return _Lease(run_id)


class _AuthorityProvider:
    def __init__(self) -> None:
        self.consumed = False

    def consume(self, *, run_id, draft):
        del run_id
        if self.consumed:
            return None
        self.consumed = True
        return user_confirmed_evaluation_authorizer(draft.digest).authorize(draft)


@dataclass(frozen=True)
class _Baseline:
    receipt_sha256: str = "b" * 64

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "fixture.release-candidate-baseline/v1",
            "receipt_sha256": self.receipt_sha256,
        }


def _load_baseline(path: Path) -> _Baseline:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("invalid fixture baseline", "release_identity_invalid") from error
    expected = _Baseline().to_dict()
    if value != expected:
        raise ContractError("invalid fixture baseline", "release_identity_invalid")
    return _Baseline()


class _Measurement:
    adapter_id = "fixture-evaluator-v1"
    measurement_method_sha256 = "1" * 64

    def measure(self, request):
        request.report_path.write_bytes(canonical_json_bytes(_report()))
        return KernelMeasurementOutput(self.adapter_id, request.report_path)


def _report() -> dict[str, object]:
    health = {
        "device": "gfx950:0",
        "healthy": True,
        "temperature_c": 45.0,
        "clock_mhz": 2100.0,
    }
    order = (
        "reference", "optimized", "optimized", "reference",
        "optimized", "reference", "reference", "optimized",
    )
    return {
        "schema": "apex.kernel-measurement/v1",
        "policy_id": "kernel_invocation_nearest_rank_v1",
        "sample_unit": "kernel_invocation",
        "quantile_method": "nearest_rank_v1",
        "timer": "hip_event",
        "timer_resolution_ns": 1.0,
        "inner_repeats": 1,
        "measurement_method_sha256": "1" * 64,
        "abba_seed": 17,
        "warmup_samples": 20,
        "cases": [{
            "case_id": "fixture-case",
            "blocks": [
                {
                    "block_id": index,
                    "order_position": index,
                    "implementation": implementation,
                    "samples_ms": [10.0 if implementation == "reference" else 5.0] * 75,
                    "invalid_sample_counts": {},
                    "gpu_health_before": health,
                    "gpu_health_after": health,
                }
                for index, implementation in enumerate(order)
            ],
        }],
    }


def _event_document(source, event, role: str) -> dict[str, object]:
    binding = next(
        item for item in event.payload["artifacts"] if item["role"] == role
    )
    receipt = ArtifactReceipt.from_dict(binding["receipt"])
    raw = source.artifacts.read_bytes(receipt)
    value = json.loads(raw)
    assert canonical_json_bytes(value) == raw
    return value


def _registry(
    workspace: Path,
    results: Path,
    *,
    verifier=None,
    authority_provider=True,
    leases=None,
) -> CapabilityRegistry:
    results.mkdir(parents=True, exist_ok=True)
    (results / "baseline.json").write_bytes(
        canonical_json_bytes(_Baseline().to_dict())
    )
    scope = CapabilityScope(workspace, results)
    selected_leases = leases or _Leases()
    evaluator = KernelFormalEvaluator(
        verifier=verifier or _Verifier(),  # type: ignore[arg-type]
        gpu_leases=selected_leases,  # type: ignore[arg-type]
        measurement_evaluator=_Measurement(),  # type: ignore[arg-type]
        authority_provider=(
            _AuthorityProvider() if authority_provider else None
        ),
        baseline_loader=_load_baseline,
    )
    handler = KernelEvaluatorHandler(
        scope, KernelFormalCapabilityUseCase(evaluator)
    )
    registry = CapabilityRegistry()
    for descriptor in planned_capability_descriptors():
        if descriptor.capability_id == "campaign.start":
            registry.register(
                descriptor,
                CampaignStartHandler(
                    scope, KernelCampaignDraftUseCase(), _load_baseline
                ),
            )
        elif descriptor.capability_id == "campaign.stop":
            registry.register(
                descriptor,
                CampaignStopHandler(scope, stop_formal_campaign),
            )
        elif descriptor.capability_id in {
            "bundle.build", "kernel.compile", "kernel.correctness",
            "kernel.grade", "kernel.measure",
        }:
            registry.register(descriptor, handler)
    return registry


def _invoke(registry, capability_id: str, arguments: dict[str, object]):
    authority = (
        CapabilityAuthority.WORKSPACE_USER
        if capability_id in {"campaign.start", "campaign.stop"}
        else CapabilityAuthority.FORMAL_EVALUATOR
    )
    return registry.invoke(
        CapabilityRequest(capability_id, arguments, frozenset({authority}))
    )


def _start(registry, *, with_baseline: bool = True) -> dict[str, object]:
    task = {
        "task_id": "chat-formal",
        "instructions": "Optimize the kernel",
        "language": "triton",
        "editable_files": ["kernel.py"],
        "target_functions": ["kernel"],
        "commands": {
            phase: {"argv": ["true"]}
            for phase in ("compile", "correctness", "performance")
        },
        "measurement": {
            "schema": "apex.kernel-measurement/v1",
            "adapter_id": "fixture-evaluator-v1",
            "harness_files": ["harness.py"],
            "measurement_method_sha256": "1" * 64,
            "runner": {"argv": ["true"]},
            "aggregation": "equal_case",
        },
    }
    arguments: dict[str, object] = {"task": task}
    if with_baseline:
        arguments["release_candidate_receipt"] = "baseline.json"
    return _invoke(
        registry,
        "campaign.start",
        arguments,
    ).content["campaign"]


def _candidate(results: Path, campaign: dict[str, object]) -> Path:
    projection = campaign["candidate_projection"]
    assert isinstance(projection, dict)
    return results / str(projection["relative_path"])


def test_formal_measurement_fails_closed_without_phase_isolation(
    tmp_path: Path,
) -> None:
    workspace, results = tmp_path / "workspace", tmp_path / "results"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n")
    (workspace / "harness.py").write_text("# protected\n")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/formal.git")
    _git(workspace, "add", "kernel.py", "harness.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    leases = _Leases()
    registry = _registry(workspace, results, leases=leases)
    campaign = _start(registry)
    locator = campaign["run_locator"]["relative_path"]

    unverified = _invoke(registry, "kernel.grade", {"run_locator": locator})
    assert unverified.content["receipt"]["status"] == "unverified"
    assert unverified.reward_eligible is False

    (_candidate(results, campaign) / "kernel.py").write_text(
        "def kernel(x): return x + 0\n"
    )
    assert (workspace / "kernel.py").read_text() == "def kernel(x): return x\n"
    assert subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""
    recovered = FormalKernelCampaign.load(
        results / locator, workspace=workspace, results=results
    )
    assert recovered.ensure_candidate_projection() == _candidate(results, campaign)
    assert (recovered.ensure_candidate_projection() / "kernel.py").read_text().endswith(
        "return x + 0\n"
    )
    compiled = _invoke(
        registry,
        "kernel.compile",
        {
            "run_locator": locator,
            "confirmed_draft_digest": campaign["evaluation_contract_draft_digest"],
        },
    ).content["receipt"]
    common = {
        "run_locator": locator,
        "attempt_id": compiled["attempt_id"],
        "contract_digest": compiled["contract_digest"],
        "candidate_digest": compiled["candidate_digest"],
    }
    assert compiled["status"] == "passed"
    forged = {**common, "contract_digest": "f" * 64}
    from apex.core import IntegrityError

    with pytest.raises(IntegrityError) as mismatch:
        _invoke(registry, "kernel.grade", forged)
    assert mismatch.value.reason_code == "evaluation_authority_mismatch"
    assert _invoke(registry, "kernel.correctness", common).content["receipt"]["status"] == "passed"
    measured = _invoke(registry, "kernel.measure", common)
    assert measured.content["receipt"]["status"] == "no_measurement"
    assert measured.content["receipt"]["reason_code"] == "phase_isolation_incomplete"
    assert measured.reward_eligible is False
    assert leases.calls == 2
    source = resolve_run_source(results / locator)
    before_grade = source.journal.iter_events(source.run_id, verify=True)
    assert all(event.event_type != "reward_committed" for event in before_grade)

    graded = _invoke(registry, "kernel.grade", common)
    assert graded.reward_eligible is False
    assert graded.content["receipt"]["status"] == "no_measurement"
    assert graded.content["receipt"]["reward"] is None
    events = source.journal.iter_events(source.run_id, verify=True)
    heartbeats = [
        event
        for event in events
        if event.payload.get("kind") == "gpu_lease_heartbeat"
    ]
    assert [event.payload["phase"] for event in heartbeats] == [
        "compile",
        "correctness",
    ]
    assert all(event.event_type != "reward_committed" for event in events)
    assert all(event.event_type != "performance_command_result" for event in events)
    assert all(
        event.payload.get("kind") != "kernel_measurement_capture"
        for event in events
    )
    safety = next(event for event in events if event.event_type == "safety_result")
    assert safety.payload["allowed_to_measure"] is False
    assert safety.payload["promotion_eligible"] is False
    assert safety.payload["safety_certified"] is False
    assert safety.payload["reason_codes"] == ["phase_isolation_incomplete"]
    isolation = _event_document(source, safety, "phase_isolation")
    assert isolation["agent_process_tree_terminated"] is False
    assert isolation["credentials_revoked"] is False
    assert isolation["tool_channels_revoked"] is False
    assert isolation["report_directory_hidden_from_agent"] is False
    assert isolation["candidate_read_only"] is True
    assert all(
        not event.payload.get("kind", "").startswith("sanitizer")
        for event in events
    )
    calls = [event for event in events if event.event_type == "tool_called"]
    results = [event for event in events if event.event_type == "tool_result"]
    assert {event.payload["call_id"] for event in calls} == {
        event.payload["call_id"] for event in results
    }
    assert {event.payload["tool_name"] for event in calls} == {
        "campaign.start",
        "kernel.compile",
        "kernel.correctness",
        "kernel.measure",
        "kernel.grade",
    }
    for event in calls:
        document = _event_document(
            source, event, "formal_capability_arguments"
        )
        assert document["schema"] == "apex.formal-capability-arguments/v2"
        assert document["capability_grant"] is None
        assert document["capability_id"] == event.payload["tool_name"]
    for event in results:
        document = _event_document(source, event, "formal_capability_result")
        assert document["schema"] == "apex.formal-capability-result/v1"
        assert document["call_id"] == event.payload["call_id"]


def test_agent_echo_cannot_mint_formal_authority(tmp_path: Path) -> None:
    workspace, results = tmp_path / "workspace", tmp_path / "results"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n")
    (workspace / "harness.py").write_text("# protected\n")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/echo.git")
    _git(workspace, "add", "kernel.py", "harness.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    leases = _Leases()
    registry = _registry(
        workspace, results, authority_provider=False, leases=leases
    )
    campaign = _start(registry)
    (_candidate(results, campaign) / "kernel.py").write_text(
        "def kernel(x): return x + 0\n"
    )

    compiled = _invoke(
        registry,
        "kernel.compile",
        {
            "run_locator": campaign["run_locator"]["relative_path"],
            "confirmed_draft_digest": campaign["evaluation_contract_draft_digest"],
        },
    ).content["receipt"]

    assert compiled["status"] == "unverified"
    assert compiled["reason_code"] == "evaluation_authority_missing"
    assert compiled["attempt_id"] is None
    assert leases.calls == 0
    source = resolve_run_source(
        results / campaign["run_locator"]["relative_path"]
    )
    events = tuple(source.journal.iter_events(source.run_id, verify=True))
    assert all(
        event.payload.get("kind") != "evaluation_contract_authorized"
        for event in events
    )
    assert all(event.event_type != "candidate_frozen" for event in events)
    assert all(event.event_type != "reward_committed" for event in events)


def test_missing_release_baseline_stays_unverified_before_gpu(tmp_path: Path) -> None:
    workspace, results = tmp_path / "workspace", tmp_path / "results"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n")
    (workspace / "harness.py").write_text("# protected\n")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/baseline.git")
    _git(workspace, "add", "kernel.py", "harness.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    leases = _Leases()
    registry = _registry(workspace, results, leases=leases)
    campaign = _start(registry, with_baseline=False)

    compiled = _invoke(
        registry,
        "kernel.compile",
        {
            "run_locator": campaign["run_locator"]["relative_path"],
            "confirmed_draft_digest": campaign["evaluation_contract_draft_digest"],
        },
    ).content["receipt"]

    assert campaign["candidate_projection"] is None
    assert compiled["status"] == "unverified"
    assert compiled["reason_code"] == "campaign_baseline_receipt_required"
    assert leases.calls == 0


def test_formal_compile_rejects_noneditable_projection_write(tmp_path: Path) -> None:
    workspace, results = tmp_path / "workspace", tmp_path / "results"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n")
    (workspace / "harness.py").write_text("# protected\n")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/formal.git")
    _git(workspace, "add", "kernel.py", "harness.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    registry = _registry(workspace, results)
    campaign = _start(registry)
    (_candidate(results, campaign) / "harness.py").write_text("# tampered\n")

    from apex.core import IntegrityError

    with pytest.raises(IntegrityError) as raised:
        _invoke(
            registry,
            "kernel.compile",
            {
                "run_locator": campaign["run_locator"]["relative_path"],
                "confirmed_draft_digest": campaign["evaluation_contract_draft_digest"],
            },
        )
    assert raised.value.reason_code == "undeclared_agent_edit"


def test_formal_stop_without_attempt_is_untrainable_and_idempotent(
    tmp_path: Path,
) -> None:
    workspace, results = tmp_path / "workspace", tmp_path / "results"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n")
    (workspace / "harness.py").write_text("# protected\n")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/stop.git")
    _git(workspace, "add", "kernel.py", "harness.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    registry = _registry(workspace, results)
    draft = _start(registry)
    locator = draft["run_locator"]["relative_path"]

    stopped = _invoke(
        registry, "campaign.stop", {"run_locator": locator}
    ).content["campaign"]
    source = resolve_run_source(results / locator)
    first_events = tuple(source.journal.iter_events(source.run_id, verify=True))
    repeated = _invoke(
        registry, "campaign.stop", {"run_locator": locator}
    ).content["campaign"]

    assert stopped["terminal_status"] == "cancelled"
    assert stopped["task_reward"] is None
    assert stopped["trainability"] == "untrainable"
    assert stopped["stop"]["untrainable_reason"] == (
        "terminal_measurement_unavailable"
    )
    assert repeated["stop"] == stopped["stop"]
    assert tuple(source.journal.iter_events(source.run_id, verify=True)) == first_events
    assert [event.event_type for event in first_events[-2:]] == [
        "tool_result",
        "run.cancelled",
    ]
    assert sum(
        event.payload.get("kind") == "kernel_terminal_result"
        for event in first_events
    ) == 1
    assert all(event.event_type != "reward_committed" for event in first_events)


def test_formal_stop_is_untrainable_after_missing_phase_isolation(
    tmp_path: Path,
) -> None:
    workspace, results = tmp_path / "workspace", tmp_path / "results"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n")
    (workspace / "harness.py").write_text("# protected\n")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/noop.git")
    _git(workspace, "add", "kernel.py", "harness.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    registry = _registry(workspace, results)
    draft = _start(registry)
    locator = draft["run_locator"]["relative_path"]
    (_candidate(results, draft) / "kernel.py").write_text(
        "def kernel(x): return x + 0\n"
    )
    compiled = _invoke(
        registry,
        "kernel.compile",
        {
            "run_locator": locator,
            "confirmed_draft_digest": draft["evaluation_contract_draft_digest"],
        },
    ).content["receipt"]
    attempt = {
        "run_locator": locator,
        "attempt_id": compiled["attempt_id"],
        "contract_digest": compiled["contract_digest"],
        "candidate_digest": compiled["candidate_digest"],
    }
    _invoke(registry, "kernel.correctness", attempt)
    measured = _invoke(registry, "kernel.measure", attempt)
    assert measured.content["receipt"]["reason_code"] == "phase_isolation_incomplete"
    assert _invoke(registry, "kernel.grade", attempt).reward_eligible is False

    stopped = _invoke(
        registry, "campaign.stop", {"run_locator": locator}
    ).content["campaign"]
    source = resolve_run_source(results / locator)
    events = tuple(source.journal.iter_events(source.run_id, verify=True))
    rewards = [event for event in events if event.event_type == "reward_committed"]
    decision = next(event for event in events if event.event_type == "decision")

    assert stopped["terminal_status"] == "cancelled"
    assert stopped["task_reward"] is None
    assert stopped["trainability"] == "untrainable"
    assert stopped["stop"]["untrainable_reason"] == (
        "terminal_measurement_unavailable"
    )
    assert rewards == []
    assert decision.payload["verdict"] == "needs_more_measurement"
    assert decision.payload["reason"] == "phase_isolation_incomplete"


def test_formal_stop_projects_compile_failure_as_zero_terminal_reward(
    tmp_path: Path,
) -> None:
    workspace, results = tmp_path / "workspace", tmp_path / "results"
    workspace.mkdir()
    (workspace / "kernel.py").write_text("def kernel(x): return x\n")
    (workspace / "harness.py").write_text("# protected\n")
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(workspace, "remote", "add", "origin", "https://example.invalid/fail.git")
    _git(workspace, "add", "kernel.py", "harness.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    registry = _registry(
        workspace, results, verifier=_CompileFailingVerifier()
    )
    draft = _start(registry)
    locator = draft["run_locator"]["relative_path"]
    (_candidate(results, draft) / "kernel.py").write_text(
        "def kernel(x): return x + 0\n"
    )

    compiled = _invoke(
        registry,
        "kernel.compile",
        {
            "run_locator": locator,
            "confirmed_draft_digest": draft["evaluation_contract_draft_digest"],
        },
    ).content["receipt"]
    stopped = _invoke(
        registry, "campaign.stop", {"run_locator": locator}
    ).content["campaign"]
    source = resolve_run_source(results / locator)
    events = tuple(source.journal.iter_events(source.run_id, verify=True))

    assert compiled["status"] == "failed"
    assert stopped["task_reward"] == 0.0
    assert stopped["trainability"] == "complete"
    assert [
        event.payload["scalar_reward"]
        for event in events
        if event.event_type == "reward_committed"
    ] == [0.0, 0.0]
    assert next(
        event for event in events if event.event_type == "decision"
    ).payload["verdict"] == "reject"
