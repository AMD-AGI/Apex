from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import pytest

from apex.core import ContractError, sha256_bytes, sha256_file
from apex.evaluation.safety import (
    ArtifactKind,
    CapabilityCheck,
    CapabilityStatus,
    ExecutionStatus,
    FindingStatus,
    FrozenCandidate,
    InstrumentationControl,
    KernelLanguage,
    PhaseIsolationReceipt,
    SafetyGate,
    SafetyGateRequest,
    SafetyRequirement,
    SubprocessSafetyToolRunner,
    TaskSafetyProfile,
    ToolCapability,
    ToolPolicy,
    ToolRuntimeIdentity,
    ToolVerificationPlan,
    VerificationPlan,
    VerificationPolicy,
)
from apex.evaluation.safety.results import TOOL_REPORT_SCHEMA_VERSION
from apex.ports import SafetyToolRunRequest, SafetyToolRunResult


def _digest(label: str) -> str:
    return sha256_bytes(label.encode())


def _ready_capability(tool: str = "gpu_asan") -> ToolCapability:
    ready = CapabilityCheck(CapabilityStatus.READY)
    return ToolCapability(tool=tool, engine=ready, adapter=ready, runtime=ready)


def _blocked_capability(status: CapabilityStatus, tool: str = "gpu_asan") -> ToolCapability:
    ready = CapabilityCheck(CapabilityStatus.READY)
    blocked = CapabilityCheck(status, reason_code=f"{status.value}_fixture")
    return ToolCapability(tool=tool, engine=blocked, adapter=ready, runtime=ready)


@dataclass
class _Scenario:
    profile: TaskSafetyProfile
    frozen: FrozenCandidate
    policy: VerificationPolicy
    tool_plan: ToolVerificationPlan | None
    plan: VerificationPlan
    isolation: PhaseIsolationReceipt
    artifact_root: Path

    def request(self, **updates: object) -> SafetyGateRequest:
        values: dict[str, object] = {
            "plan": self.plan,
            "policy": self.policy,
            "frozen_candidate": self.frozen,
            "isolation_receipt": self.isolation,
            "artifact_root": self.artifact_root,
            "current_run_id": self.plan.run_id,
            "current_candidate_id": self.plan.candidate_id,
            "current_anchor_generation": self.plan.anchor_generation,
            "current_deployed_digest": self.plan.deployed_digest,
        }
        values.update(updates)
        return SafetyGateRequest(**values)  # type: ignore[arg-type]


def _scenario(
    tmp_path: Path,
    *,
    requirement: SafetyRequirement = SafetyRequirement.ADVISORY,
    qualified: bool = True,
    capability: ToolCapability | None = None,
    tools_enabled: bool = True,
    artifact_kind: ArtifactKind = ArtifactKind.PYTHON_JIT,
    instrumentation_control: InstrumentationControl = InstrumentationControl.COMPILER_CONTROLLED,
) -> _Scenario:
    candidate_root = tmp_path / "candidate-cas"
    candidate_root.mkdir()
    source = candidate_root / "kernel.py"
    source.write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    source.chmod(0o444)
    profile = TaskSafetyProfile(
        language=KernelLanguage.TRITON,
        artifact_kind=artifact_kind,
        instrumentation_control=instrumentation_control,
        submission_paths=("kernel.py",),
        target_symbols=("kernel",),
        adapter_capabilities=("python_dispatch",),
    )
    frozen = FrozenCandidate.capture(candidate_root, profile)
    rules = (
        ToolPolicy(
            tool="gpu_asan",
            requirement=requirement if tools_enabled else SafetyRequirement.DISABLED,
            qualified=qualified if tools_enabled else False,
            qualification_digest=_digest("qualification") if tools_enabled and qualified else None,
        ),
    )
    policy = VerificationPolicy(rules=rules)
    tool_plan = None
    tools: tuple[ToolVerificationPlan, ...] = ()
    if tools_enabled:
        tool_plan = ToolVerificationPlan(
            identity=ToolRuntimeIdentity(
                tool="gpu_asan",
                version="1.2.3",
                plugin_digest=_digest("plugin"),
                runtime_image_id=f"sha256:{_digest('image')}",
                helper_digest=_digest("helper"),
                dispatch_digest=_digest("dispatch"),
            ),
            capability=capability or _ready_capability(),
            argv=(sys.executable, "-m", "trusted_safety_helper"),
            cases=("case-1", "case-2"),
            positive_control_digest=_digest("positive-control"),
            timeout_seconds=5,
            output_limit_bytes=4096,
            environment=(("PATH", os.environ.get("PATH", "")),),
        )
        tools = (tool_plan,)
    plan = VerificationPlan(
        run_id="run-1",
        candidate_id="candidate-1",
        anchor_generation=3,
        profile=profile,
        policy_fingerprint=policy.fingerprint,
        source_digest=_digest("baseline-source"),
        candidate_digest=frozen.candidate_digest,
        deployed_digest=_digest("normal-deployed-artifact"),
        tools=tools,
    )
    artifact_root = tmp_path / "evaluator-only"
    isolation = PhaseIsolationReceipt(
        run_id=plan.run_id,
        plan_fingerprint=plan.fingerprint,
        anchor_generation=plan.anchor_generation,
        candidate_digest=plan.candidate_digest,
        frozen_root=str(candidate_root),
        evaluator_artifact_root=str(artifact_root),
        agent_process_tree_terminated=True,
        credentials_revoked=True,
        tool_channels_revoked=True,
        report_directory_hidden_from_agent=True,
        candidate_read_only=True,
    )
    return _Scenario(profile, frozen, policy, tool_plan, plan, isolation, artifact_root)


class _FakeRunner:
    def __init__(self, handler: Callable[[SafetyToolRunRequest], SafetyToolRunResult]) -> None:
        self.handler = handler
        self.calls: list[SafetyToolRunRequest] = []

    def run(self, request: SafetyToolRunRequest) -> SafetyToolRunResult:
        self.calls.append(request)
        return self.handler(request)


def _report(
    scenario: _Scenario,
    request: SafetyToolRunRequest,
    *,
    findings: tuple[str, str] = ("clean", "clean"),
    executions: tuple[str, str] = ("completed", "completed"),
    complete: bool = True,
    mutate: Callable[[dict[str, object]], None] | None = None,
    exit_code: int = 0,
    timed_out: bool = False,
    stdout: str = "",
    stderr: str = "",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    report_path: Path | None = None,
    report_origin: str = "evaluator",
) -> SafetyToolRunResult:
    assert scenario.tool_plan is not None
    artifact = request.artifact_root / "instrumented.bin"
    artifact.write_bytes(b"instrumented sanitizer build")
    body: dict[str, object] = {
        "schema_version": TOOL_REPORT_SCHEMA_VERSION,
        "complete": complete,
        "plan_fingerprint": scenario.plan.fingerprint,
        "identity": scenario.tool_plan.identity.to_dict(),
        "lineage": {
            "source_digest": scenario.plan.source_digest,
            "candidate_digest": scenario.plan.candidate_digest,
            "deployed_digest": scenario.plan.deployed_digest,
        },
        "positive_control": {
            "digest": scenario.tool_plan.positive_control_digest,
            "status": "passed",
        },
        "cases": [
            {
                "case_id": case_id,
                "execution": execution,
                "finding": finding,
                "dispatch_digest": scenario.tool_plan.identity.dispatch_digest,
            }
            for case_id, execution, finding in zip(
                scenario.tool_plan.cases,
                executions,
                findings,
            )
        ],
        "artifacts": [
            {
                "role": "instrumented_artifact",
                "path": "instrumented.bin",
                "sha256": sha256_file(artifact),
                "size": artifact.stat().st_size,
            }
        ],
    }
    if mutate is not None:
        mutate(body)
    destination = report_path or request.report_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body), encoding="utf-8")
    return SafetyToolRunResult(
        exit_code=exit_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        duration_seconds=0.25,
        report_path=destination,
        report_origin=report_origin,
    )


def _evaluate(
    scenario: _Scenario,
    handler: Callable[[SafetyToolRunRequest], SafetyToolRunResult],
    **request_updates: object,
):
    runner = _FakeRunner(handler)
    result = SafetyGate(runner).evaluate(scenario.request(**request_updates))
    return result, runner


def test_qualified_clean_receipt_is_certified_and_never_timing(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    result, runner = _evaluate(scenario, lambda request: _report(scenario, request))

    assert len(runner.calls) == 1
    evaluation = result.evaluations[0]
    assert (evaluation.capability, evaluation.execution, evaluation.finding) == (
        CapabilityStatus.READY,
        ExecutionStatus.COMPLETED,
        FindingStatus.CLEAN,
    )
    assert evaluation.lineage is not None
    assert evaluation.lineage.identity == scenario.tool_plan.identity
    assert result.decision.allowed_to_measure
    assert result.decision.promotion_eligible
    assert result.safety_certified
    artifact_digest = evaluation.artifacts[0].digest
    assert artifact_digest in result.forbidden_timing_digests
    with pytest.raises(ContractError, match="cannot be used for timing"):
        result.assert_performance_artifact_allowed(artifact_digest)
    result.assert_performance_artifact_allowed(scenario.plan.deployed_digest)


@pytest.mark.parametrize("exit_code", [0, 17])
def test_confirmed_finding_always_rejects_even_on_nonzero_exit(
    tmp_path: Path, exit_code: int
) -> None:
    scenario = _scenario(tmp_path, qualified=False)
    result, _ = _evaluate(
        scenario,
        lambda request: _report(
            scenario,
            request,
            findings=("found", "clean"),
            exit_code=exit_code,
        ),
    )

    assert result.evaluations[0].finding is FindingStatus.FOUND
    assert result.evaluations[0].execution is (
        ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.TOOL_ERROR
    )
    assert result.decision.reject
    assert not result.decision.allowed_to_measure
    assert result.decision.reason_codes == ("confirmed_safety_finding",)


@pytest.mark.parametrize(
    ("status", "required", "expected_reject"),
    [
        (CapabilityStatus.UNSUPPORTED, False, False),
        (CapabilityStatus.UNAVAILABLE_RUNTIME, False, False),
        (CapabilityStatus.ADAPTER_REQUIRED, True, True),
    ],
)
def test_nonready_capabilities_are_never_clean(
    tmp_path: Path,
    status: CapabilityStatus,
    required: bool,
    expected_reject: bool,
) -> None:
    scenario = _scenario(
        tmp_path,
        requirement=SafetyRequirement.REQUIRED if required else SafetyRequirement.ADVISORY,
        qualified=True if required else False,
        capability=_blocked_capability(status),
    )
    runner = _FakeRunner(lambda request: pytest.fail("non-ready tool must not run"))
    result = SafetyGate(runner).evaluate(scenario.request())

    evaluation = result.evaluations[0]
    assert not runner.calls
    assert evaluation.capability is status
    assert evaluation.execution is ExecutionStatus.NOT_RUN
    assert evaluation.finding is FindingStatus.NOT_EVALUATED
    assert result.decision.reject is expected_reject
    assert not result.safety_certified


def test_not_applicable_is_not_clean_or_certified(tmp_path: Path) -> None:
    scenario = _scenario(
        tmp_path,
        requirement=SafetyRequirement.REQUIRED,
        qualified=True,
        capability=_blocked_capability(CapabilityStatus.NOT_APPLICABLE),
    )
    runner = _FakeRunner(lambda request: pytest.fail("not-applicable tool must not run"))
    result = SafetyGate(runner).evaluate(scenario.request())

    evaluation = result.evaluations[0]
    assert evaluation.finding is FindingStatus.NOT_EVALUATED
    assert result.decision.allowed_to_measure
    assert result.decision.promotion_eligible
    assert not result.safety_certified
    assert result.decision.reason_codes == ("no_applicable_safety_check",)


def test_precompiled_artifact_without_instrumentation_stays_unsupported(tmp_path: Path) -> None:
    scenario = _scenario(
        tmp_path,
        qualified=False,
        artifact_kind=ArtifactKind.PRECOMPILED,
        instrumentation_control=InstrumentationControl.NONE,
        capability=_blocked_capability(CapabilityStatus.UNSUPPORTED),
    )
    runner = _FakeRunner(lambda request: pytest.fail("unsupported precompiled artifact must not run"))
    result = SafetyGate(runner).evaluate(scenario.request())
    assert not runner.calls
    assert result.evaluations[0].capability is CapabilityStatus.UNSUPPORTED
    assert result.evaluations[0].finding is FindingStatus.NOT_EVALUATED
    assert result.decision.allow_ordinary_keep
    assert not result.safety_certified


@pytest.mark.parametrize(
    ("required", "expected_reject"),
    [(False, False), (True, True)],
)
def test_inconclusive_policy_truth_table(
    tmp_path: Path, required: bool, expected_reject: bool
) -> None:
    scenario = _scenario(
        tmp_path,
        requirement=SafetyRequirement.REQUIRED if required else SafetyRequirement.ADVISORY,
        qualified=True if required else False,
    )
    result, _ = _evaluate(
        scenario,
        lambda request: _report(
            scenario,
            request,
            executions=("tool_error", "completed"),
            findings=("inconclusive", "clean"),
        ),
    )

    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert result.decision.reject is expected_reject
    assert not result.safety_certified


def test_candidate_owned_report_cannot_claim_clean(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)
    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, report_origin="candidate_workspace"),
    )

    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "untrusted_safety_report_origin" in result.evaluations[0].reason_codes
    assert result.decision.allow_ordinary_keep
    assert not result.safety_certified


def test_stale_anchor_blocks_before_runner(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    runner = _FakeRunner(lambda request: pytest.fail("stale plan must not execute"))
    result = SafetyGate(runner).evaluate(
        scenario.request(current_anchor_generation=scenario.plan.anchor_generation + 1)
    )

    assert not runner.calls
    assert result.decision.reject
    assert result.gate_errors == ("stale_safety_anchor",)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_truncated_output_cannot_be_clean(tmp_path: Path, stream: str) -> None:
    scenario = _scenario(tmp_path, qualified=False)
    kwargs = {f"{stream}_truncated": True}
    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, **kwargs),  # type: ignore[arg-type]
    )

    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_tool_output_truncated" in result.evaluations[0].reason_codes
    assert not result.safety_certified


def test_valid_json_marked_incomplete_cannot_be_clean(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)
    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, complete=False),
    )
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_report_incomplete" in result.evaluations[0].reason_codes


def test_syntactically_truncated_report_cannot_be_clean(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def truncated(request: SafetyToolRunRequest) -> SafetyToolRunResult:
        request.report_path.write_text('{"schema_version":', encoding="utf-8")
        return SafetyToolRunResult(
            exit_code=0,
            timed_out=False,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
            report_path=request.report_path,
        )

    result, _ = _evaluate(scenario, truncated)
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "invalid_or_truncated_safety_report" in result.evaluations[0].reason_codes


@pytest.mark.parametrize(
    "identity_field",
    ["version", "plugin_digest", "runtime_image_id", "helper_digest", "dispatch_digest"],
)
def test_exact_tool_runtime_identity_is_required(tmp_path: Path, identity_field: str) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        identity = dict(body["identity"])  # type: ignore[arg-type]
        identity[identity_field] = "forged" if identity_field == "version" else _digest("forged")
        if identity_field == "runtime_image_id":
            identity[identity_field] = f"sha256:{_digest('forged-image')}"
        body["identity"] = identity

    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, mutate=mutate),
    )
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert result.evaluations[0].lineage is None
    assert "safety_tool_identity_mismatch" in result.evaluations[0].reason_codes


@pytest.mark.parametrize("digest_field", ["source_digest", "candidate_digest", "deployed_digest"])
def test_exact_source_candidate_and_deployed_lineage_is_required(
    tmp_path: Path, digest_field: str
) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        lineage = dict(body["lineage"])  # type: ignore[arg-type]
        lineage[digest_field] = _digest("wrong-lineage")
        body["lineage"] = lineage

    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, mutate=mutate),
    )
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_lineage_digest_mismatch" in result.evaluations[0].reason_codes


def test_wrong_dispatch_is_inconclusive(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        cases = list(body["cases"])  # type: ignore[arg-type]
        cases[0] = {**cases[0], "dispatch_digest": _digest("wrong-dispatch")}  # type: ignore[arg-type]
        body["cases"] = cases

    result, _ = _evaluate(scenario, lambda request: _report(scenario, request, mutate=mutate))
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_dispatch_digest_mismatch" in result.evaluations[0].reason_codes


def test_artifact_path_escape_is_inconclusive(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        artifacts = list(body["artifacts"])  # type: ignore[arg-type]
        artifacts[0] = {**artifacts[0], "path": "../candidate-cas/kernel.py"}  # type: ignore[arg-type]
        body["artifacts"] = artifacts

    result, _ = _evaluate(scenario, lambda request: _report(scenario, request, mutate=mutate))
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_artifact_path_escape" in result.evaluations[0].reason_codes


def test_artifact_digest_forgery_is_inconclusive(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        artifacts = list(body["artifacts"])  # type: ignore[arg-type]
        artifacts[0] = {**artifacts[0], "sha256": _digest("forged-artifact")}  # type: ignore[arg-type]
        body["artifacts"] = artifacts

    result, _ = _evaluate(scenario, lambda request: _report(scenario, request, mutate=mutate))
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_artifact_digest_mismatch" in result.evaluations[0].reason_codes


def test_wrong_positive_control_is_inconclusive(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        body["positive_control"] = {"digest": _digest("wrong"), "status": "passed"}

    result, _ = _evaluate(scenario, lambda request: _report(scenario, request, mutate=mutate))
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_positive_control_failed" in result.evaluations[0].reason_codes


def test_missing_case_is_inconclusive(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        body["cases"] = list(body["cases"])[0:1]  # type: ignore[arg-type]

    result, _ = _evaluate(scenario, lambda request: _report(scenario, request, mutate=mutate))
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_case_set_mismatch" in result.evaluations[0].reason_codes


def test_stale_report_fingerprint_is_inconclusive(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)

    def mutate(body: dict[str, object]) -> None:
        body["plan_fingerprint"] = _digest("stale-plan")

    result, _ = _evaluate(scenario, lambda request: _report(scenario, request, mutate=mutate))
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_report_plan_mismatch" in result.evaluations[0].reason_codes


def test_report_outside_evaluator_directory_is_rejected(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)
    outside = tmp_path / "candidate-authored-report.json"
    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, report_path=outside),
    )
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert "safety_report_path_mismatch" in result.evaluations[0].reason_codes


def test_timeout_is_bounded_and_inconclusive(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)
    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, timed_out=True, exit_code=-15),
    )
    assert result.evaluations[0].execution is ExecutionStatus.TIMEOUT
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE
    assert not result.safety_certified


def test_oversized_output_is_treated_as_truncated_even_if_runner_lies(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, qualified=False)
    assert scenario.tool_plan is not None
    output = "x" * (scenario.tool_plan.output_limit_bytes + 1)
    result, _ = _evaluate(
        scenario,
        lambda request: _report(scenario, request, stdout=output),
    )
    assert result.evaluations[0].stdout_truncated
    assert result.evaluations[0].finding is FindingStatus.INCONCLUSIVE


def test_incomplete_phase_isolation_blocks_all_tools(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    scenario.isolation = replace(scenario.isolation, credentials_revoked=False)
    runner = _FakeRunner(lambda request: pytest.fail("isolated verifier must not start"))
    result = SafetyGate(runner).evaluate(scenario.request())

    assert not runner.calls
    assert result.decision.reject
    assert "phase_isolation_incomplete" in result.gate_errors


def test_evaluator_artifacts_cannot_overlap_frozen_candidate(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    overlapping = scenario.frozen.root / "candidate-authored-evidence"
    isolation = replace(scenario.isolation, evaluator_artifact_root=str(overlapping))
    runner = _FakeRunner(lambda request: pytest.fail("overlapping verifier must not start"))
    result = SafetyGate(runner).evaluate(
        scenario.request(artifact_root=overlapping, isolation_receipt=isolation)
    )
    assert result.decision.reject
    assert "safety_phase_path_overlap" in result.gate_errors
    assert not overlapping.exists()


def test_candidate_digest_drift_blocks_before_tool_execution(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    source = scenario.frozen.root / "kernel.py"
    source.chmod(0o644)
    source.write_text("def kernel(x):\n    return x + 1\n", encoding="utf-8")
    runner = _FakeRunner(lambda request: pytest.fail("mutated candidate must not execute"))
    result = SafetyGate(runner).evaluate(scenario.request())
    assert result.decision.reject
    assert "mutable_frozen_candidate" in result.gate_errors


def test_symlink_candidate_source_cannot_be_frozen(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    outside.chmod(0o444)
    (root / "kernel.py").symlink_to(outside)
    profile = TaskSafetyProfile(
        language=KernelLanguage.TRITON,
        artifact_kind=ArtifactKind.PYTHON_JIT,
        instrumentation_control=InstrumentationControl.COMPILER_CONTROLLED,
        submission_paths=("kernel.py",),
    )
    with pytest.raises(ContractError, match="symlinks are forbidden"):
        FrozenCandidate.capture(root, profile)


@pytest.mark.parametrize("path", ["../kernel.py", "/tmp/kernel.py", "dir\\kernel.py", "./kernel.py"])
def test_submission_path_escape_is_rejected(path: str) -> None:
    with pytest.raises(ContractError):
        TaskSafetyProfile(
            language=KernelLanguage.TRITON,
            artifact_kind=ArtifactKind.PYTHON_JIT,
            instrumentation_control=InstrumentationControl.COMPILER_CONTROLLED,
            submission_paths=(path,),
        )


def test_mutable_runtime_image_tag_is_rejected() -> None:
    with pytest.raises(ContractError, match="immutable sha256 image ID"):
        ToolRuntimeIdentity(
            tool="gpu_asan",
            version="1",
            plugin_digest=_digest("plugin"),
            runtime_image_id="magpie-sanitizer:latest",
            helper_digest=_digest("helper"),
            dispatch_digest=_digest("dispatch"),
        )


def test_missing_candidate_digest_is_rejected_at_plan_boundary(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    with pytest.raises(ContractError, match="candidate_digest"):
        replace(scenario.plan, candidate_digest="")


def test_effective_capability_cannot_be_forged() -> None:
    ready = CapabilityCheck(CapabilityStatus.READY)
    unsupported = CapabilityCheck(CapabilityStatus.UNSUPPORTED, "unsupported_fixture")
    with pytest.raises(ContractError, match="disagrees"):
        ToolCapability(
            tool="gpu_asan",
            engine=unsupported,
            adapter=ready,
            runtime=ready,
            effective=ready,
        )


def test_shell_command_is_rejected_but_metacharacters_stay_argv_data(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    assert scenario.tool_plan is not None
    with pytest.raises(ContractError, match="without a shell"):
        replace(scenario.tool_plan, argv=("bash", "-lc", "tool"))
    direct = replace(scenario.tool_plan, argv=(sys.executable, "literal;$(not-executed)"))
    assert direct.argv[1] == "literal;$(not-executed)"


def test_tools_disabled_keeps_ordinary_flow_byte_independent(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, tools_enabled=False)
    runner = _FakeRunner(lambda request: pytest.fail("disabled tool must not run"))
    before = {"compiled": True, "correct": True, "reward": 241.25}
    result = SafetyGate(runner).evaluate(scenario.request())
    after = {"compiled": True, "correct": True, "reward": 241.25}

    assert before == after
    assert not runner.calls
    assert result.evaluations == ()
    assert result.decision.allowed_to_measure
    assert result.decision.promotion_eligible
    assert not result.safety_certified


def test_plan_and_isolation_round_trip_reject_tampering(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path)
    constructed = VerificationPlan.create(
        run_id=scenario.plan.run_id,
        candidate_id=scenario.plan.candidate_id,
        anchor_generation=scenario.plan.anchor_generation,
        profile=scenario.profile,
        policy=scenario.policy,
        source_digest=scenario.plan.source_digest,
        candidate_digest=scenario.plan.candidate_digest,
        deployed_digest=scenario.plan.deployed_digest,
        tools=scenario.plan.tools,
    )
    assert constructed == scenario.plan
    restored = VerificationPlan.from_dict(scenario.plan.to_dict())
    assert restored == scenario.plan
    assert restored.canonical_bytes() == scenario.plan.canonical_bytes()
    tampered_plan = scenario.plan.to_dict()
    tampered_plan["candidate_digest"] = _digest("forged")
    with pytest.raises(ContractError, match="fingerprint mismatch"):
        VerificationPlan.from_dict(tampered_plan)

    restored_receipt = PhaseIsolationReceipt.from_dict(scenario.isolation.to_dict())
    assert restored_receipt == scenario.isolation
    tampered_receipt = scenario.isolation.to_dict()
    tampered_receipt["credentials_revoked"] = False
    with pytest.raises(ContractError, match="fingerprint mismatch"):
        PhaseIsolationReceipt.from_dict(tampered_receipt)

    restored_policy = VerificationPolicy.from_dict(scenario.policy.to_dict())
    assert restored_policy == scenario.policy
    assert restored_policy.canonical_bytes() == scenario.policy.canonical_bytes()
    tampered_policy = scenario.policy.to_dict()
    tampered_policy["rules"][0]["requirement"] = "disabled"  # type: ignore[index]
    with pytest.raises(ContractError, match="fingerprint mismatch"):
        VerificationPolicy.from_dict(tampered_policy)


def test_unqualified_tool_cannot_be_required() -> None:
    with pytest.raises(ContractError, match="qualified before it can be required"):
        ToolPolicy(
            tool="gpu_asan",
            requirement=SafetyRequirement.REQUIRED,
            qualified=False,
        )


def test_no_tools_policy_is_explicit_and_uncertified() -> None:
    policy = VerificationPolicy.no_tools()
    assert policy.rules == ()
    assert len(policy.fingerprint) == 64


def test_real_runner_uses_direct_argv_and_bounded_timeout(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    request = SafetyToolRunRequest(
        tool="fixture",
        plan_fingerprint=_digest("plan"),
        argv=(sys.executable, "-c", "print('ok')"),
        cwd=tmp_path,
        environment=(("PATH", os.environ.get("PATH", "")),),
        timeout_seconds=2,
        output_limit_bytes=16,
        report_path=report,
        candidate_root=tmp_path,
        artifact_root=tmp_path,
    )
    result = SubprocessSafetyToolRunner().run(request)
    assert result.exit_code == 0
    assert not result.timed_out
    assert result.stdout.strip() == "ok"
    assert result.report_path is None
