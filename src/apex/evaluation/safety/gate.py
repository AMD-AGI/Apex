"""Fail-closed orchestration-neutral safety evaluation gate."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from apex.core import ContractError, sha256_bytes, sha256_file
from apex.ports import SafetyToolRunRequest, SafetyToolRunResult, SafetyToolRunner

from .artifact_validation import validate_artifacts
from .plan import FrozenCandidate, PhaseIsolationReceipt, ToolVerificationPlan, VerificationPlan
from .policy import SafetyRequirement, VerificationPolicy, decide_safety
from .profile import CapabilityStatus
from .results import (
    EvidenceArtifactReceipt,
    ExecutionStatus,
    FindingStatus,
    LineageReceipt,
    RESULT_SCHEMA_VERSION,
    TOOL_REPORT_SCHEMA_VERSION,
    SafetyGateResult,
    ToolEvaluation,
    parse_execution_status,
    parse_finding_status,
)


@dataclass(frozen=True, slots=True)
class SafetyGateRequest:
    """Current controller state plus an immutable evaluator-owned plan."""

    plan: VerificationPlan
    policy: VerificationPolicy
    frozen_candidate: FrozenCandidate
    isolation_receipt: PhaseIsolationReceipt
    artifact_root: Path
    current_run_id: str
    current_candidate_id: str
    current_anchor_generation: int
    current_deployed_digest: str

    def __post_init__(self) -> None:
        if (
            not self.artifact_root.is_absolute()
            or self.artifact_root != self.artifact_root.resolve(strict=False)
        ):
            raise ContractError("safety artifact root must be absolute", "invalid_safety_artifact_root")
        if self.current_anchor_generation < 0:
            raise ContractError("invalid current anchor generation", "stale_safety_plan")


class SafetyGate:
    """Validate phase isolation, run bounded tools, and normalize receipts.

    The gate never executes performance timing.  Every artifact accepted here
    is explicitly diagnostic-only and its digest is exported as forbidden for
    later timing adapters.
    """

    def __init__(self, runner: SafetyToolRunner) -> None:
        self._runner = runner

    def evaluate(self, request: SafetyGateRequest) -> SafetyGateResult:
        errors = self._preflight(request)
        rules_by_tool = {rule.tool: rule for rule in request.policy.rules}
        enabled = tuple(
            tool
            for tool in request.plan.tools
            if tool.tool in rules_by_tool
            and rules_by_tool[tool.tool].requirement is not SafetyRequirement.DISABLED
        )
        if errors:
            evaluations = tuple(self._not_run(tool, "safety_gate_preflight_failed") for tool in enabled)
        else:
            evaluations = tuple(self._evaluate_tool(request, tool) for tool in enabled)
        decision = decide_safety(evaluations, policy=request.policy, blocking_errors=errors)
        return SafetyGateResult(
            schema_version=RESULT_SCHEMA_VERSION,
            run_id=request.plan.run_id,
            candidate_id=request.plan.candidate_id,
            anchor_generation=request.plan.anchor_generation,
            plan_fingerprint=request.plan.fingerprint,
            policy_fingerprint=request.plan.policy_fingerprint,
            source_digest=request.plan.source_digest,
            candidate_digest=request.plan.candidate_digest,
            deployed_digest=request.plan.deployed_digest,
            isolation_receipt_fingerprint=request.isolation_receipt.fingerprint,
            evaluations=evaluations,
            decision=decision,
            gate_errors=tuple(errors),
        )

    def _preflight(self, request: SafetyGateRequest) -> tuple[str, ...]:
        plan = request.plan
        receipt = request.isolation_receipt
        errors: list[str] = []

        def require(condition: bool, code: str) -> None:
            if not condition:
                errors.append(code)

        require(plan.policy_fingerprint == request.policy.fingerprint, "safety_policy_fingerprint_mismatch")
        require(plan.run_id == request.current_run_id, "stale_safety_run")
        require(plan.candidate_id == request.current_candidate_id, "stale_safety_candidate")
        require(plan.anchor_generation == request.current_anchor_generation, "stale_safety_anchor")
        require(plan.deployed_digest == request.current_deployed_digest, "stale_deployed_digest")
        require(
            plan.candidate_digest == request.frozen_candidate.candidate_digest,
            "candidate_digest_mismatch",
        )
        require(
            plan.profile.submission_paths == request.frozen_candidate.submission_paths,
            "candidate_path_set_mismatch",
        )

        require(receipt.run_id == plan.run_id, "phase_isolation_run_mismatch")
        require(receipt.plan_fingerprint == plan.fingerprint, "phase_isolation_plan_mismatch")
        require(receipt.anchor_generation == plan.anchor_generation, "phase_isolation_anchor_mismatch")
        require(receipt.candidate_digest == plan.candidate_digest, "phase_isolation_candidate_mismatch")
        require(receipt.frozen_root == str(request.frozen_candidate.root), "phase_isolation_root_mismatch")
        require(
            receipt.evaluator_artifact_root == str(request.artifact_root),
            "phase_isolation_artifact_mismatch",
        )
        require(receipt.complete, "phase_isolation_incomplete")

        expected_tools = tuple(
            rule.tool for rule in request.policy.rules if rule.requirement is not SafetyRequirement.DISABLED
        )
        planned_tools = tuple(tool.tool for tool in plan.tools)
        require(planned_tools == expected_tools, "safety_plan_tool_set_mismatch")

        try:
            request.frozen_candidate.verify()
        except ContractError as error:
            errors.append(error.reason_code)

        artifact_root = request.artifact_root
        candidate_root = request.frozen_candidate.root
        try:
            artifact_resolved = artifact_root.resolve(strict=False)
            candidate_resolved = candidate_root.resolve(strict=True)
            overlaps_candidate = artifact_resolved.is_relative_to(
                candidate_resolved
            ) or candidate_resolved.is_relative_to(artifact_resolved)
            if overlaps_candidate:
                errors.append("safety_phase_path_overlap")
            elif artifact_root.exists() and (artifact_root.is_symlink() or not artifact_root.is_dir()):
                errors.append("invalid_safety_artifact_root")
            else:
                artifact_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                if any(artifact_root.iterdir()):
                    errors.append("safety_artifact_root_not_empty")
        except OSError:
            errors.append("invalid_safety_artifact_root")
        return tuple(dict.fromkeys(errors))

    def _evaluate_tool(
        self,
        request: SafetyGateRequest,
        tool_plan: ToolVerificationPlan,
    ) -> ToolEvaluation:
        capability = tool_plan.capability.effective
        assert capability is not None
        if capability.status is not CapabilityStatus.READY:
            return self._not_run(
                tool_plan,
                capability.reason_code or f"capability_{capability.status.value}",
            )

        tool_root = request.artifact_root / tool_plan.tool
        try:
            tool_root.mkdir(mode=0o700)
        except OSError:
            return self._failed_without_process(tool_plan, "safety_tool_artifact_setup_failed")
        report_path = tool_root / "report.json"
        environment = dict(tool_plan.environment)
        reserved = {
            "APEX_FROZEN_CANDIDATE_ROOT": str(request.frozen_candidate.root),
            "APEX_SAFETY_ARTIFACT_ROOT": str(tool_root),
            "APEX_SAFETY_PLAN_FINGERPRINT": request.plan.fingerprint,
            "APEX_SAFETY_REPORT_PATH": str(report_path),
        }
        if set(environment).intersection(reserved):
            return self._failed_without_process(tool_plan, "reserved_safety_environment_override")
        environment.update(reserved)
        run_request = SafetyToolRunRequest(
            tool=tool_plan.tool,
            plan_fingerprint=request.plan.fingerprint,
            argv=tool_plan.argv,
            cwd=tool_root,
            environment=tuple(sorted(environment.items())),
            timeout_seconds=tool_plan.timeout_seconds,
            output_limit_bytes=tool_plan.output_limit_bytes,
            report_path=report_path,
            candidate_root=request.frozen_candidate.root,
            artifact_root=tool_root,
        )
        try:
            process = self._runner.run(run_request)
        except Exception:
            return self._failed_without_process(tool_plan, "safety_runner_exception")
        return self._normalize_process(request.plan, tool_plan, run_request, process)

    @staticmethod
    def _not_run(tool_plan: ToolVerificationPlan, reason: str) -> ToolEvaluation:
        capability = tool_plan.capability.effective
        assert capability is not None
        finding = FindingStatus.NOT_EVALUATED
        return ToolEvaluation(
            tool=tool_plan.tool,
            capability=capability.status,
            execution=ExecutionStatus.NOT_RUN,
            finding=finding,
            reason_codes=(reason,),
        )

    @staticmethod
    def _failed_without_process(tool_plan: ToolVerificationPlan, reason: str) -> ToolEvaluation:
        return ToolEvaluation(
            tool=tool_plan.tool,
            capability=CapabilityStatus.READY,
            execution=ExecutionStatus.TOOL_ERROR,
            finding=FindingStatus.INCONCLUSIVE,
            reason_codes=(reason,),
        )

    def _normalize_process(
        self,
        plan: VerificationPlan,
        tool_plan: ToolVerificationPlan,
        request: SafetyToolRunRequest,
        process: SafetyToolRunResult,
    ) -> ToolEvaluation:
        stdout_bytes = process.stdout.encode("utf-8", errors="replace")
        stderr_bytes = process.stderr.encode("utf-8", errors="replace")
        stdout_truncated = (
            process.stdout_truncated
            or len(stdout_bytes) > tool_plan.output_limit_bytes
        )
        stderr_truncated = (
            process.stderr_truncated
            or len(stderr_bytes) > tool_plan.output_limit_bytes
        )
        execution = _execution_status(process)
        issues = _process_issues(
            process,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
        parsed, report_issues = _load_process_report(
            plan, tool_plan, request, process
        )
        issues.extend(report_issues)
        finding = _derive_finding(
            process,
            parsed,
            execution=execution,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            issues=issues,
        )
        if process.exit_code not in (0, None):
            issues.append("safety_tool_nonzero_exit")
        if not issues and finding is FindingStatus.CLEAN:
            issues.append("safety_clean")
        elif not issues and finding is FindingStatus.FOUND:
            issues.append("safety_finding")
        artifacts = _diagnostic_artifacts(parsed, process, request)
        return ToolEvaluation(
            tool=tool_plan.tool,
            capability=CapabilityStatus.READY,
            execution=execution,
            finding=finding,
            reason_codes=tuple(dict.fromkeys(issues)),
            lineage=parsed.lineage,
            artifacts=artifacts,
            exit_code=process.exit_code,
            timed_out=process.timed_out,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            stdout_digest=sha256_bytes(stdout_bytes),
            stderr_digest=sha256_bytes(stderr_bytes),
            duration_seconds=process.duration_seconds,
        )


def _execution_status(process: SafetyToolRunResult) -> ExecutionStatus:
    if process.timed_out:
        return ExecutionStatus.TIMEOUT
    if process.exit_code == 0:
        return ExecutionStatus.COMPLETED
    return ExecutionStatus.TOOL_ERROR


def _process_issues(
    process: SafetyToolRunResult,
    *,
    stdout_truncated: bool,
    stderr_truncated: bool,
) -> list[str]:
    issues: list[str] = []
    if process.timed_out:
        issues.append("safety_tool_timeout")
    if stdout_truncated or stderr_truncated:
        issues.append("safety_tool_output_truncated")
    if process.report_origin != "evaluator":
        issues.append("untrusted_safety_report_origin")
    return issues


@dataclass(frozen=True, slots=True)
class _ParsedReport:
    clean: bool = False
    confirmed_finding: bool = False
    lineage: LineageReceipt | None = None
    artifacts: tuple[EvidenceArtifactReceipt, ...] = ()
    issues: tuple[str, ...] = ()


def _load_process_report(
    plan: VerificationPlan,
    tool_plan: ToolVerificationPlan,
    request: SafetyToolRunRequest,
    process: SafetyToolRunResult,
) -> tuple[_ParsedReport, list[str]]:
    if process.report_path is None:
        return _ParsedReport(), ["missing_safety_report"]
    if process.report_path != request.report_path:
        return _ParsedReport(), ["safety_report_path_mismatch"]
    if not _valid_evaluator_report_path(process.report_path, request.artifact_root):
        return _ParsedReport(), ["invalid_safety_report_path"]
    try:
        if process.report_path.stat().st_size > tool_plan.output_limit_bytes:
            return _ParsedReport(), ["safety_report_truncated"]
        parsed = _parse_tool_report(
            process.report_path, request.artifact_root, plan, tool_plan
        )
        return parsed, list(parsed.issues)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ContractError,
        ValueError,
        TypeError,
    ):
        return _ParsedReport(), ["invalid_or_truncated_safety_report"]


def _derive_finding(
    process: SafetyToolRunResult,
    parsed: _ParsedReport,
    *,
    execution: ExecutionStatus,
    stdout_truncated: bool,
    stderr_truncated: bool,
    issues: Sequence[str],
) -> FindingStatus:
    # An exact-lineage finding remains a hard reject even after nonzero exit.
    if parsed.confirmed_finding:
        return FindingStatus.FOUND
    if (
        execution is ExecutionStatus.COMPLETED
        and not stdout_truncated
        and not stderr_truncated
        and parsed.clean
        and process.report_origin == "evaluator"
        and not issues
    ):
        return FindingStatus.CLEAN
    return FindingStatus.INCONCLUSIVE


def _diagnostic_artifacts(
    parsed: _ParsedReport,
    process: SafetyToolRunResult,
    request: SafetyToolRunRequest,
) -> tuple[EvidenceArtifactReceipt, ...]:
    artifacts = list(parsed.artifacts)
    if process.report_path is not None and _valid_evaluator_report_path(
        process.report_path, request.artifact_root
    ):
        artifacts.append(
            EvidenceArtifactReceipt(
                role="tool_report",
                path=f"{request.artifact_root.name}/report.json",
                digest=sha256_file(process.report_path),
                size=process.report_path.stat().st_size,
            )
        )
    artifacts.sort(key=lambda artifact: (artifact.role, artifact.path))
    return tuple(artifacts)


def _valid_evaluator_report_path(report_path: Path, artifact_root: Path) -> bool:
    try:
        metadata = report_path.lstat()
        return (
            not artifact_root.parent.is_symlink()
            and not artifact_root.is_symlink()
            and not stat.S_ISLNK(metadata.st_mode)
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and report_path.resolve(strict=True).is_relative_to(artifact_root.resolve(strict=True))
        )
    except OSError:
        return False


def _parse_tool_report(
    report_path: Path,
    artifact_root: Path,
    plan: VerificationPlan,
    tool_plan: ToolVerificationPlan,
) -> _ParsedReport:
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        return _ParsedReport(issues=("invalid_safety_report_shape",))
    issues, header_valid = _validate_report_header(raw, plan, tool_plan)
    artifacts, artifact_issues = validate_artifacts(
        raw.get("artifacts"), artifact_root, plan
    )
    issues.extend(artifact_issues)
    case_states, case_issues = _validate_cases(raw.get("cases"), tool_plan)
    issues.extend(case_issues)
    complete = raw.get("complete") is True
    if not complete:
        issues.append("safety_report_incomplete")

    core_receipt_valid = header_valid and not artifact_issues and not case_issues
    lineage = None
    if core_receipt_valid:
        lineage = LineageReceipt(
            identity=tool_plan.identity,
            source_digest=plan.source_digest,
            candidate_digest=plan.candidate_digest,
            deployed_digest=plan.deployed_digest,
            positive_control_digest=tool_plan.positive_control_digest,
            plan_fingerprint=plan.fingerprint,
        )
    confirmed_finding = core_receipt_valid and any(
        finding is FindingStatus.FOUND for _, finding in case_states
    )
    clean = (
        core_receipt_valid
        and complete
        and bool(case_states)
        and all(
            execution is ExecutionStatus.COMPLETED and finding is FindingStatus.CLEAN
            for execution, finding in case_states
        )
    )
    return _ParsedReport(
        clean=clean,
        confirmed_finding=confirmed_finding,
        lineage=lineage,
        artifacts=artifacts,
        issues=tuple(dict.fromkeys(issues)),
    )


def _validate_report_header(
    raw: Mapping[object, object],
    plan: VerificationPlan,
    tool_plan: ToolVerificationPlan,
) -> tuple[list[str], bool]:
    expected_lineage = {
        "source_digest": plan.source_digest,
        "candidate_digest": plan.candidate_digest,
        "deployed_digest": plan.deployed_digest,
    }
    identity = raw.get("identity")
    lineage = raw.get("lineage")
    control = raw.get("positive_control")
    checks = (
        (
            raw.get("schema_version") == TOOL_REPORT_SCHEMA_VERSION,
            "safety_report_schema_mismatch",
        ),
        (raw.get("plan_fingerprint") == plan.fingerprint, "safety_report_plan_mismatch"),
        (
            isinstance(identity, Mapping)
            and dict(identity) == tool_plan.identity.to_dict(),
            "safety_tool_identity_mismatch",
        ),
        (
            isinstance(lineage, Mapping) and dict(lineage) == expected_lineage,
            "safety_lineage_digest_mismatch",
        ),
        (
            isinstance(control, Mapping)
            and control.get("digest") == tool_plan.positive_control_digest
            and control.get("status") == "passed",
            "safety_positive_control_failed",
        ),
    )
    return [reason for valid, reason in checks if not valid], all(
        valid for valid, _ in checks
    )


def _validate_cases(
    value: object,
    tool_plan: ToolVerificationPlan,
) -> tuple[tuple[tuple[ExecutionStatus, FindingStatus], ...], tuple[str, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return (), ("invalid_safety_cases",)
    by_id: dict[str, tuple[ExecutionStatus, FindingStatus]] = {}
    issues: list[str] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            issues.append("invalid_safety_case")
            continue
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or case_id in by_id:
            issues.append("invalid_safety_case_id")
            continue
        try:
            execution = parse_execution_status(raw.get("execution"))
            finding = parse_finding_status(raw.get("finding"))
        except ContractError:
            issues.append("invalid_safety_case_state")
            continue
        if raw.get("dispatch_digest") != tool_plan.identity.dispatch_digest:
            issues.append("safety_dispatch_digest_mismatch")
        if finding is FindingStatus.CLEAN and execution is not ExecutionStatus.COMPLETED:
            issues.append("false_clean_safety_case")
        by_id[case_id] = (execution, finding)
    if tuple(sorted(by_id)) != tool_plan.cases:
        issues.append("safety_case_set_mismatch")
    return tuple(by_id[case] for case in sorted(by_id)), tuple(dict.fromkeys(issues))


__all__ = ["SafetyGate", "SafetyGateRequest"]
