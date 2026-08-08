"""Bounded multi-attempt standalone kernel optimization use case."""

from __future__ import annotations

from typing import Sequence

from apex.core import (
    AgentBackendName,
    ApexError,
    ContractError,
    TaskStatus,
    new_identifier,
)
from apex.delivery import TaskResult, write_task_result
from apex.evaluation.safety import (
    SafetyGate,
    SafetyGateRequest,
    SafetyGateResult,
    SafetyRequirement,
    ToolVerificationPlan,
    VerificationPlan,
    VerificationPolicy,
)
from apex.execution import AgentRegistry
from apex.intake import TaskResolver, TaskSpec
from apex.ports import (
    AgentRequest,
    AgentTerminationKind,
    KernelMeasurementPort,
    SafetyToolRunRequest,
    SafetyToolRunResult,
    SafetyVerificationPort,
)
from apex.runtime import GpuLeaseManager, GpuLeaseReceipt, LocalGpuLeaseManager
from apex.storage import ArtifactReceipt

from .attempts import (
    AttemptSession,
    CompileCorrectnessReceipts,
    KernelAttemptOutcome,
    KernelOptimizeRequest,
    MeasurementEvidence,
    PreparedCandidate,
    RunSession,
    SafetyEvidence,
    select_best,
)
from .context import KernelContextBuilder
from .finalization import deliver_best, failure_result, finish_without_candidate, publish
from .lifecycle import (
    close_attempt,
    close_prepared,
    phase_isolation_receipt,
    safety_gate_request,
)
from .measurement import evaluate_kernel_measurement
from .run_record import KernelRunRecord, candidate_file_bytes
from .safety_bridge import (
    baseline_source_digest,
    materialize_safety_candidate,
    safety_profile,
    safety_rejection_reason,
    validate_safety_result,
)
from .verification import CandidateVerifier, candidate_source_digest
from .workspace import CandidateWorkspace


class _UnexpectedSafetyRunner:
    def run(self, request: SafetyToolRunRequest) -> SafetyToolRunResult:
        raise RuntimeError(f"no safety runtime is bound for {request.tool}")


class KernelOptimizeUseCase:
    """Search isolated candidates and deliver exactly one stable best bundle."""

    def __init__(
        self,
        *,
        agents: AgentRegistry,
        verifier: CandidateVerifier | None = None,
        contexts: KernelContextBuilder | None = None,
        safety_gate: SafetyVerificationPort[SafetyGateRequest, SafetyGateResult] | None = None,
        safety_policy: VerificationPolicy | None = None,
        safety_tools: Sequence[ToolVerificationPlan] = (),
        gpu_leases: GpuLeaseManager | None = None,
        measurement_evaluator: KernelMeasurementPort | None = None,
    ) -> None:
        self._agents = agents
        self._verifier = verifier or CandidateVerifier()
        self._contexts = contexts or KernelContextBuilder()
        self._safety_policy = safety_policy or VerificationPolicy.no_tools()
        self._safety_tools = tuple(safety_tools)
        enabled = tuple(
            rule.tool
            for rule in self._safety_policy.rules
            if rule.requirement is not SafetyRequirement.DISABLED
        )
        if tuple(tool.tool for tool in self._safety_tools) != enabled:
            raise ContractError(
                "trusted safety policy and tool plans disagree",
                reason_code="safety_plan_tool_set_mismatch",
            )
        self._safety_gate = safety_gate or SafetyGate(_UnexpectedSafetyRunner())
        self._gpu_leases = gpu_leases or LocalGpuLeaseManager()
        self._measurement_evaluator = measurement_evaluator

    @property
    def measurement_adapter_id(self) -> str | None:
        """Expose the composed trusted measurement authority without its internals."""

        evaluator = self._measurement_evaluator
        return evaluator.adapter_id if evaluator is not None else None

    def run(self, request: KernelOptimizeRequest) -> TaskResult:
        run_id = new_identifier("run")
        try:
            with self._gpu_leases.acquire(run_id) as lease:
                return self._run(request, run_id, lease.receipt)
        except ApexError as error:
            result = failure_result(request.task, error, run_id=run_id)
            write_task_result(result, request.result_json)
            return result

    def _run(
        self,
        request: KernelOptimizeRequest,
        run_id: str,
        gpu_lease: GpuLeaseReceipt,
    ) -> TaskResult:
        session = self._start_session(request, run_id, gpu_lease)
        try:
            result = self._execute_search(session)
            publish(session)
            return result
        except ApexError as error:
            session.record.fail_active(error.reason_code)
            publish(session)
            result = failure_result(
                request.task,
                error,
                run_id=session.run_id,
                session=session,
            )
            write_task_result(result, request.result_json)
            return result

    def _start_session(
        self,
        request: KernelOptimizeRequest,
        run_id: str,
        gpu_lease: GpuLeaseReceipt,
    ) -> RunSession:
        resolved = TaskResolver().resolve(request.task)
        run_root = request.task.results_dir / "runs" / run_id
        record = KernelRunRecord.create(
            run_id=run_id,
            root=run_root,
            initial_anchor_id=f"anchor-{resolved.resolution_hash[:16]}",
            dataset_split=request.task.dataset_split,
            data_visibility=request.task.data_visibility,
        )
        lease_artifact = record.record_gpu_lease(gpu_lease)
        return RunSession(
            request,
            resolved,
            run_id,
            run_root,
            record,
            gpu_lease,
            lease_artifact,
        )

    def _execute_search(self, session: RunSession) -> TaskResult:
        outcomes: list[KernelAttemptOutcome] = []
        for cycle in range(session.request.task.budget.max_iterations):
            attempt = self._start_attempt(session, cycle)
            outcome = self._execute_attempt(attempt)
            outcomes.append(outcome)
            if outcome.stop_search:
                break
        frozen = tuple(outcomes)
        best = select_best(frozen)
        if best is not None:
            return deliver_best(session, frozen, best)
        return finish_without_candidate(session, frozen)

    def _start_attempt(self, run: RunSession, cycle: int) -> AttemptSession:
        attempt_id = new_identifier("attempt")
        candidate = CandidateWorkspace.create(
            run.resolved,
            destination=run.run_root / "worktrees" / attempt_id,
        )
        run.record.start_attempt(attempt_id)
        context = self._contexts.compile(
            run.resolved,
            record=run.record,
            attempt_id=attempt_id,
            cycle=cycle,
        )
        return AttemptSession(run, attempt_id, cycle, candidate, context)

    def _execute_attempt(self, attempt: AttemptSession) -> KernelAttemptOutcome:
        agent_evidence = self._invoke_agent(attempt)
        if isinstance(agent_evidence, KernelAttemptOutcome):
            return agent_evidence
        prepared = self._prepare_candidate(attempt, agent_evidence)
        if isinstance(prepared, KernelAttemptOutcome):
            return prepared
        verified = self._verify_compile_correctness(attempt, prepared, agent_evidence)
        if isinstance(verified, KernelAttemptOutcome):
            return verified
        safety = self._verify_safety(attempt, prepared, verified)
        if isinstance(safety, KernelAttemptOutcome):
            return safety
        return self._measure_attempt(attempt, prepared, verified, safety)

    def _invoke_agent(
        self, attempt: AttemptSession
    ) -> tuple[ArtifactReceipt, ...] | KernelAttemptOutcome:
        run = attempt.run
        backend_name = run.request.backend_override or run.request.task.agent_backend
        result = self._agents.get(backend_name).run(
            self._agent_request(attempt, backend_name)
        )
        receipts = run.record.record_agent(attempt.attempt_id, result=result)
        evidence = (attempt.context.packet_receipt, *receipts)
        if result.candidate_capture_allowed:
            return evidence
        status = _agent_failure_status(result.termination_kind)
        reason = result.candidate_rejection_reason or "agent_failed"
        return close_attempt(
            attempt,
            status,
            reason,
            strategy=attempt.context.packet_receipt.digest,
            evidence=evidence,
            closure="reject",
            stop_search=status in {
                TaskStatus.TIMEOUT,
                TaskStatus.BUDGET_EXHAUSTED,
                TaskStatus.INFRASTRUCTURE_ERROR,
            }
            and result.termination_kind is not AgentTerminationKind.PROCESS_FAILED,
        )

    def _prepare_candidate(
        self,
        attempt: AttemptSession,
        evidence: tuple[ArtifactReceipt, ...],
    ) -> PreparedCandidate | KernelAttemptOutcome:
        run = attempt.run
        frozen = attempt.candidate.freeze(
            destination=run.run_root / "projections" / attempt.attempt_id
        )
        if not frozen.changed_files:
            return close_attempt(
                attempt,
                TaskStatus.NO_GAIN,
                "agent_made_no_source_change",
                strategy=attempt.context.packet_receipt.digest,
                evidence=evidence,
                closure="abort",
            )
        receipts = run.record.record_candidate(
            attempt.attempt_id,
            candidate_files=candidate_file_bytes(
                attempt.candidate.root, run.request.task.editable_files
            ),
            changed_files=frozen.changed_files,
        )
        normal_digest = candidate_source_digest(
            attempt.candidate.root, run.request.task.editable_files
        )
        profile = safety_profile(run.resolved)
        safety_candidate = materialize_safety_candidate(
            attempt.candidate.root,
            destination=run.run_root / "frozen" / attempt.attempt_id,
            profile=profile,
        )
        plan = VerificationPlan.create(
            run_id=run.run_id,
            candidate_id=attempt.attempt_id,
            anchor_generation=run.record.controller.state.anchor_generation,
            profile=profile,
            policy=self._safety_policy,
            source_digest=baseline_source_digest(run.resolved),
            candidate_digest=safety_candidate.candidate_digest,
            deployed_digest=normal_digest,
            tools=self._safety_tools,
        )
        return PreparedCandidate(
            normal_digest,
            frozen.changed_files,
            receipts,
            safety_candidate,
            plan,
        )

    def _verify_compile_correctness(
        self,
        attempt: AttemptSession,
        prepared: PreparedCandidate,
        prior: tuple[ArtifactReceipt, ...],
    ) -> CompileCorrectnessReceipts | KernelAttemptOutcome:
        run = attempt.run
        compile_result = self._verifier.compile(
            run.resolved,
            candidate_root=attempt.candidate.root,
            expected_source_digest=prepared.normal_source_digest,
        )
        compile_receipt = run.record.record_command(attempt.attempt_id, compile_result)
        evidence = (*prior, *prepared.candidate_receipts, compile_receipt)
        if not compile_result.passed:
            return close_prepared(
                attempt,
                prepared,
                TaskStatus.REJECTED,
                "compile_failed",
                evidence,
                closure="reject",
            )
        correctness = self._verifier.correctness(
            run.resolved,
            candidate_root=attempt.candidate.root,
            expected_source_digest=prepared.normal_source_digest,
        )
        correctness_receipt = run.record.record_command(attempt.attempt_id, correctness)
        evidence = (*evidence, correctness_receipt)
        if not correctness.passed:
            return close_prepared(
                attempt,
                prepared,
                TaskStatus.REJECTED,
                "correctness_failed",
                evidence,
                closure="reject",
            )
        return CompileCorrectnessReceipts(
            compile_receipt,
            correctness_receipt,
            evidence,
        )

    def _verify_safety(
        self,
        attempt: AttemptSession,
        prepared: PreparedCandidate,
        verified: CompileCorrectnessReceipts,
    ) -> SafetyEvidence | KernelAttemptOutcome:
        isolation = phase_isolation_receipt(attempt, prepared)
        result = self._safety_gate.evaluate(
            safety_gate_request(
                attempt,
                prepared,
                isolation,
                policy=self._safety_policy,
            )
        )
        validate_safety_result(prepared.safety_plan, isolation, result, self._safety_policy)
        receipt = attempt.run.record.record_safety(
            attempt.attempt_id,
            plan=prepared.safety_plan,
            isolation=isolation,
            result=result,
        )
        evidence = (*verified.evidence, receipt)
        if result.decision.allowed_to_measure and result.decision.promotion_eligible:
            return SafetyEvidence(result, receipt, evidence)
        reason = safety_rejection_reason(result)
        return close_prepared(
            attempt,
            prepared,
            TaskStatus.REJECTED,
            reason,
            evidence,
            safety=result,
            safety_receipt=receipt,
            closure="reject",
        )

    def _measure_attempt(
        self,
        attempt: AttemptSession,
        prepared: PreparedCandidate,
        verified: CompileCorrectnessReceipts,
        safety: SafetyEvidence,
    ) -> KernelAttemptOutcome:
        run = attempt.run
        performance = self._verifier.performance(
            run.resolved,
            candidate_root=attempt.candidate.root,
            expected_source_digest=prepared.normal_source_digest,
        )
        performance_receipt = run.record.record_command(attempt.attempt_id, performance)
        evidence = (*safety.evidence, performance_receipt)
        if not performance.passed:
            return close_prepared(
                attempt,
                prepared,
                TaskStatus.NO_MEASUREMENT,
                "performance_command_failed",
                evidence,
                safety=safety.result,
                safety_receipt=safety.receipt,
                closure="defer",
                measurement_fields={"measurement_status": "error"},
            )
        return self._evaluate_and_close(
            attempt,
            prepared,
            verified,
            safety,
            performance_receipt,
            evidence,
        )

    def _evaluate_and_close(
        self,
        attempt: AttemptSession,
        prepared: PreparedCandidate,
        verified: CompileCorrectnessReceipts,
        safety: SafetyEvidence,
        performance_receipt: ArtifactReceipt,
        evidence: tuple[ArtifactReceipt, ...],
    ) -> KernelAttemptOutcome:
        evaluated = self._evaluate_measurement(attempt, prepared, safety, evidence)
        if isinstance(evaluated, KernelAttemptOutcome):
            return evaluated
        measurement = evaluated.measurement
        verification = attempt.run.record.mark_verified(
            attempt.attempt_id,
            compile_receipt=verified.compile,
            correctness_receipt=verified.correctness,
            safety_receipt=safety.receipt,
            performance_receipt=performance_receipt,
            measurement_receipt=evaluated.receipt,
        )
        complete_evidence = (*evaluated.evidence, verification)
        if measurement is not None and not measurement.improved:
            reason = measurement.grade.promotion_reason_code
            return close_prepared(
                attempt,
                prepared,
                TaskStatus.NO_GAIN,
                reason,
                complete_evidence,
                safety=safety.result,
                safety_receipt=safety.receipt,
                closure="complete_revert",
                measurement=measurement,
            )
        reason = (
            "candidate_verified_by_trusted_measurement"
            if measurement is not None
            else "candidate_deferred_to_external_evaluator"
        )
        return close_prepared(
            attempt,
            prepared,
            TaskStatus.CANDIDATE_READY,
            reason,
            complete_evidence,
            safety=safety.result,
            safety_receipt=safety.receipt,
            closure="complete",
            measurement=measurement,
            eligible=True,
        )

    def _evaluate_measurement(
        self,
        attempt: AttemptSession,
        prepared: PreparedCandidate,
        safety: SafetyEvidence,
        evidence: tuple[ArtifactReceipt, ...],
    ) -> MeasurementEvidence | KernelAttemptOutcome:
        if attempt.run.resolved.task.measurement is None:
            if _uses_external_evaluator(attempt.run.resolved.task):
                return MeasurementEvidence(None, None, evidence)
            return close_prepared(
                attempt,
                prepared,
                TaskStatus.NO_MEASUREMENT,
                "measurement_contract_missing",
                evidence,
                safety=safety.result,
                safety_receipt=safety.receipt,
                closure="defer",
                measurement_fields={"measurement_status": "not_configured"},
            )
        try:
            measurement = evaluate_kernel_measurement(
                attempt.run.resolved,
                candidate_root=attempt.candidate.root,
                run_id=attempt.run.run_id,
                attempt_id=attempt.attempt_id,
                output_root=(
                    attempt.run.run_root / "measurements" / attempt.attempt_id
                ),
                evaluator=self._measurement_evaluator,
            )
        except ApexError as error:
            attempt.run.record.record_measurement_error(
                attempt.attempt_id,
                reason_code=error.reason_code,
            )
            return close_prepared(
                attempt,
                prepared,
                TaskStatus.NO_MEASUREMENT,
                error.reason_code,
                evidence,
                safety=safety.result,
                safety_receipt=safety.receipt,
                closure="defer",
                measurement_fields={"measurement_status": "error"},
            )
        measurement_receipt: ArtifactReceipt | None = None
        if measurement is not None:
            measurement_receipt = attempt.run.record.record_measurement(
                attempt.attempt_id,
                artifact=measurement.artifact,
                execution=measurement.execution,
                grade=measurement.grade,
                harness_receipt=attempt.context.harness_receipt,
            )
            evidence = (*evidence, measurement_receipt)
            if not measurement.reward_eligible:
                reason = measurement.grade.reason_code or "robust_measurement_unavailable"
                return close_prepared(
                    attempt,
                    prepared,
                    TaskStatus.NO_MEASUREMENT,
                    reason,
                    evidence,
                    safety=safety.result,
                    safety_receipt=safety.receipt,
                    closure="defer",
                    measurement=measurement,
                )
        return MeasurementEvidence(measurement, measurement_receipt, evidence)

    def _agent_request(
        self,
        attempt: AttemptSession,
        backend: AgentBackendName,
    ) -> AgentRequest:
        request = attempt.run.request
        task = request.task
        return AgentRequest(
            run_id=attempt.run.run_id,
            attempt_id=attempt.attempt_id,
            backend=backend,
            prompt=attempt.context.prompt,
            workspace=attempt.candidate.root,
            allowed_files=task.editable_files,
            model=request.model_override or task.agent_options.model,
            effort=request.effort_override or task.agent_options.effort,
            max_turns=task.budget.max_turns,
            timeout_seconds=task.budget.timeout_seconds,
            runtime_closure_sha256=task.agent_options.runtime_closure_sha256,
        )


def _uses_external_evaluator(task: TaskSpec) -> bool:
    recipe = task.recipe
    return recipe is not None and recipe.provenance == "external_evaluator"


def _agent_failure_status(kind: AgentTerminationKind) -> TaskStatus:
    if kind is AgentTerminationKind.TIMEOUT:
        return TaskStatus.TIMEOUT
    if kind is AgentTerminationKind.TURN_OVERRUN:
        return TaskStatus.BUDGET_EXHAUSTED
    return TaskStatus.INFRASTRUCTURE_ERROR


__all__ = ["KernelOptimizeRequest", "KernelOptimizeUseCase"]
