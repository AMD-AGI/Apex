"""Replayable evaluator phases for a user-confirmed chat-started kernel draft."""

from __future__ import annotations

from pathlib import Path

from apex.core import ContractError, IntegrityError, canonical_json_bytes
from apex.evaluation.safety import (
    PhaseIsolationReceipt,
    SafetyGate,
    SafetyGateRequest,
    VerificationPlan,
    VerificationPolicy,
)
from apex.ports import (
    KernelMeasurementPort,
    SafetyToolRunRequest,
    SafetyToolRunResult,
)
from apex.runtime import ApexExecutionIdentity, GpuLeaseManager, require_gpu_lease_heartbeat
from apex.storage import ArtifactReceipt

from .formal_campaign import FormalKernelCampaign
from .formal_authority import FormalEvaluationAuthorityProvider
from .formal_delivery import build_formal_bundle
from .formal_evidence import (
    attempt_event,
    event_receipt,
    kind_event,
    load_grade,
    require_passed,
)
from .formal_measurement_recording import (
    record_measurement_capture,
    record_measurement_error,
)
from .formal_result import FormalEvaluatorResult
from .formal_measurement_execution import execute_formal_measurement
from .gpu_recording import record_gpu_lease_heartbeat
from .reward_recording import (
    record_attempt_gate_reward,
)
from .safety_bridge import (
    baseline_source_digest,
    materialize_safety_candidate,
    safety_profile,
    validate_safety_result,
)
from .verification import CandidateVerifier
from ..execution_identity_recording import recorded_execution_identity_reason


class _NoSafetyRuntime:
    def run(self, request: SafetyToolRunRequest) -> SafetyToolRunResult:
        raise RuntimeError(f"no safety runtime is configured for {request.tool}")


class KernelFormalEvaluator:
    """Own fixed verification adapters while the coding backend only requests work."""

    def __init__(
        self,
        *,
        verifier: CandidateVerifier,
        gpu_leases: GpuLeaseManager,
        measurement_evaluator: KernelMeasurementPort,
        authority_provider: FormalEvaluationAuthorityProvider | None = None,
        execution_identity: ApexExecutionIdentity,
    ) -> None:
        self._verifier = verifier
        self._gpu_leases = gpu_leases
        self._measurement_evaluator = measurement_evaluator
        self._authority_provider = authority_provider
        self._execution_identity = execution_identity
        self._safety_policy = VerificationPolicy.no_tools()
        self._safety_gate = SafetyGate(_NoSafetyRuntime())

    def compile(
        self,
        campaign: FormalKernelCampaign,
        *,
        confirmed_draft_digest: str,
        requested_devices: str | None,
    ) -> FormalEvaluatorResult:
        identity_reason = recorded_execution_identity_reason(
            campaign.record.artifacts, campaign.record.iter_events(), self._execution_identity
        )
        if identity_reason is not None:
            return _unverified_compile(identity_reason)
        contract = campaign.confirm(
            confirmed_draft_digest, self._authority_provider
        )
        if contract is None:
            return _unverified_compile("evaluation_authority_missing")
        projection = campaign.capture_candidate()
        try:
            attempt_id = f"attempt-{projection.source_digest[:24]}"
            existing = attempt_event(
                campaign, attempt_id, "compile_result", required=False
            )
            if existing is not None:
                return _gate_projection(
                    campaign, attempt_id, contract.digest, projection.source_digest, existing
                )
            frozen = attempt_event(
                campaign, attempt_id, "candidate_frozen", required=False
            )
            if frozen is None:
                campaign.record.start_attempt(attempt_id)
                campaign.record.record_candidate(
                    attempt_id,
                    candidate_files=projection.files,
                    changed_files=projection.changed_files,
                    source_digest=projection.source_digest,
                )
            else:
                _validate_candidate_event(frozen, projection.source_digest)
            with self._gpu_leases.acquire(
                campaign.record.run_id, requested_devices=requested_devices
            ) as lease:
                require_gpu_lease_heartbeat(lease)
                campaign.record.record_gpu_lease(
                    lease.receipt, attempt_id=attempt_id, phase="compile"
                )
                evidence = self._verifier.compile(
                    projection.resolved,
                    candidate_root=projection.root,
                    expected_source_digest=projection.source_digest,
                )
                heartbeat = require_gpu_lease_heartbeat(lease)
                record_gpu_lease_heartbeat(
                    campaign.record,
                    heartbeat,
                    attempt_id=attempt_id,
                    phase="compile",
                )
            receipt = campaign.record.record_command(attempt_id, evidence)
            if not evidence.passed:
                record_attempt_gate_reward(
                    campaign.record,
                    attempt_id,
                    stage="compile",
                    command_receipt=receipt,
                )
                campaign.record.reject_attempt(attempt_id, "compile_failed")
            event = attempt_event(campaign, attempt_id, "compile_result")
            assert event is not None
            return _gate_projection(
                campaign, attempt_id, contract.digest, projection.source_digest, event
            )
        finally:
            del projection

    def correctness(
        self,
        campaign: FormalKernelCampaign,
        *,
        attempt_id: str,
        contract_digest: str,
        candidate_digest: str,
        requested_devices: str | None,
    ) -> FormalEvaluatorResult:
        _require_contract(campaign, contract_digest)
        compile_event = require_passed(campaign, attempt_id, "compile_result")
        existing = attempt_event(
            campaign, attempt_id, "correctness_result", required=False
        )
        if existing is not None:
            return _gate_projection(
                campaign, attempt_id, contract_digest, candidate_digest, existing
            )
        files = campaign.candidate_files(attempt_id)
        with campaign.project(files, phase="correctness") as projection:
            _require_candidate_digest(projection.source_digest, candidate_digest)
            with self._gpu_leases.acquire(
                campaign.record.run_id, requested_devices=requested_devices
            ) as lease:
                require_gpu_lease_heartbeat(lease)
                campaign.record.record_gpu_lease(
                    lease.receipt, attempt_id=attempt_id, phase="correctness"
                )
                evidence = self._verifier.correctness(
                    projection.resolved,
                    candidate_root=projection.root,
                    expected_source_digest=candidate_digest,
                )
                heartbeat = require_gpu_lease_heartbeat(lease)
                record_gpu_lease_heartbeat(
                    campaign.record,
                    heartbeat,
                    attempt_id=attempt_id,
                    phase="correctness",
                )
            receipt = campaign.record.record_command(attempt_id, evidence)
        if not evidence.passed:
            record_attempt_gate_reward(
                campaign.record,
                attempt_id,
                stage="correctness",
                command_receipt=receipt,
            )
            campaign.record.reject_attempt(attempt_id, "correctness_failed")
        event = attempt_event(campaign, attempt_id, "correctness_result")
        assert event is not None
        del compile_event
        return _gate_projection(
            campaign, attempt_id, contract_digest, candidate_digest, event
        )

    def measure(
        self,
        campaign: FormalKernelCampaign,
        *,
        attempt_id: str,
        contract_digest: str,
        candidate_digest: str,
        requested_devices: str | None,
    ) -> FormalEvaluatorResult:
        _require_contract(campaign, contract_digest)
        require_passed(campaign, attempt_id, "compile_result")
        require_passed(campaign, attempt_id, "correctness_result")
        existing = kind_event(
            campaign, attempt_id, "kernel_measurement_capture", required=False
        )
        if existing is not None:
            raw = event_receipt(existing, "raw_measurement")
            return _measurement_projection(
                attempt_id, contract_digest, candidate_digest, raw
            )
        if campaign.task.measurement is None:
            return self._no_measurement(campaign, attempt_id, "measurement_contract_missing")
        files = campaign.candidate_files(attempt_id)
        with campaign.project(files, phase="measurement") as projection:
            _require_candidate_digest(projection.source_digest, candidate_digest)
            safety_receipt, allowed_to_measure, safety_reason = (
                self._record_no_tools_safety(
                    campaign, attempt_id, projection.root, candidate_digest
                )
            )
            if not allowed_to_measure:
                return self._no_measurement(campaign, attempt_id, safety_reason)
            execution = execute_formal_measurement(
                campaign,
                projection=projection,
                attempt_id=attempt_id,
                candidate_digest=candidate_digest,
                requested_devices=requested_devices,
                gpu_leases=self._gpu_leases,
                verifier=self._verifier,
                evaluator=self._measurement_evaluator,
            )
            if execution.capture is None:
                return self._no_measurement(
                    campaign,
                    attempt_id,
                    execution.reason_code or "measurement_capture_failed",
                )
            harness = _harness_receipt(campaign)
            raw, execution = record_measurement_capture(
                campaign.record,
                attempt_id,
                capture=execution.capture,
                harness_receipt=harness,
            )
        del safety_receipt, execution
        return _measurement_projection(
            attempt_id, contract_digest, candidate_digest, raw
        )

    def grade(
        self,
        campaign: FormalKernelCampaign,
        *,
        attempt_id: str | None,
        contract_digest: str | None,
        candidate_digest: str | None,
    ) -> FormalEvaluatorResult:
        contract = campaign.authorized_contract
        if contract is None:
            return _unverified_grade("evaluation_authority_missing")
        if contract_digest is not None and contract_digest != contract.digest:
            raise IntegrityError(
                "Grade request names another evaluation contract",
                "evaluation_authority_mismatch",
            )
        if not contract_digest or not attempt_id or not candidate_digest:
            return _unverified_grade("no_measurement")
        capture_event = kind_event(
            campaign, attempt_id, "kernel_measurement_capture", required=False
        )
        if capture_event is None:
            return _unverified_grade("no_measurement")
        evaluation, harness = load_grade(campaign, attempt_id)
        if evaluation.execution.candidate_source_sha256 != candidate_digest:
            raise IntegrityError(
                "Measurement candidate digest differs from the request",
                "candidate_digest_mismatch",
            )
        existing = attempt_event(
            campaign, attempt_id, "measurement_result", required=False
        )
        if existing is None:
            grade_receipt = campaign.record.record_measurement(
                attempt_id,
                artifact=evaluation.artifact,
                execution=evaluation.execution,
                grade=evaluation.grade,
                harness_receipt=harness,
            )
            self._close_grade(
                campaign, attempt_id, evaluation, grade_receipt
            )
        else:
            grade_receipt = event_receipt(existing, "kernel_grade")
        grade = evaluation.grade
        return FormalEvaluatorResult(
            {
                "schema": "apex.kernel-grade-capability/v1",
                "status": "verified" if evaluation.reward_eligible else "no_measurement",
                "attempt_id": attempt_id,
                "contract_digest": contract.digest,
                "candidate_digest": candidate_digest,
                "measurement_status": grade.measurement_status.value,
                "reward": grade.reward,
                "srobust": grade.srobust,
                "promotion_eligible": grade.promotion_eligible,
                "reason_code": grade.reason_code or grade.promotion_reason_code,
                "grade_receipt": grade_receipt.to_dict(),
            },
            (grade_receipt,),
            evaluation.reward_eligible,
        )

    def build_bundle(
        self,
        campaign: FormalKernelCampaign,
        *,
        attempt_id: str,
        contract_digest: str,
        candidate_digest: str,
        finish: bool = True,
    ) -> FormalEvaluatorResult:
        return build_formal_bundle(
            campaign,
            attempt_id=attempt_id,
            contract_digest=contract_digest,
            candidate_digest=candidate_digest,
            finish=finish,
        )

    def _record_no_tools_safety(
        self,
        campaign: FormalKernelCampaign,
        attempt_id: str,
        candidate_root: Path,
        candidate_digest: str,
    ) -> tuple[ArtifactReceipt, bool, str]:
        existing = attempt_event(campaign, attempt_id, "safety_result", required=False)
        if existing is not None:
            reasons = existing.payload.get("reason_codes")
            reason = (
                str(reasons[0])
                if isinstance(reasons, list) and reasons
                else "phase_isolation_incomplete"
            )
            return (
                event_receipt(existing, "safety_result"),
                existing.payload.get("allowed_to_measure") is True,
                reason,
            )
        profile = safety_profile(campaign.resolved)
        frozen = materialize_safety_candidate(
            candidate_root,
            destination=candidate_root.parent / "safety-candidate",
            profile=profile,
        )
        plan = VerificationPlan.create(
            run_id=campaign.record.run_id,
            candidate_id=attempt_id,
            anchor_generation=campaign.record.controller.state.anchor_generation,
            profile=profile,
            policy=self._safety_policy,
            source_digest=baseline_source_digest(campaign.resolved),
            candidate_digest=frozen.candidate_digest,
            deployed_digest=candidate_digest,
            tools=(),
        )
        artifact_root = campaign.record.root / "safety" / attempt_id
        isolation = PhaseIsolationReceipt(
            run_id=campaign.record.run_id,
            plan_fingerprint=plan.fingerprint,
            anchor_generation=plan.anchor_generation,
            candidate_digest=plan.candidate_digest,
            frozen_root=str(frozen.root),
            evaluator_artifact_root=str(artifact_root),
            agent_process_tree_terminated=False,
            credentials_revoked=False,
            tool_channels_revoked=False,
            report_directory_hidden_from_agent=False,
            candidate_read_only=True,
        )
        result = self._safety_gate.evaluate(
            SafetyGateRequest(
                plan=plan,
                policy=self._safety_policy,
                frozen_candidate=frozen,
                isolation_receipt=isolation,
                artifact_root=artifact_root,
                current_run_id=campaign.record.run_id,
                current_candidate_id=attempt_id,
                current_anchor_generation=plan.anchor_generation,
                current_deployed_digest=candidate_digest,
            )
        )
        validate_safety_result(plan, isolation, result, self._safety_policy)
        receipt = campaign.record.record_safety(
            attempt_id, plan=plan, isolation=isolation, result=result
        )
        reason = result.decision.reason_codes[0] if result.decision.reason_codes else (
            "phase_isolation_incomplete"
        )
        return receipt, result.decision.allowed_to_measure, reason

    def _close_grade(
        self,
        campaign: FormalKernelCampaign,
        attempt_id: str,
        evaluation,
        grade_receipt: ArtifactReceipt,
    ) -> None:
        if not evaluation.reward_eligible:
            campaign.record.defer_attempt(
                attempt_id,
                evaluation.grade.reason_code or "robust_measurement_unavailable",
            )
            return
        compile_receipt = event_receipt(
            require_passed(campaign, attempt_id, "compile_result"), "compile_evidence"
        )
        correctness_receipt = event_receipt(
            require_passed(campaign, attempt_id, "correctness_result"),
            "correctness_evidence",
        )
        safety = attempt_event(campaign, attempt_id, "safety_result")
        performance = require_passed(campaign, attempt_id, "performance_command_result")
        assert safety is not None
        campaign.record.mark_verified(
            attempt_id,
            compile_receipt=compile_receipt,
            correctness_receipt=correctness_receipt,
            safety_receipt=event_receipt(safety, "safety_result"),
            performance_receipt=event_receipt(performance, "performance_evidence"),
            measurement_receipt=grade_receipt,
        )
        campaign.record.complete_verified(attempt_id)
        if not evaluation.improved:
            campaign.record.record_decision(
                attempt_id,
                verdict="revert",
                reason=evaluation.grade.promotion_reason_code,
                srobust=evaluation.grade.srobust,
                reward=evaluation.grade.reward,
            )

    @staticmethod
    def _no_measurement(
        campaign: FormalKernelCampaign, attempt_id: str, reason: str
    ) -> FormalEvaluatorResult:
        if attempt_event(campaign, attempt_id, "measurement_result", required=False) is None:
            record_measurement_error(
                campaign.record, attempt_id, reason_code=reason
            )
            campaign.record.defer_attempt(attempt_id, reason)
        return FormalEvaluatorResult(
            {
                "schema": "apex.kernel-measure-capability/v1",
                "status": "no_measurement",
                "attempt_id": attempt_id,
                "reason_code": reason,
                "reward": None,
            }
        )


def _require_contract(campaign, digest: str):
    contract = campaign.authorized_contract
    if contract is None or contract.digest != digest:
        raise ContractError(
            "Verified evaluation contract is missing or differs",
            "evaluation_authority_mismatch",
        )
    return contract


def _require_candidate_digest(observed: str, expected: str) -> None:
    if observed != expected:
        raise IntegrityError(
            "Candidate digest differs from the frozen attempt",
            "candidate_digest_mismatch",
            {"expected": expected, "observed": observed},
        )


def _validate_candidate_event(event, expected: str) -> None:
    if event.payload.get("candidate_source_sha256") != expected:
        raise IntegrityError(
            "Candidate event binds different source bytes",
            "candidate_digest_mismatch",
        )


def _gate_projection(campaign, attempt_id, contract_digest, candidate_digest, event):
    phase = {
        "compile_result": "compile",
        "correctness_result": "correctness",
    }[event.event_type]
    receipt = event_receipt(event, f"{phase}_evidence")
    return FormalEvaluatorResult(
        {
            "schema": f"apex.kernel-{phase}-capability/v1",
            "status": "passed" if event.payload.get("passed") is True else "failed",
            "attempt_id": attempt_id,
            "contract_digest": contract_digest,
            "candidate_digest": candidate_digest,
            "evidence_receipt": receipt.to_dict(),
            "reward": None,
        },
        (receipt,),
    )


def _measurement_projection(attempt_id, contract_digest, candidate_digest, raw):
    return FormalEvaluatorResult(
        {
            "schema": "apex.kernel-measure-capability/v1",
            "status": "captured_ungraded",
            "attempt_id": attempt_id,
            "contract_digest": contract_digest,
            "candidate_digest": candidate_digest,
            "raw_measurement_receipt": raw.to_dict(),
            "reward": None,
        },
        (raw,),
    )


def _unverified_grade(reason: str) -> FormalEvaluatorResult:
    return FormalEvaluatorResult(
        {
            "schema": "apex.kernel-grade-capability/v1",
            "status": "unverified" if reason != "no_measurement" else "no_measurement",
            "attempt_id": None,
            "contract_digest": None,
            "candidate_digest": None,
            "measurement_status": "not_configured",
            "reward": None,
            "srobust": None,
            "promotion_eligible": False,
            "reason_code": reason,
            "grade_receipt": None,
        }
    )


def _unverified_compile(reason: str) -> FormalEvaluatorResult:
    return FormalEvaluatorResult(
        {
            "schema": "apex.kernel-compile-capability/v1",
            "status": "unverified",
            "attempt_id": None,
            "contract_digest": None,
            "candidate_digest": None,
            "reason_code": reason,
            "reward": None,
        }
    )


def _harness_receipt(campaign: FormalKernelCampaign) -> ArtifactReceipt:
    document = {
        "schema_version": 1,
        "commands": {
            name: command.to_dict()
            for name, command in sorted(campaign.task.commands.items())
        },
        "baseline_file_hashes": dict(
            sorted(campaign.resolved.baseline_file_hashes.items())
        ),
        "measurement": campaign.task.measurement.to_dict(),
        "harness_file_hashes": dict(
            sorted(campaign.resolved.harness_file_hashes.items())
        ),
        "harness_sha256": campaign.resolved.harness_sha256,
    }
    return campaign.record.artifacts.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )


__all__ = ["KernelFormalEvaluator"]
