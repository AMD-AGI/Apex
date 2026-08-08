"""Canonical event/CAS recording for one standalone kernel optimization run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from apex.context import CompiledContext
from apex.core import IntegrityError, canonical_json_bytes
from apex.evaluation import KernelGrade, KernelMeasurementArtifact, MeasurementStatus
from apex.evaluation.safety import PhaseIsolationReceipt, SafetyGateResult, VerificationPlan
from apex.knowledge import ExperienceIdentity, ExperienceOutcome
from apex.orchestration import RunController, RunPhase
from apex.ports import AgentResult
from apex.runtime import GpuLeaseReceipt
from apex.storage import (
    ArtifactReceipt,
    ArtifactStore,
    EventJournal,
    EventRecord,
    SnapshotStore,
)

from .verification import CommandEvidence
from .grading_record import (
    kernel_reward_policy_source,
    measurement_payload,
    reward_vector,
)
from .transcript import transcript_document, transcript_metadata


@dataclass(slots=True)
class KernelRunRecord:
    run_id: str
    root: Path
    artifacts: ArtifactStore
    journal: EventJournal
    controller: RunController
    dataset_split: str
    data_visibility: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        root: Path,
        initial_anchor_id: str,
        dataset_split: str,
        data_visibility: str,
    ) -> "KernelRunRecord":
        root.mkdir(parents=True, exist_ok=False)
        journal = EventJournal(root / "events" / "run.db")
        snapshots = SnapshotStore(root / "state.snapshot.json")
        controller = RunController.create(
            run_id,
            journal,
            snapshots,
            initial_anchor_id=initial_anchor_id,
        )
        return cls(
            run_id,
            root,
            ArtifactStore(root / "artifacts"),
            journal,
            controller,
            dataset_split,
            data_visibility,
        )

    def iter_events(self) -> tuple[EventRecord, ...]:
        """Return the verified canonical history used by the next ContextPacket."""

        return self.journal.iter_events(self.run_id, verify=True)

    def record_gpu_lease(self, lease: GpuLeaseReceipt) -> ArtifactReceipt:
        """Bind the exact run-scoped GPU ownership receipt before any command."""

        receipt = self.artifacts.put_bytes(
            canonical_json_bytes(lease.to_dict()), media_type="application/json"
        )
        self.controller.record_domain_event(
            "dependency_verified",
            {
                "kind": "gpu_lease",
                "device_scope": lease.device_scope,
                "lease_digest": lease.digest,
                "artifacts": [_artifact_binding("gpu_lease", receipt)],
            },
            idempotency_key="gpu_lease.acquired",
        )
        return receipt

    def start_attempt(self, attempt_id: str) -> None:
        self.controller.queue_action(attempt_id, "kernel-candidate")
        self.controller.start_action(attempt_id)

    def record_agent(
        self,
        attempt_id: str,
        *,
        result: AgentResult,
    ) -> tuple[ArtifactReceipt, ...]:
        stdout = self.artifacts.put_bytes(
            result.stdout.encode(), media_type="application/x-ndjson"
        )
        stderr = self.artifacts.put_bytes(result.stderr.encode(), media_type="text/plain")
        transcript = self.artifacts.put_bytes(
            canonical_json_bytes(transcript_document(result)),
            media_type="application/json",
        )
        event_type = (
            "agent_completed" if result.candidate_capture_allowed else "agent_failed"
        )
        self.controller.record_domain_event(
            event_type,
            {
                **self._attempt_payload(attempt_id),
                **transcript_metadata(result),
                "backend": result.backend.value,
                "model": result.model,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "termination_kind": result.termination_kind.value,
                "termination_reason": result.termination_reason,
                "capture_status": result.capture_status.value,
                "candidate_capture_allowed": result.candidate_capture_allowed,
                "observer_stop_sent": result.observer_stop_sent,
                "process_containment_policy_id": (
                    result.invocation.process_containment_policy_id
                    if result.invocation
                    else None
                ),
                "process_containment": (
                    result.process_containment.to_dict()
                    if result.process_containment is not None
                    else None
                ),
                "discarded_stdout_lines": result.discarded_stdout_lines,
                "discarded_stdout_bytes": result.discarded_stdout_bytes,
                "discarded_stdout_sha256": result.discarded_stdout_sha256,
                "observed_turns": result.observed_turns,
                "duration_seconds": result.duration_seconds,
                "artifacts": [
                    _artifact_binding("agent_transcript", transcript),
                    _artifact_binding("agent_stdout", stdout),
                    _artifact_binding("agent_stderr", stderr),
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.agent",
        )
        return transcript, stdout, stderr

    def record_context(
        self,
        attempt_id: str,
        *,
        compiled: CompiledContext,
        packet: ArtifactReceipt,
        sources: Sequence[ArtifactReceipt],
        harness: ArtifactReceipt,
        knowledge: ArtifactReceipt,
        prompt: str,
    ) -> ArtifactReceipt:
        """Bind the exact observation, advisory read, and prompt before invocation."""

        prompt_receipt = self.artifacts.put_bytes(prompt.encode(), media_type="text/plain")
        common = self._attempt_payload(attempt_id)
        context = compiled.packet
        self.controller.record_domain_event(
            "context_packet_created",
            {
                **common,
                "context_packet_id": context.context_packet_id,
                "task_id": context.workload_id.removeprefix("task-"),
                "kernel_id": context.target.opportunity_id,
                "state_generation": context.state_generation,
                "anchor_generation": context.current_anchor.generation,
                "cycle": context.cycle,
                "compiler_receipt": compiled.receipt,
                "artifacts": [
                    _artifact_binding("context_packet", packet),
                    *(_artifact_binding("source", receipt) for receipt in sources),
                    _artifact_binding("harness", harness),
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.context",
        )
        selection = compiled.knowledge_selection
        self.controller.record_domain_event(
            "knowledge_read",
            {
                **common,
                "context_packet_id": context.context_packet_id,
                "selection_receipt": selection.digest,
                "selection_policy": selection.selection_policy,
                "card_ids": [card.card_id for card in selection.cards],
                "unavailable_reason": selection.unavailable_reason,
                "artifacts": [_artifact_binding("knowledge_selection", knowledge)],
            },
            idempotency_key=f"attempt.{attempt_id}.knowledge",
        )
        self.controller.record_domain_event(
            "prompt_sent",
            {
                **common,
                "context_packet_id": context.context_packet_id,
                "artifacts": [_artifact_binding("prompt", prompt_receipt)],
            },
            idempotency_key=f"attempt.{attempt_id}.prompt",
        )
        return prompt_receipt

    def record_candidate(
        self,
        attempt_id: str,
        *,
        candidate_files: Mapping[str, bytes],
        changed_files: Sequence[str],
    ) -> tuple[ArtifactReceipt, ...]:
        receipts = tuple(
            self.artifacts.put_bytes(content, media_type=_source_media_type(relative))
            for relative, content in sorted(candidate_files.items())
        )
        self.controller.record_domain_event(
            "candidate_frozen",
            {
                **self._attempt_payload(attempt_id),
                "changed_files": list(changed_files),
                "artifacts": [
                    _artifact_binding("candidate", receipt) for receipt in receipts
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.candidate",
        )
        self.controller.mark_artifacts_ready(
            attempt_id, [receipt.digest for receipt in receipts]
        )
        return receipts

    def record_command(
        self,
        attempt_id: str,
        evidence: CommandEvidence,
    ) -> ArtifactReceipt:
        receipt = self.artifacts.put_bytes(
            canonical_json_bytes(evidence.to_dict()), media_type="application/json"
        )
        common = self._attempt_payload(attempt_id)
        event_type = {
            "compile": "compile_result",
            "correctness": "correctness_result",
            "performance": "performance_command_result",
        }[evidence.phase]
        payload: dict[str, object] = {
            **common,
            "passed": evidence.passed,
            "exit_code": evidence.exit_code,
            "timed_out": evidence.timed_out,
            "duration_seconds": evidence.duration_seconds,
            "evidence_class": "measured",
            "artifacts": [_artifact_binding(f"{evidence.phase}_evidence", receipt)],
        }
        if evidence.phase == "performance":
            payload["status"] = "command_completed_without_robust_timing_grade"
            payload["runtime"] = "normal_uninstrumented"
        self.controller.record_domain_event(
            event_type,
            payload,
            idempotency_key=f"attempt.{attempt_id}.{evidence.phase}",
        )
        return receipt

    def record_measurement(
        self,
        attempt_id: str,
        *,
        artifact: KernelMeasurementArtifact,
        grade: KernelGrade,
    ) -> ArtifactReceipt:
        """Persist raw samples, recomputed grade, and the sole reward event."""

        raw_receipt = self.artifacts.put_file(
            artifact.path,
            media_type="application/json",
        )
        if raw_receipt.digest != artifact.sha256:
            raise IntegrityError(
                "Kernel timing report changed after validation",
                "measurement_report_changed",
            )
        grade_receipt = self.artifacts.put_bytes(
            canonical_json_bytes(grade.to_dict()),
            media_type="application/json",
        )
        policy_receipt = self.artifacts.put_bytes(
            canonical_json_bytes(kernel_reward_policy_source(artifact.policy)),
            media_type="application/json",
        )
        self.controller.record_domain_event(
            "measurement_result",
            {
                **self._attempt_payload(attempt_id),
                **measurement_payload(artifact, grade),
                "evidence_class": "measured",
                "artifacts": [
                    _artifact_binding("raw_measurement", raw_receipt),
                    _artifact_binding("kernel_grade", grade_receipt),
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.measurement",
        )
        if (
            grade.measurement_status is MeasurementStatus.VALID
            and grade.reward is not None
        ):
            self.controller.record_domain_event(
                "reward_committed",
                {
                    **self._attempt_payload(attempt_id),
                    "policy_id": grade.policy_id,
                    "scalar_reward": grade.reward,
                    "reward_vector": reward_vector(grade),
                    "evidence_class": "measured",
                    "artifacts": [
                        _artifact_binding("raw_measurement", raw_receipt),
                        _artifact_binding("reward_policy", policy_receipt),
                        _artifact_binding("kernel_grade", grade_receipt),
                    ],
                },
                idempotency_key=f"attempt.{attempt_id}.reward",
            )
        return grade_receipt

    def record_measurement_error(
        self,
        attempt_id: str,
        *,
        reason_code: str,
    ) -> None:
        self.controller.record_domain_event(
            "measurement_result",
            {
                **self._attempt_payload(attempt_id),
                "measurement_status": "error",
                "reason_code": reason_code,
                "reward": None,
                "evidence_class": "measured",
            },
            idempotency_key=f"attempt.{attempt_id}.measurement",
        )

    def record_safety(
        self,
        attempt_id: str,
        *,
        plan: VerificationPlan,
        isolation: PhaseIsolationReceipt,
        result: SafetyGateResult,
    ) -> ArtifactReceipt:
        plan_receipt = self.artifacts.put_bytes(
            plan.canonical_bytes(), media_type="application/json"
        )
        isolation_receipt = self.artifacts.put_bytes(
            isolation.canonical_bytes(), media_type="application/json"
        )
        result_receipt = self.artifacts.put_bytes(
            result.canonical_bytes(), media_type="application/json"
        )
        self.controller.record_domain_event(
            "safety_result",
            {
                **self._attempt_payload(attempt_id),
                "plan_fingerprint": plan.fingerprint,
                "result_fingerprint": result.fingerprint,
                "allowed_to_measure": result.decision.allowed_to_measure,
                "promotion_eligible": result.decision.promotion_eligible,
                "safety_certified": result.safety_certified,
                "reason_codes": list(result.decision.reason_codes),
                "tools": [
                    {
                        "tool": evaluation.tool,
                        "capability": evaluation.capability.value,
                        "execution": evaluation.execution.value,
                        "finding": evaluation.finding.value,
                    }
                    for evaluation in result.evaluations
                ],
                "evidence_class": "measured",
                "artifacts": [
                    _artifact_binding("safety_plan", plan_receipt),
                    _artifact_binding("phase_isolation", isolation_receipt),
                    _artifact_binding("safety_result", result_receipt),
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.safety",
        )
        return result_receipt

    def record_experience(
        self,
        attempt_id: str,
        *,
        identity: ExperienceIdentity,
        outcome: ExperienceOutcome,
        strategy_fingerprint: str,
        mechanism: str,
        micro_verdict: str,
        evidence: Sequence[ArtifactReceipt],
        failure_reason: str | None,
        retry_condition: str | None,
    ) -> None:
        """Append the typed measured outcome consumed by the next fresh context."""

        unique = tuple({item.digest: item for item in evidence}.values())
        if not unique:
            raise IntegrityError("Experience has no evidence", "missing_experience_evidence")
        self.controller.record_domain_event(
            "experience.measured",
            {
                **self._attempt_payload(attempt_id),
                "evidence_class": "measured",
                "dry_run": False,
                "identity": identity.to_dict(),
                "outcome": outcome.value,
                "strategy_fingerprint": strategy_fingerprint,
                "mechanism": mechanism,
                "micro_verdict": micro_verdict,
                "e2e_verdict": None,
                "evidence_receipts": [item.digest for item in unique],
                "failure_reason": failure_reason,
                "retry_condition": retry_condition,
                "artifacts": [
                    _artifact_binding("experience_evidence", item) for item in unique
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.experience",
        )

    def mark_verified(
        self,
        attempt_id: str,
        *,
        compile_receipt: ArtifactReceipt,
        correctness_receipt: ArtifactReceipt,
        safety_receipt: ArtifactReceipt,
        performance_receipt: ArtifactReceipt,
        measurement_receipt: ArtifactReceipt | None = None,
    ) -> ArtifactReceipt:
        summary = {
            "schema_version": "apex.kernel-verification-summary/v1",
            "compile": compile_receipt.digest,
            "correctness": correctness_receipt.digest,
            "safety": safety_receipt.digest,
            "normal_performance": performance_receipt.digest,
            "robust_measurement": (
                measurement_receipt.digest if measurement_receipt is not None else None
            ),
        }
        receipt = self.artifacts.put_bytes(
            canonical_json_bytes(summary), media_type="application/json"
        )
        self.controller.verify_action(attempt_id, receipt.digest)
        return receipt

    def record_decision(
        self,
        attempt_id: str,
        *,
        verdict: str,
        reason: str,
        bundle_digest: str | None = None,
        safety_result: SafetyGateResult | None = None,
        srobust: float | None = None,
        reward: float | None = None,
    ) -> None:
        if verdict not in {"keep", "revert", "reject", "needs_more_measurement"}:
            raise ValueError(f"unsupported kernel decision: {verdict}")
        payload: dict[str, object] = {
            **self._attempt_payload(attempt_id),
            "verdict": verdict,
            "reason": reason,
            "bundle_digest": bundle_digest,
            "srobust": srobust,
            "reward": reward,
        }
        if safety_result is not None:
            payload.update(
                {
                    "safety_certified": safety_result.safety_certified,
                    "safety_result_fingerprint": safety_result.fingerprint,
                }
            )
        self.controller.record_domain_event(
            "decision",
            payload,
            idempotency_key=f"attempt.{attempt_id}.decision",
        )

    def complete_verified(self, attempt_id: str) -> None:
        """Close a verified candidate action without advancing the workload anchor."""

        self.controller.complete_action(attempt_id)

    def abort_no_gain(self, attempt_id: str, reason: str) -> None:
        """Use the action abort as the attempt's sole REVERT decision."""

        self.controller.abort_pending(reason)

    def reject_attempt(self, attempt_id: str, reason: str) -> None:
        self.controller.fail_action(attempt_id, reason)
        self.record_decision(attempt_id, verdict="reject", reason=reason)

    def defer_attempt(self, attempt_id: str, reason: str) -> None:
        self.controller.fail_action(attempt_id, reason)
        self.record_decision(
            attempt_id,
            verdict="needs_more_measurement",
            reason=reason,
        )

    def finish(self, status: RunPhase, reason: str) -> None:
        self.controller.finish(status, reason=reason)

    def fail_active(self, reason: str) -> None:
        action = self.controller.state.pending_action
        if action is not None:
            self.controller.fail_action(action.action_id, reason)
        if self.controller.state.phase is RunPhase.RUNNING:
            self.controller.finish(RunPhase.FAILED, reason=reason)

    def _attempt_payload(self, attempt_id: str) -> dict[str, object]:
        return {
            "attempt_id": attempt_id,
            "candidate_id": attempt_id,
            "anchor_generation": self.controller.state.anchor_generation,
            "split": self.dataset_split,
            "visibility": self.data_visibility,
        }


def candidate_file_bytes(root: Path, relative_paths: Sequence[str]) -> dict[str, bytes]:
    return {
        relative: root.joinpath(*relative.split("/")).read_bytes()
        for relative in relative_paths
    }


def _artifact_binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def _source_media_type(relative: str) -> str:
    suffix = Path(relative).suffix.lower()
    return "text/x-c++" if suffix in {".hip", ".cpp", ".cc", ".cxx"} else "text/x-python"


__all__ = ["KernelRunRecord", "candidate_file_bytes"]
