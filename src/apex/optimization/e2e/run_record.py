"""Canonical event/CAS facade for a workload optimization run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.benchmark import NormalizedBenchmarkResult
from apex.context import CompiledContext
from apex.core import ContractError, canonical_json_bytes
from apex.execution import agent_transcript_document
from apex.evaluation import E2ERewardGrade, E2ERewardPolicy
from apex.knowledge import ExperienceIdentity
from apex.orchestration import RunController
from apex.ports import DiagnosticsResult, TraceComparisonResult
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal, SnapshotStore

from ..agent_recording import record_agent_observations

from .benchmark_artifacts import (
    BenchmarkEvidenceReceipts,
    persist_benchmark_evidence,
)
from .candidate import E2ECandidate
from .candidate_record import candidate_manifest, store_candidate_sources
from .deployment_artifacts import persist_deployment_configs
from .services import (
    CandidateDeployment,
    FinalDeliveryResult,
    MicroQualification,
    SafetyQualification,
)
from .trace_comparison_record import persist_trace_comparison


@dataclass(slots=True)
class E2ERunRecord:
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
    ) -> "E2ERunRecord":
        root.mkdir(parents=True, exist_ok=False)
        journal = EventJournal(root / "events" / "run.db")
        controller = RunController.create(
            run_id,
            journal,
            SnapshotStore(root / "state.snapshot.json"),
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

    def iter_events(self):
        """Return verified canonical history for fresh context projections."""

        return self.journal.iter_events(self.run_id, verify=True)

    def put_json(self, value: Mapping[str, Any]) -> ArtifactReceipt:
        return self.artifacts.put_bytes(canonical_json_bytes(value), media_type="application/json")

    def begin_action(self, action_id: str, action_type: str) -> None:
        self.controller.queue_action(action_id, action_type)
        self.controller.start_action(action_id)

    def record_benchmark(
        self,
        action_id: str,
        result: NormalizedBenchmarkResult,
        config_path: Path,
        *,
        attempt_id: str | None = None,
        candidate_id: str | None = None,
        opportunity_id: str | None = None,
    ) -> BenchmarkEvidenceReceipts:
        evidence = persist_benchmark_evidence(
            self.artifacts, result, config_path
        )
        self.controller.mark_artifacts_ready(
            action_id, [item.digest for item in evidence.receipts]
        )
        if result.succeeded:
            self.controller.verify_action(action_id, evidence.normalized.digest)
            self.controller.complete_action(action_id)
        else:
            self.controller.fail_action(action_id, ";".join(result.errors) or "benchmark_failed")
        lineage: dict[str, object] = {}
        if attempt_id is not None:
            lineage = self._attempt_payload(attempt_id, candidate_id=candidate_id)
            if opportunity_id is not None:
                lineage["opportunity_id"] = opportunity_id
        self.controller.record_domain_event(
            "measurement_result",
            {
                **lineage,
                "action_id": action_id,
                "pass_type": result.pass_type.value,
                "succeeded": result.succeeded,
                "metrics": {
                    key: value
                    for key, value in result.metric_mapping().items()
                    if value is not None
                },
                "evidence_class": (
                    "diagnostic" if result.profiling_enabled else "measured"
                ),
                "run_kind": result.run_kind,
                "reward_eligible": result.reward_eligible,
                "model_revision_verified": result.model_revision.passed,
                "inferencex_runtime_verified": result.inferencex_runtime.passed,
                "lm_eval_runtime_verified": result.lm_eval_runtime.passed,
                "serving_runtime_verified": result.serving_runtime.passed,
                "resolved_image_id": result.serving_runtime.resolved_image_id,
                "config_sha256": evidence.config.digest,
                "normalized_benchmark_receipt": evidence.normalized.digest,
                "quality_receipt": evidence.quality.digest,
                "artifacts": [dict(item) for item in evidence.bindings],
            },
            idempotency_key=f"benchmark.{action_id}.measurement",
        )
        return evidence

    def record_diagnostics(
        self,
        result: DiagnosticsResult,
        *,
        action_id: str,
        terminal: bool = False,
    ) -> tuple[ArtifactReceipt, ...]:
        stored = tuple(
            (
                path,
                self.artifacts.put_file(path, media_type=_media_type(path)),
            )
            for path in result.artifacts
            if path.is_file()
        )
        receipts = tuple(receipt for _, receipt in stored)
        self.controller.record_domain_event(
            "tool_result",
            {
                "tool": (
                    "tracelens_terminal_diagnostics"
                    if terminal
                    else "tracelens_diagnostics"
                ),
                "succeeded": result.succeeded,
                "terminal": terminal,
                "evidence_class": "diagnostic",
                "reward_eligible": False,
                "summary": dict(result.summary),
                "artifacts": [
                    _artifact_binding(
                        result.artifact_roles.get(
                            str(path.resolve()), "diagnostic_artifact"
                        ),
                        receipt,
                    )
                    for path, receipt in stored
                ],
            },
            idempotency_key=f"diagnostics.{action_id}.result",
        )
        return receipts

    def record_trace_comparison(
        self,
        result: TraceComparisonResult,
        *,
        action_id: str,
    ) -> ArtifactReceipt:
        comparison = persist_trace_comparison(self.artifacts, result)
        self.controller.record_domain_event(
            "tool_result",
            comparison.event_payload,
            idempotency_key=f"diagnostics.{action_id}.comparison",
        )
        return comparison.receipt

    def record_context(
        self,
        attempt_id: str,
        *,
        compiled: CompiledContext,
        packet: ArtifactReceipt,
        source: ArtifactReceipt,
        harness: ArtifactReceipt,
        knowledge: ArtifactReceipt,
        prompt: str,
        experience_identity: ExperienceIdentity,
        experience_mechanism: str,
    ) -> ArtifactReceipt:
        prompt_receipt = self.artifacts.put_bytes(prompt.encode(), media_type="text/plain")
        context = compiled.packet
        common = self._attempt_payload(attempt_id)
        self.controller.record_domain_event(
            "context_packet_created",
            {
                **common,
                "context_packet_id": context.context_packet_id,
                "workload_id": context.workload_id,
                "opportunity_id": context.target.opportunity_id,
                "state_generation": context.state_generation,
                "anchor_generation": context.current_anchor.generation,
                "cycle": context.cycle,
                "compiler_receipt": compiled.receipt,
                "experience_identity": experience_identity.to_dict(),
                "experience_mechanism": experience_mechanism,
                "artifacts": [
                    _artifact_binding("context_packet", packet),
                    _artifact_binding("baseline_source", source),
                    _artifact_binding("protected_harness", harness),
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.context",
        )
        selection = compiled.knowledge_selection
        self.controller.record_domain_event(
            "knowledge_read",
            {
                **common,
                "read_id": f"read-{attempt_id}",
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

    def record_candidate(self, candidate: E2ECandidate) -> ArtifactReceipt:
        self._record_agent_result(candidate)
        sources = store_candidate_sources(self.artifacts, candidate)
        manifest = self.put_json(candidate_manifest(candidate, sources))
        self.controller.record_domain_event(
            "candidate_frozen",
            {
                **self._attempt_payload(candidate.attempt_id),
                "candidate_id": candidate.candidate_id,
                "succeeded": candidate.succeeded,
                "reason_code": candidate.reason_code,
                "changed_files": list(candidate.changed_files),
                "artifacts": [
                    _artifact_binding("candidate_manifest", manifest),
                    *(_artifact_binding("candidate_source", receipt) for receipt in sources),
                ],
            },
            idempotency_key=f"attempt.{candidate.attempt_id}.candidate",
        )
        return manifest

    def _record_agent_result(self, candidate: E2ECandidate) -> None:
        result = candidate.agent_result
        stdout = self.artifacts.put_bytes(
            result.stdout.encode(), media_type="application/x-ndjson"
        )
        stderr = self.artifacts.put_bytes(
            result.stderr.encode(), media_type="text/plain"
        )
        transcript = self.put_json(agent_transcript_document(result))
        common = self._attempt_payload(
            candidate.attempt_id, candidate_id=candidate.candidate_id
        )
        record_agent_observations(
            self.controller,
            result=result,
            common_payload=common,
            transcript=transcript,
            idempotency_prefix=f"attempt.{candidate.attempt_id}",
        )
        agent_type = (
            "agent_completed" if result.candidate_capture_allowed else "agent_failed"
        )
        self.controller.record_domain_event(
            agent_type,
            {
                **common,
                "backend": result.backend.value,
                "model": result.model,
                "effort": result.effort,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                **_agent_execution_payload(result),
                "duration_seconds": result.duration_seconds,
                "transcript_event_count": len(result.events),
                "semantic_event_count": len(result.semantic_events),
                "artifacts": [
                    _artifact_binding("agent_transcript", transcript),
                    _artifact_binding("agent_stdout", stdout),
                    _artifact_binding("agent_stderr", stderr),
                ],
            },
            idempotency_key=f"attempt.{candidate.attempt_id}.agent",
        )

    def record_micro(self, attempt_id: str, result: MicroQualification) -> ArtifactReceipt:
        receipt = self.put_json(result.to_dict())
        common = self._attempt_payload(attempt_id, candidate_id=result.candidate_id)
        if result.qualification_mode == "e2e_quality_deferred":
            self.controller.record_domain_event(
                "tool_result",
                {
                    **common,
                    "tool": "e2e_micro_qualification",
                    "qualification_mode": result.qualification_mode,
                    "kernel_reward_available": False,
                    "claims": {
                        "compiled": "unmeasured",
                        "correct": "unmeasured",
                        "p50": "unmeasured",
                        "p99": "unmeasured",
                    },
                    "evidence_class": "diagnostic",
                    "artifacts": [_artifact_binding("micro_qualification", receipt)],
                },
                idempotency_key=f"attempt.{attempt_id}.micro.deferred",
            )
            return receipt
        for event_type, passed in (
            ("compile_result", result.compiled),
            ("correctness_result", result.correct and result.integrity_passed),
            ("measurement_result", result.performance_valid),
        ):
            payload: dict[str, Any] = {
                **common,
                "passed": passed,
                "reason_code": result.reason_code,
                "evidence_class": "measured",
                "artifacts": [_artifact_binding("micro_qualification", receipt)],
            }
            if event_type == "measurement_result":
                grade = result.grade
                payload.update(
                    {
                        "sample_count": result.sample_count,
                        "s50": result.s50,
                        "s99": result.s99,
                        "srobust": result.srobust,
                        "worst_case_srobust": grade.worst_case_srobust if grade else None,
                        "max_cv": grade.max_cv if grade else None,
                        "srobust_ci_lower": grade.srobust_ci_lower if grade else None,
                        "srobust_ci_upper": grade.srobust_ci_upper if grade else None,
                        "confidence_level": grade.confidence_level if grade else None,
                        "bootstrap_seed": grade.bootstrap_seed if grade else None,
                        "bootstrap_repetitions": grade.bootstrap_repetitions if grade else None,
                        "min_bootstrap_units": grade.min_bootstrap_units if grade else None,
                        "keep_srobust_threshold": grade.keep_srobust_threshold if grade else None,
                        "confidence_srobust_floor": grade.confidence_srobust_floor if grade else None,
                        "worst_case_srobust_floor": grade.worst_case_srobust_floor if grade else None,
                        "max_cv_threshold": grade.max_cv_threshold if grade else None,
                        "grade_policy_id": grade.policy_id if grade else None,
                        "reward": grade.reward if grade else None,
                        "threshold_pass": grade.threshold_pass if grade else False,
                        "confidence_pass": grade.confidence_pass if grade else False,
                        "noise_pass": grade.noise_pass if grade else False,
                        "worst_case_pass": grade.worst_case_pass if grade else False,
                        "promotion_eligible": grade.promotion_eligible if grade else False,
                    }
                )
            self.controller.record_domain_event(
                event_type,
                payload,
                idempotency_key=f"attempt.{attempt_id}.{event_type}",
            )
        return receipt

    def record_safety(self, attempt_id: str, result: SafetyQualification) -> ArtifactReceipt:
        receipt = self.put_json(result.to_dict())
        self.controller.record_domain_event(
            "safety_result",
            {
                **self._attempt_payload(attempt_id, candidate_id=result.candidate_id),
                "allowed_to_measure": result.allowed_to_measure,
                "promotion_eligible": result.promotion_eligible,
                "safety_certified": result.safety_certified,
                "finding": result.finding,
                "reason_codes": list(result.reason_codes),
                "evidence_class": "diagnostic",
                "artifacts": [_artifact_binding("safety_qualification", receipt)],
            },
            idempotency_key=f"attempt.{attempt_id}.safety",
        )
        return receipt

    def record_delivery(self, attempt_id: str, result: CandidateDeployment) -> ArtifactReceipt:
        receipt = self.put_json(result.to_dict())
        config_bindings = persist_deployment_configs(
            self.artifacts, self.root, result
        )
        self.controller.record_domain_event(
            "delivery_result",
            {
                **self._attempt_payload(attempt_id, candidate_id=result.candidate_id),
                "deployed": result.deployed,
                "engagement_verified": result.engagement_verified,
                "validation_level": result.validation_level.value,
                "reason_code": result.reason_code,
                "infrastructure_failure": result.infrastructure_failure,
                "config_sha256": (
                    result.config_sha256.to_dict()
                    if result.config_sha256 is not None
                    else None
                ),
                "artifacts": [
                    _artifact_binding("primary_delivery", receipt),
                    *config_bindings,
                ],
            },
            idempotency_key=f"attempt.{attempt_id}.delivery",
        )
        return receipt

    def prepare_outcome(
        self,
        attempt_id: str,
        *,
        candidate_id: str | None,
        verdict: str,
        reason: str,
        evidence: Mapping[str, Any],
        grade: E2ERewardGrade,
        evidence_artifacts: tuple[tuple[str, ArtifactReceipt], ...],
    ) -> tuple[ArtifactReceipt, tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
        if grade.verdict != verdict or grade.reason_code != reason:
            raise ContractError(
                "E2E decision and reward grade differ",
                "e2e_reward_verdict_mismatch",
            )
        if grade.candidate_present != (candidate_id is not None):
            raise ContractError(
                "E2E reward candidate presence differs",
                "e2e_reward_candidate_mismatch",
            )
        policy = E2ERewardPolicy()
        if grade.policy_id != policy.policy_id or grade.policy_digest != policy.digest:
            raise ContractError(
                "E2E reward policy is not the active policy",
                "e2e_reward_policy_mismatch",
            )
        decision = self.put_json(dict(evidence))
        grade_receipt = self.put_json(grade.to_dict())
        policy_receipt = self.put_json(policy.to_dict())
        bindings = (
            _artifact_binding("decision_evidence", decision),
            _artifact_binding("e2e_grade", grade_receipt),
            _artifact_binding("reward_policy", policy_receipt),
            *(
                _artifact_binding(role, receipt)
                for role, receipt in evidence_artifacts
            ),
        )
        reward_payload = {
            **self._attempt_payload(attempt_id, candidate_id=candidate_id),
            "verdict": verdict,
            "reason_code": reason,
            "policy_id": grade.policy_id,
            "policy_digest": grade.policy_digest,
            "scalar_reward": grade.scalar_reward,
            "reward_vector": grade.to_dict(),
            "evidence_class": "derived",
            "artifacts": [dict(item) for item in bindings],
        }
        return decision, (bindings[0],), reward_payload

    def record_final_delivery(self, result: FinalDeliveryResult) -> ArtifactReceipt:
        receipt = self.put_json(result.to_dict())
        self.controller.record_domain_event(
            "delivery_result",
            {
                "phase": "second_clean_replay",
                "verified": result.verified,
                "status": result.status.value,
                "validation_level": result.validation_level.value,
                "clean_replay_verified": result.clean_replay_verified,
                "reason_code": result.reason_code,
                "artifacts": [_artifact_binding("final_delivery", receipt)],
            },
            idempotency_key="delivery.final",
        )
        return receipt

    def _attempt_payload(
        self, attempt_id: str, *, candidate_id: str | None = None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "attempt_id": attempt_id,
            "anchor_generation": self.controller.state.anchor_generation,
            "split": self.dataset_split,
            "visibility": self.data_visibility,
        }
        if candidate_id is not None:
            payload["candidate_id"] = candidate_id
        return payload


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".gz": "application/gzip",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(path.suffix.lower(), "application/octet-stream")


def _artifact_binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def _agent_execution_payload(result: AgentResult) -> dict[str, object]:
    return {
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
        "invocation": result.invocation.to_dict() if result.invocation else None,
    }


__all__ = ["E2ERunRecord"]
