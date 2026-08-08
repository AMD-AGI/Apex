"""Canonical event/CAS facade for a workload optimization run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.benchmark import NormalizedBenchmarkResult
from apex.context import CompiledContext
from apex.core import canonical_json_bytes
from apex.execution import agent_transcript_document
from apex.orchestration import RunController
from apex.ports import AgentSemanticEvent
from apex.ports import DiagnosticsResult
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal, SnapshotStore

from .candidate import E2ECandidate
from .candidate_record import candidate_manifest, store_candidate_sources
from .services import (
    CandidateDeployment,
    FinalDeliveryResult,
    MicroQualification,
    SafetyQualification,
)


@dataclass(slots=True)
class E2ERunRecord:
    run_id: str
    root: Path
    artifacts: ArtifactStore
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
        controller = RunController.create(
            run_id,
            EventJournal(root / "events" / "run.db"),
            SnapshotStore(root / "state.snapshot.json"),
            initial_anchor_id=initial_anchor_id,
        )
        return cls(
            run_id,
            root,
            ArtifactStore(root / "artifacts"),
            controller,
            dataset_split,
            data_visibility,
        )

    def put_json(self, value: Mapping[str, Any]) -> ArtifactReceipt:
        return self.artifacts.put_bytes(canonical_json_bytes(value), media_type="application/json")

    def begin_action(self, action_id: str, action_type: str) -> None:
        self.controller.queue_action(action_id, action_type)
        self.controller.start_action(action_id)

    def record_benchmark(
        self, action_id: str, result: NormalizedBenchmarkResult
    ) -> ArtifactReceipt:
        receipts = [
            self.artifacts.put_file(path, media_type=_media_type(path))
            for path in result.artifacts
            if path.is_file()
        ]
        normalized = self.put_json(_benchmark_dict(result))
        receipts.append(normalized)
        self.controller.mark_artifacts_ready(action_id, [item.digest for item in receipts])
        if result.succeeded:
            self.controller.verify_action(action_id, normalized.digest)
            self.controller.complete_action(action_id)
        else:
            self.controller.fail_action(action_id, ";".join(result.errors) or "benchmark_failed")
        self.controller.record_domain_event(
            "measurement_result",
            {
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
                "artifacts": [_artifact_binding("normalized_benchmark", normalized)],
            },
            idempotency_key=f"benchmark.{action_id}.measurement",
        )
        return normalized

    def record_diagnostics(self, result: DiagnosticsResult) -> tuple[ArtifactReceipt, ...]:
        receipts = tuple(
            self.artifacts.put_file(path, media_type=_media_type(path))
            for path in result.artifacts
            if path.is_file()
        )
        self.controller.record_domain_event(
            "tool_result",
            {
                "tool": "tracelens_diagnostics",
                "succeeded": result.succeeded,
                "summary": dict(result.summary),
                "artifacts": [
                    _artifact_binding("diagnostic_artifact", receipt) for receipt in receipts
                ],
            },
            idempotency_key=f"diagnostics.{self.controller.state.e2e.bottleneck_generation if self.controller.state.e2e else 0}.result",
        )
        return receipts

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
        binding = [_artifact_binding("agent_transcript", transcript)]
        for event in result.semantic_events:
            self.controller.record_domain_event(
                event.kind,
                {
                    **common,
                    "backend": result.backend.value,
                    "model": result.model,
                    "effort": result.effort,
                    **_semantic_agent_payload(event),
                    "artifacts": binding,
                },
                idempotency_key=(
                    f"attempt.{candidate.attempt_id}.agent_event.{event.index}"
                ),
            )
        if result.usage is not None:
            self.controller.record_domain_event(
                "usage_recorded",
                {
                    **common,
                    "backend": result.backend.value,
                    "model": result.model,
                    "effort": result.effort,
                    "evidence_class": "self_reported",
                    **result.usage.to_dict(),
                    "artifacts": binding,
                },
                idempotency_key=f"attempt.{candidate.attempt_id}.usage",
            )
        if result.cost is not None:
            self.controller.record_domain_event(
                "cost_recorded",
                {
                    **common,
                    "backend": result.backend.value,
                    "model": result.model,
                    "effort": result.effort,
                    "evidence_class": "self_reported",
                    **result.cost.to_dict(),
                    "artifacts": binding,
                },
                idempotency_key=f"attempt.{candidate.attempt_id}.cost",
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
        self.controller.record_domain_event(
            "delivery_result",
            {
                **self._attempt_payload(attempt_id, candidate_id=result.candidate_id),
                "deployed": result.deployed,
                "engagement_verified": result.engagement_verified,
                "validation_level": result.validation_level.value,
                "reason_code": result.reason_code,
                "infrastructure_failure": result.infrastructure_failure,
                "artifacts": [_artifact_binding("primary_delivery", receipt)],
            },
            idempotency_key=f"attempt.{attempt_id}.delivery",
        )
        return receipt

    def record_decision(
        self,
        attempt_id: str,
        *,
        candidate_id: str,
        verdict: str,
        reason: str,
        evidence: Mapping[str, Any],
    ) -> ArtifactReceipt:
        receipt = self.put_json(dict(evidence))
        self.controller.record_domain_event(
            "decision",
            {
                **self._attempt_payload(attempt_id, candidate_id=candidate_id),
                "verdict": verdict,
                "reason": reason,
                "artifacts": [_artifact_binding("decision_evidence", receipt)],
            },
            idempotency_key=f"attempt.{attempt_id}.decision.evidence",
        )
        return receipt

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
        return {
            "attempt_id": attempt_id,
            "candidate_id": candidate_id or attempt_id,
            "anchor_generation": self.controller.state.anchor_generation,
            "split": self.dataset_split,
            "visibility": self.data_visibility,
        }


def _benchmark_dict(result: NormalizedBenchmarkResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "run_id": result.run_id,
        "pass_type": result.pass_type.value,
        "succeeded": result.succeeded,
        "framework": result.framework,
        "model": result.model,
        "workspace_path": str(result.workspace_path),
        "report_path": str(result.report_path) if result.report_path else None,
        "throughput": asdict(result.throughput),
        "latency": asdict(result.latency),
        "quality": {
            **asdict(result.quality),
            "source_paths": [str(path) for path in result.quality.source_paths],
            "raw_artifact_paths": [
                str(path) for path in result.quality.raw_artifact_paths
            ],
        },
        "profiling_enabled": result.profiling_enabled,
        "run_kind": result.run_kind,
        "reward_eligible": result.reward_eligible,
        "model_revision": {
            **asdict(result.model_revision),
            "source_path": (
                str(result.model_revision.source_path)
                if result.model_revision.source_path
                else None
            ),
        },
        "inferencex_runtime": {
            **asdict(result.inferencex_runtime),
            "source_root": (
                str(result.inferencex_runtime.source_root)
                if result.inferencex_runtime.source_root
                else None
            ),
            "runtime_path": (
                str(result.inferencex_runtime.runtime_path)
                if result.inferencex_runtime.runtime_path
                else None
            ),
            "receipt_path": (
                str(result.inferencex_runtime.receipt_path)
                if result.inferencex_runtime.receipt_path
                else None
            ),
        },
        "lm_eval_runtime": {
            **asdict(result.lm_eval_runtime),
            "manifest_path": (
                str(result.lm_eval_runtime.manifest_path)
                if result.lm_eval_runtime.manifest_path
                else None
            ),
            "receipt_path": (
                str(result.lm_eval_runtime.receipt_path)
                if result.lm_eval_runtime.receipt_path
                else None
            ),
        },
        "artifacts": [str(path) for path in result.artifacts],
        "errors": list(result.errors),
        "command_exit_code": result.command_exit_code,
        "timed_out": result.timed_out,
    }


def _media_type(path: Path) -> str:
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
        ".gz": "application/gzip",
    }.get(path.suffix.lower(), "application/octet-stream")


def _artifact_binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def _semantic_agent_payload(event: AgentSemanticEvent) -> dict[str, object]:
    payload: dict[str, object] = {
        "semantic_index": event.index,
        "source_event_index": event.source_event_index,
        "source_kind": event.source_kind,
        "evidence_class": "self_reported",
    }
    if event.kind == "agent_message":
        payload.update(
            {
                "role": event.role or "assistant",
                "has_text": event.text is not None,
                "text_length": len(event.text) if event.text is not None else 0,
            }
        )
    else:
        payload.update(
            {
                "tool_name": event.tool_name,
                "call_id": event.tool_call_id,
                "succeeded": event.succeeded,
            }
        )
    return payload


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
