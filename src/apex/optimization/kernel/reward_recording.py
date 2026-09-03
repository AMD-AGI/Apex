"""Canonical attempt-gate and task-terminal kernel reward recording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from apex.core import IntegrityError, canonical_json_bytes, sha256_json
from apex.evaluation import (
    GateVerdict,
    KERNEL_REWARD_POLICY_ID,
    KernelTerminalGrade,
    kernel_reward,
    kernel_terminal_policy_source,
)
from apex.storage import ArtifactReceipt

from .attempts import KernelAttemptOutcome
from .run_record import KernelRunRecord


@dataclass(frozen=True, slots=True)
class KernelTerminalEvidence:
    grade: KernelTerminalGrade
    result: ArtifactReceipt
    source: ArtifactReceipt | None
    policy: ArtifactReceipt
    vector: ArtifactReceipt
    raw_measurements: tuple[ArtifactReceipt, ...]


def record_attempt_gate_reward(
    record: KernelRunRecord,
    attempt_id: str,
    *,
    stage: str,
    command_receipt: ArtifactReceipt,
) -> None:
    if stage not in {"compile", "correctness"}:
        raise ValueError("gate reward stage must be compile or correctness")
    gates = GateVerdict(
        compiled=stage == "correctness",
        correct=False,
        integrity_passed=stage == "correctness",
        tampering_passed=stage == "correctness",
    )
    scalar = kernel_reward(gates, None)
    vector = _attempt_gate_vector(stage, gates, scalar)
    policy_document = kernel_terminal_policy_source()
    policy = record.artifacts.put_bytes(
        canonical_json_bytes(policy_document),
        media_type="application/json",
    )
    record.controller.record_domain_event(
        "reward_committed",
        {
            **record.attempt_payload(attempt_id),
            "scope": "attempt",
            "policy_id": KERNEL_REWARD_POLICY_ID,
            "policy_digest": sha256_json(policy_document),
            "scalar_reward": scalar,
            "reward_vector": vector,
            "evidence_class": "measured",
            "artifacts": [
                _binding(f"{stage}_evidence", command_receipt),
                _binding("reward_policy", policy),
            ],
        },
        idempotency_key=f"attempt.{attempt_id}.reward",
    )


def record_kernel_terminal_reward(
    record: KernelRunRecord,
    *,
    task_id: str,
    contract_digest: str,
    grade: KernelTerminalGrade,
    outcomes: tuple[KernelAttemptOutcome, ...],
) -> KernelTerminalEvidence:
    policy = record.artifacts.put_bytes(
        canonical_json_bytes(kernel_terminal_policy_source()),
        media_type="application/json",
    )
    vector = record.artifacts.put_bytes(
        canonical_json_bytes(grade.to_dict()), media_type="application/json"
    )
    source_outcome = _source_outcome(grade, outcomes)
    source = _terminal_source(record, contract_digest, grade, source_outcome)
    supporting = _supporting_receipts(record, grade)
    raw = tuple(
        receipt for role, receipt in supporting if role == "raw_measurement"
    )
    if grade.trainability == "trainable":
        assert source is not None and grade.scalar_reward is not None
        record.controller.record_domain_event(
            "reward_committed",
            {
                "scope": "task_terminal",
                "task_id": task_id,
                "policy_id": grade.policy_id,
                "policy_digest": grade.policy_digest,
                "scalar_reward": grade.scalar_reward,
                "reward_vector": grade.to_dict(),
                "reward_source_receipt": source.digest,
                "raw_measurement_receipts": [item.digest for item in raw],
                "evidence_class": "derived",
                "artifacts": [
                    _binding("terminal_reward_source", source),
                    _binding("kernel_terminal_grade", vector),
                    _binding("reward_policy", policy),
                    *(_binding(role, receipt) for role, receipt in supporting),
                ],
            },
            idempotency_key="kernel.task_terminal.reward",
        )
    result_document = {
        "schema": "apex.kernel-terminal-result/v1",
        "task_kind": "single_kernel",
        "run_id": record.run_id,
        "task_id": task_id,
        "evaluation_contract_receipt_digest": contract_digest,
        "task_reward": grade.scalar_reward,
        "reward_vector": grade.to_dict() if grade.scalar_reward is not None else None,
        "reward_policy_id": grade.policy_id,
        "reward_policy_digest": grade.policy_digest,
        "reward_source_receipt": source.digest if source is not None else None,
        "raw_measurement_receipts": [item.digest for item in raw],
        "trainability": grade.trainability,
        "untrainable_reason": grade.untrainable_reason,
    }
    result = record.artifacts.put_bytes(
        canonical_json_bytes(result_document), media_type="application/json"
    )
    record.controller.record_domain_event(
        "delivery_result",
        {
            "kind": "kernel_terminal_result",
            "task_id": task_id,
            "trainability": grade.trainability,
            "artifacts": [_binding("kernel_terminal_result", result)],
        },
        idempotency_key="kernel.task_terminal.result",
    )
    return KernelTerminalEvidence(grade, result, source, policy, vector, raw)


def _source_outcome(
    grade: KernelTerminalGrade,
    outcomes: tuple[KernelAttemptOutcome, ...],
) -> KernelAttemptOutcome | None:
    if grade.source_attempt_id is None:
        return None
    matches = tuple(
        item for item in outcomes if item.attempt_id == grade.source_attempt_id
    )
    if len(matches) != 1:
        raise IntegrityError(
            "Kernel terminal reward source attempt is missing",
            "kernel_terminal_source_missing",
        )
    return matches[0]


def _terminal_source(
    record: KernelRunRecord,
    contract_digest: str,
    grade: KernelTerminalGrade,
    outcome: KernelAttemptOutcome | None,
) -> ArtifactReceipt | None:
    if outcome is None:
        return None
    document = {
        "schema": "apex.kernel-terminal-reward-source/v1",
        "run_id": record.run_id,
        "evaluation_contract_receipt_digest": contract_digest,
        "source_attempt_id": outcome.attempt_id,
        "implementation": (
            "frozen_reference" if grade.outcome == "measured_noop" else "candidate"
        ),
        "candidate_source_sha256": (
            None if grade.outcome == "measured_noop" else outcome.strategy_fingerprint
        ),
        "measurement_candidate_source_sha256": (
            outcome.strategy_fingerprint
            if grade.outcome in {"selected_candidate", "measured_noop"}
            else None
        ),
        "outcome": grade.outcome,
        "reason_code": grade.reason_code,
        "attempt_evidence_receipts": list(outcome.evidence_receipts),
    }
    return record.artifacts.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )


def _supporting_receipts(
    record: KernelRunRecord,
    grade: KernelTerminalGrade,
) -> tuple[tuple[str, ArtifactReceipt], ...]:
    attempt_id = grade.source_attempt_id
    if attempt_id is None:
        return ()
    roles = (
        ("raw_measurement", "measurement_result", "raw_measurement"),
        ("measurement_execution", "measurement_result", "measurement_execution"),
        ("harness", "measurement_result", "harness"),
        ("kernel_grade", "measurement_result", "kernel_grade"),
        ("attempt_reward_policy", "reward_committed", "reward_policy"),
        ("compile_evidence", "compile_result", "compile_evidence"),
        ("correctness_evidence", "correctness_result", "correctness_evidence"),
    )
    found: list[tuple[str, ArtifactReceipt]] = []
    for output_role, event_type, event_role in roles:
        receipt = _event_role(record, attempt_id, event_type, event_role)
        if receipt is not None:
            found.append((output_role, receipt))
    required = {
        "selected_candidate": {
            "raw_measurement",
            "measurement_execution",
            "harness",
            "kernel_grade",
            "attempt_reward_policy",
            "compile_evidence",
            "correctness_evidence",
        },
        "measured_noop": {
            "raw_measurement",
            "measurement_execution",
            "harness",
            "kernel_grade",
            "attempt_reward_policy",
            "compile_evidence",
            "correctness_evidence",
        },
        "compile_failure": {"compile_evidence"},
        "correctness_failure": {"compile_evidence", "correctness_evidence"},
    }.get(grade.outcome, set())
    if not required <= {role for role, _ in found}:
        raise IntegrityError(
            "Kernel terminal reward supporting evidence is incomplete",
            "kernel_terminal_evidence_missing",
        )
    return tuple(found)


def _event_role(
    record: KernelRunRecord,
    attempt_id: str,
    event_type: str,
    role: str,
) -> ArtifactReceipt | None:
    matches: list[ArtifactReceipt] = []
    for event in record.iter_events():
        if event.event_type != event_type or event.payload.get("attempt_id") != attempt_id:
            continue
        for binding in event.payload.get("artifacts", ()):
            if isinstance(binding, Mapping) and binding.get("role") == role:
                receipt = binding.get("receipt")
                if isinstance(receipt, Mapping):
                    matches.append(ArtifactReceipt.from_dict(receipt))
    if len(matches) > 1:
        raise IntegrityError(
            "Kernel terminal evidence role is ambiguous",
            "kernel_terminal_evidence_ambiguous",
        )
    return matches[0] if matches else None


def _attempt_gate_vector(
    stage: str,
    gates: GateVerdict,
    scalar: float | None,
) -> dict[str, object]:
    return {
        "kernel_reward_stage": stage,
        "compile": gates.compiled,
        "correctness": gates.correct,
        "integrity": gates.integrity_passed,
        "anti_tampering": gates.tampering_passed,
        "safety": {"finding": gates.safety_finding},
        "kernel_srobust": None,
        "kernel_robust_reward": scalar,
    }


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = [
    "KernelTerminalEvidence",
    "record_attempt_gate_reward",
    "record_kernel_terminal_reward",
]
