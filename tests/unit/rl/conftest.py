from __future__ import annotations

from pathlib import Path

import pytest

from apex.context import (
    AnchorView,
    ContextBudget,
    ContextContract,
    ContextPacket,
    Hypothesis,
    TargetEvidence,
    freeze_metrics,
)
from apex.core import canonical_json_bytes, sha256_json
from apex.evaluation import (
    E2ERewardPolicy,
    KernelMeasurementExecutionReceipt,
    MeasurementPolicy,
    grade_e2e_outcome,
)
from apex.orchestration.replay import replay_workload_state
from apex.storage import (
    ArtifactReceipt,
    ArtifactStore,
    EventInput,
    EventJournal,
    derive_event_id,
)


def artifact_binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def append_event(
    journal: EventJournal,
    run_id: str,
    event_type: str,
    payload: dict[str, object],
    key: str,
):
    head = journal.last_event(run_id)
    return journal.append(
        run_id=run_id,
        event_type=event_type,
        payload=payload,
        idempotency_key=key,
        parent_event_id=head.event_id if head else None,
    )


def append_event_transaction(
    journal: EventJournal,
    run_id: str,
    events: tuple[tuple[str, dict[str, object], str], ...],
):
    head = journal.last_event(run_id)
    parent = head.event_id if head else None
    inputs: list[EventInput] = []
    for event_type, payload, key in events:
        inputs.append(EventInput(event_type, payload, key, parent))
        parent = derive_event_id(run_id, key)
    return journal.append_transaction(run_id=run_id, events=tuple(inputs))


def make_packet(
    run_id: str,
    *,
    cycle: int = 0,
    state_generation: int = 1,
    anchor_generation: int = 0,
) -> ContextPacket:
    digest = "a" * 64
    return ContextPacket(
        run_id=run_id,
        workload_id="workload-1",
        phase="executing",
        cycle=cycle,
        state_generation=state_generation,
        role_kind="kernel_optimizer",
        role_objective="Improve the frozen kernel without changing semantics.",
        primary_metric="kernel_srobust",
        hard_constraints=("correctness must pass", "source-only candidate"),
        target=TargetEvidence("opportunity-1", "kernels/norm.py:norm", "decode-m64", (digest,)),
        hypothesis=Hypothesis("hypothesis-1", "memory traffic dominates", "no bandwidth change"),
        current_anchor=AnchorView(
            f"anchor-{anchor_generation}",
            anchor_generation,
            freeze_metrics({"throughput": 10.0}),
        ),
        attempts=(),
        dead_ends=(),
        knowledge_cards=(),
        knowledge_selection_receipt=None,
        knowledge_unavailable_reason="disabled_for_test",
        budget=ContextBudget(4096, 2048, 8, 600, 300),
        contract=ContextContract(
            ("edit", "test"),
            ("kernels/norm.py",),
            "KEEP only evaluator-owned measured wins",
            "stop after the attempt budget",
        ),
        artifact_refs=(),
    )


@pytest.fixture
def canonical_run(tmp_path: Path):
    run_id = "run-rl-1"
    journal = EventJournal(tmp_path / "events" / "run.db")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    packet = make_packet(run_id)
    packet_receipt = artifacts.put_bytes(packet.canonical_bytes, media_type="application/json")
    second_packet = make_packet(
        run_id, cycle=1, state_generation=2, anchor_generation=0
    )
    second_packet_receipt = artifacts.put_bytes(
        second_packet.canonical_bytes, media_type="application/json"
    )
    source = artifacts.put_bytes(b"def baseline(x): return x\n", media_type="text/x-python")
    harness_sha256 = "b" * 64
    harness = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "harness_sha256": harness_sha256,
                "harness_file_hashes": {"harness.py": "a" * 64},
            }
        ),
        media_type="application/json",
    )
    prompt = artifacts.put_bytes(b"optimize the kernel", media_type="text/plain")
    tool = artifacts.put_bytes(b'{"bandwidth":123}', media_type="application/json")
    candidate = artifacts.put_bytes(
        b"def optimized(x):\n    return x\n", media_type="text/x-python"
    )
    method_sha256 = "d" * 64
    measurement = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "measurement_method_sha256": method_sha256,
                "reference_ms": [1.0],
                "optimized_ms": [0.9],
            }
        ),
        media_type="application/json",
    )
    measurement_policy = MeasurementPolicy().to_dict()
    policy = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema": "apex.kernel-reward-policy/v1",
                "measurement_policy": measurement_policy,
            }
        ),
        media_type="application/json",
    )
    execution_value = KernelMeasurementExecutionReceipt(
        run_id=run_id,
        attempt_id="attempt-1",
        writer_id="fixture-evaluator-v1",
        candidate_source_sha256="c" * 64,
        harness_sha256=harness_sha256,
        measurement_method_sha256=method_sha256,
        measurement_policy_sha256=sha256_json(measurement_policy),
        report_sha256=measurement.digest,
        report_size=measurement.size,
        phase_started_monotonic_ns=1,
        adapter_returned_monotonic_ns=2,
        output_observed_monotonic_ns=3,
        phase_completed_monotonic_ns=4,
    )
    execution = artifacts.put_bytes(
        execution_value.canonical_bytes, media_type="application/json"
    )
    replication = {
        "dependency_receipts": [
            {"name": "Magpie", "commit": "1" * 40, "digest": "b" * 64}
        ],
        "source_commits": [{"name": "vllm", "commit": "2" * 40}],
        "parent_image_digest": "sha256:" + "3" * 64,
        "derived_image_digest": "sha256:" + "4" * 64,
        "commands": [
            {"name": "apply_bundle", "argv": ["apex", "bundle", "verify", "bundle"]},
            {"name": "build_image", "argv": ["docker", "build", "."]},
            {"name": "clean_replay", "argv": ["apex", "optimize", "e2e", "--replay"]},
        ],
    }
    append_event(
        journal,
        run_id,
        "run.started",
        {
            "initial_anchor_id": "anchor-0",
            "workload_id": "workload-1",
            "task_id": "task-1",
            "provenance": {
                "gpu": "gfx950",
                "backend": "codex",
                "model": "gpt-test",
            },
            "replication": replication,
        },
        "run-started",
    )
    common = {
        "attempt_id": "attempt-1",
        "candidate_id": "candidate-1",
        "task_id": "task-1",
        "kernel_id": "kernel-1",
        "state_generation": 1,
        "anchor_generation": 0,
        "split": "train",
        "visibility": "public",
    }
    append_event(
        journal,
        run_id,
        "context_packet_created",
        {
            **common,
            "context_packet_id": packet.context_packet_id,
            "artifacts": [
                artifact_binding("context_packet", packet_receipt),
                artifact_binding("source", source),
                artifact_binding("harness", harness),
            ],
        },
        "attempt-1-context",
    )
    append_event(
        journal,
        run_id,
        "prompt_sent",
        {**common, "artifacts": [artifact_binding("prompt", prompt)]},
        "attempt-1-prompt",
    )
    append_event(
        journal,
        run_id,
        "tool_called",
        {**common, "tool_name": "profile", "call_id": "tool-1"},
        "attempt-1-tool-call",
    )
    append_event(
        journal,
        run_id,
        "tool_result",
        {
            **common,
            "tool_name": "profile",
            "call_id": "tool-1",
            "artifacts": [artifact_binding("tool_result", tool)],
        },
        "attempt-1-tool-result",
    )
    append_event(
        journal,
        run_id,
        "candidate_frozen",
        {**common, "artifacts": [artifact_binding("candidate", candidate)]},
        "attempt-1-candidate",
    )
    for event_type, key in (
        ("compile_result", "compile"),
        ("correctness_result", "correctness"),
    ):
        append_event(
            journal,
            run_id,
            event_type,
            {**common, "passed": True, "evidence_class": "measured"},
            f"attempt-1-{key}",
        )
    append_event(
        journal,
        run_id,
        "measurement_result",
        {
            **common,
            "evidence_class": "measured",
            "metrics": {"s50": 1.2, "s99": 1.1, "srobust": 1.1},
            "measurement_execution_sha256": execution_value.fingerprint,
            "measurement_writer_id": execution_value.writer_id,
            "measurement_harness_sha256": execution_value.harness_sha256,
            "artifacts": [
                artifact_binding("raw_measurement", measurement),
                artifact_binding("measurement_execution", execution),
                artifact_binding("harness", harness),
            ],
        },
        "attempt-1-measurement",
    )
    append_event(
        journal,
        run_id,
        "decision",
        {**common, "verdict": "keep", "reason": "measured_win"},
        "attempt-1-decision",
    )
    append_event(
        journal,
        run_id,
        "reward_committed",
        {
            **common,
            "evidence_class": "measured",
            "policy_id": "kernel_robust_v1",
            "scalar_reward": 140.0,
            "reward_vector": {
                "compile": True,
                "correctness": True,
                "integrity": True,
                "anti_tampering": True,
                "safety": {"finding": False, "policy_satisfied": True},
                "kernel_srobust": 1.1,
                "kernel_robust_reward": 140.0,
                "cost": {"gpu_seconds": 3.0, "tokens": 100},
            },
            "artifacts": [
                artifact_binding("raw_measurement", measurement),
                artifact_binding("measurement_execution", execution),
                artifact_binding("harness", harness),
                artifact_binding("reward_policy", policy),
            ],
        },
        "attempt-1-reward",
    )
    failure_common = {
        "attempt_id": "attempt-2",
        "candidate_id": "candidate-2",
        "task_id": "task-1",
        "kernel_id": "kernel-1",
        "state_generation": 2,
        "anchor_generation": 0,
        "split": "train",
        "visibility": "public",
    }
    append_event(
        journal,
        run_id,
        "context_packet_created",
        {
            **failure_common,
            "context_packet_id": second_packet.context_packet_id,
            "artifacts": [artifact_binding("context_packet", second_packet_receipt)],
        },
        "attempt-2-context",
    )
    append_event(
        journal,
        run_id,
        "agent_failed",
        {**failure_common, "reason_code": "backend_timeout", "retry": False},
        "attempt-2-error",
    )
    append_event(
        journal,
        run_id,
        "run.succeeded",
        {"workload_id": "workload-1", "reason": "completed"},
        "run-finished",
    )
    state = replay_workload_state(run_id, journal.iter_events(run_id))
    return {
        "run_id": run_id,
        "journal": journal,
        "artifacts": artifacts,
        "packet": packet,
        "packet_receipt": packet_receipt,
        "state": state,
        "root": tmp_path,
    }


@pytest.fixture
def e2e_no_source_run(tmp_path: Path):
    run_id = "run-e2e-no-source"
    attempt_id = "attempt-no-source"
    reason = "agent_made_no_source_change"
    journal = EventJournal(tmp_path / "events" / "run.db")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    packet = make_packet(run_id)
    packet_receipt = artifacts.put_bytes(
        packet.canonical_bytes, media_type="application/json"
    )
    manifest_document = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "candidate_id": None,
        "succeeded": False,
        "reason_code": reason,
        "workspace": "/isolated/worktree",
        "editable_files": ["kernels/norm.py"],
        "changed_files": [],
        "baseline_source_sha256": "a" * 64,
        "candidate_source_sha256": None,
        "frozen_sources": [],
        "source_receipts": [],
    }
    manifest = artifacts.put_bytes(
        canonical_json_bytes(manifest_document), media_type="application/json"
    )
    decision_document = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "candidate_id": None,
        "opportunity_id": "opportunity-1",
        "candidate_manifest_receipt": manifest.digest,
        "verdict": "reject",
        "reason_code": reason,
    }
    decision = artifacts.put_bytes(
        canonical_json_bytes(decision_document), media_type="application/json"
    )
    policy_document = E2ERewardPolicy().to_dict()
    policy = artifacts.put_bytes(
        canonical_json_bytes(policy_document), media_type="application/json"
    )
    grade = grade_e2e_outcome(
        verdict="reject",
        reason_code=reason,
        candidate_present=False,
    )
    grade_receipt = artifacts.put_bytes(
        canonical_json_bytes(grade.to_dict()), media_type="application/json"
    )
    common = {
        "attempt_id": attempt_id,
        "opportunity_id": "opportunity-1",
        "anchor_generation": 0,
        "split": "train",
        "visibility": "public",
    }
    append_event(
        journal,
        run_id,
        "run_started",
        {"workload_id": "workload-1", "task_id": "task-1"},
        "run-started",
    )
    append_event(
        journal,
        run_id,
        "e2e.opportunity_selected",
        {
            **common,
            "context_packet_id": packet.context_packet_id,
            "state_generation": 7,
        },
        "attempt-selected",
    )
    append_event(
        journal,
        run_id,
        "context_packet_created",
        {
            **common,
            "context_packet_id": packet.context_packet_id,
            "artifacts": [artifact_binding("context_packet", packet_receipt)],
        },
        "attempt-context",
    )
    append_event(
        journal,
        run_id,
        "candidate_frozen",
        {
            **common,
            "candidate_id": None,
            "succeeded": False,
            "reason_code": reason,
            "artifacts": [artifact_binding("candidate_manifest", manifest)],
        },
        "attempt-candidate",
    )
    append_event(
        journal,
        run_id,
        "e2e.execution_rejected",
        {
            **common,
            "receipt": manifest.digest,
            "reason": reason,
            "state_generation": 8,
        },
        "attempt-rejected",
    )
    outcome = append_event_transaction(
        journal,
        run_id,
        (
            (
                "e2e.candidate_decided",
                {
                    **common,
                    "receipt": decision.digest,
                    "verdict": "reject",
                    "reason": reason,
                    "state_generation": 9,
                    "artifacts": [artifact_binding("decision_evidence", decision)],
                },
                "attempt-decision",
            ),
            (
                "reward_committed",
                {
                    **common,
                    "verdict": "reject",
                    "reason_code": reason,
                    "policy_id": grade.policy_id,
                    "policy_digest": grade.policy_digest,
                    "scalar_reward": grade.scalar_reward,
                    "reward_vector": grade.to_dict(),
                    "evidence_class": "derived",
                    "artifacts": [
                        artifact_binding("decision_evidence", decision),
                        artifact_binding("e2e_grade", grade_receipt),
                        artifact_binding("reward_policy", policy),
                        artifact_binding("candidate_manifest", manifest),
                    ],
                },
                "attempt-reward",
            ),
        ),
    )
    append_event(
        journal,
        run_id,
        "run_finished",
        {"workload_id": "workload-1", "status": "succeeded"},
        "run-finished",
    )
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "journal": journal,
        "artifacts": artifacts,
        "packet": packet,
        "packet_receipt": packet_receipt,
        "manifest": manifest,
        "decision": decision,
        "grade": grade,
        "grade_receipt": grade_receipt,
        "policy": policy,
        "outcome_transaction_id": outcome.transaction_id,
    }
