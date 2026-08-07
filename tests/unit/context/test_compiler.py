from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import pytest

from apex.context import (
    AnchorView,
    ArtifactReference,
    ContextBudget,
    ContextCompileRequest,
    ContextCompiler,
    ContextContract,
    Hypothesis,
    TargetEvidence,
    freeze_metrics,
)
from apex.core import ContractError, sha256_bytes, sha256_json
from apex.knowledge import (
    ExperienceIdentity,
    ExperienceView,
    KnowledgeCard,
    KnowledgeRetriever,
    KnowledgeScope,
)


@dataclass(frozen=True)
class Event:
    sequence: int
    event_id: str
    event_type: str
    payload: Mapping[str, Any]


def test_response_token_allocation_is_explicitly_not_an_execution_limit() -> None:
    budget = ContextBudget(4096, 2048, 8, 600, 300).to_dict()

    assert budget["response_token_allocation"] == 2048
    assert budget["response_token_enforcement"] == "context_advisory_not_backend_enforced"
    assert "output_tokens" not in budget


def _card(claim: str, *, kind: str = "fact", gpu: str = "gfx950") -> KnowledgeCard:
    return KnowledgeCard.from_mapping(
        {
            "kind": kind,
            "status": "imported_unverified",
            "scope": {
                "operator": ["rms_norm"],
                "gpu_arch": [gpu],
                "dtype": ["fp16"],
                "regime": ["decode"],
                "language": ["triton"],
                "framework": ["vllm"],
                "versions": {"rocm": "7.2"},
            },
            "claim": claim,
            "apply": f"Try {claim}",
            "verify": "Measure it independently.",
            "caution": "This is advisory and cannot decide a winner.",
            "source": {
                "repository": "https://example.invalid/geak",
                "git_sha": "1" * 40,
                "path": f"perf_knowledge/{sha256_json(claim)[:8]}.md",
                "license": "Apache-2.0",
                "content_sha256": sha256_bytes(claim.encode()),
                "transform_version": "test_v1",
            },
        }
    )


def _identity() -> ExperienceIdentity:
    return ExperienceIdentity.from_mapping(
        {
            "task_id": "rms-norm",
            "operator": "rms_norm",
            "gpu_arch": "gfx950",
            "framework": "vllm",
            "versions": {"rocm": "7.2"},
            "shape_hash": sha256_json("shape"),
            "source_hash": sha256_json("source"),
            "harness_hash": sha256_json("harness"),
            "policy_hash": sha256_json("policy"),
        }
    )


def _experience() -> ExperienceView:
    identity = _identity().to_dict()
    events = []
    for sequence, outcome in ((1, "success"), (2, "regression"), (3, "no_gain")):
        events.append(
            Event(
                sequence,
                f"event-{sequence}",
                "experience.measured",
                {
                    "evidence_class": "measured",
                    "dry_run": False,
                    "candidate_id": f"candidate-{sequence}",
                    "identity": identity,
                    "outcome": outcome,
                    "strategy_fingerprint": sha256_json(f"strategy-{sequence}"),
                    "mechanism": f"Measured mechanism {sequence}",
                    "micro_verdict": outcome,
                    "e2e_verdict": None,
                    "evidence_receipts": [sha256_json(f"receipt-{sequence}")],
                    "failure_reason": None if outcome == "success" else f"Failure {sequence}",
                    "retry_condition": (
                        None if outcome == "success" else "Retry after anchor changes."
                    ),
                },
            )
        )
    return ExperienceView.from_events(events)


def _request(*, budget: int = 8_000, view: ExperienceView | None = None) -> ContextCompileRequest:
    digest = sha256_json("receipt")
    return ContextCompileRequest(
        run_id="run-1",
        workload_id="workload-1",
        phase="searching",
        cycle=2,
        state_generation=7,
        role_kind="kernel_candidate_generator",
        role_objective="Optimize one measured bottleneck.",
        primary_metric="throughput",
        hard_constraints=("correctness", "integrity", "p99"),
        target=TargetEvidence(
            "opportunity-1", "source/kernel.py:rms_norm", "decode/shape-a", (digest,)
        ),
        hypothesis=Hypothesis(
            "hypothesis-1",
            "Repeated normalization loads may be fused.",
            "Reject if invocation p99 or correctness regresses.",
        ),
        current_anchor=AnchorView(
            "anchor-7",
            7,
            freeze_metrics({"throughput": 100.0, "ttft_p99_ms": 20.0}),
            (sha256_json("patch"),),
        ),
        budget=ContextBudget(budget, 2_000, 20, 600, 300),
        contract=ContextContract(
            ("read", "edit", "request_evidence", "submit_patch"),
            ("source/kernel.py",),
            "Correctness first; Srobust > 1.05; no E2E regression.",
            "Stop on budget exhaustion or no open hypothesis.",
        ),
        artifact_refs=(ArtifactReference("source", digest, f"artifact://sha256/{digest}"),),
        retrieval_scope=KnowledgeScope.from_mapping(
            {
                "operator": ["rms_norm"],
                "gpu_arch": ["gfx950"],
                "dtype": ["fp16"],
                "regime": ["decode"],
                "language": ["triton"],
                "framework": ["vllm"],
                "versions": {"rocm": "7.2"},
            }
        ),
        experience_identity=_identity(),
        experience_view=view or _experience(),
    )


def _compiler() -> ContextCompiler:
    cards = (
        _card("Fuse normalization loads"),
        _card("Avoid oversized workgroups", kind="anti_pattern"),
        _card("Check vector width", kind="procedure"),
    )
    return ContextCompiler(KnowledgeRetriever(cards))


def test_compiler_is_byte_stable_and_includes_receipts_dead_ends_and_cards() -> None:
    compiler = _compiler()

    first = compiler.compile(_request())
    second = compiler.compile(_request())

    assert first == second
    assert first.packet.canonical_bytes == second.packet.canonical_bytes
    assert first.packet.context_packet_id == second.packet.context_packet_id
    assert 2 <= len(first.packet.knowledge_cards) <= 4
    assert len(first.packet.attempts) == 3
    assert len(first.packet.dead_ends) == 2
    assert all(item.retry_condition for item in first.packet.dead_ends)
    assert first.estimated_input_tokens <= first.packet.budget.input_tokens
    assert first.receipt["knowledge_selection"] == first.knowledge_selection.to_dict()


def test_history_is_trimmed_before_hard_facts() -> None:
    compiler = _compiler()
    full = compiler.compile(_request())
    constrained_budget = full.estimated_input_tokens - 250
    constrained = compiler.compile(_request(budget=constrained_budget))

    assert constrained.packet.hard_constraints == ("correctness", "integrity", "p99")
    assert constrained.packet.current_anchor.anchor_id == "anchor-7"
    assert len(constrained.packet.knowledge_cards) >= 2
    assert len(constrained.packet.attempts) < len(full.packet.attempts)
    assert constrained.estimated_input_tokens <= constrained_budget


def test_mandatory_context_fails_explicitly_when_budget_is_too_small() -> None:
    with pytest.raises(ContractError) as failure:
        _compiler().compile(_request(budget=10))
    assert failure.value.reason_code == "mandatory_context_over_budget"


def test_artifact_locator_and_metrics_are_integrity_checked() -> None:
    digest = sha256_json("artifact")
    with pytest.raises(ContractError) as locator:
        ArtifactReference("source", digest, f"artifact://sha256/{'f' * 64}")
    assert locator.value.reason_code == "invalid_artifact_locator"

    with pytest.raises(ContractError) as metric:
        freeze_metrics({"throughput": float("nan")})
    assert metric.value.reason_code == "invalid_context_metrics"


def test_scope_mismatch_returns_typed_empty_without_old_fallback() -> None:
    request = _request()
    request = replace(
        request,
        retrieval_scope=KnowledgeScope.from_mapping(
            {
                "operator": ["rms_norm"],
                "gpu_arch": ["gfx942"],
                "language": ["triton"],
                "versions": {"rocm": "7.2"},
            }
        ),
    )

    compiled = _compiler().compile(request)

    assert compiled.packet.knowledge_cards == ()
    assert compiled.packet.knowledge_unavailable_reason == "insufficient_complementary_cards"
