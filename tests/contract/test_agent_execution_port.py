from __future__ import annotations

import pytest

from apex.core import AgentBackendName, ContractError
from apex.ports import (
    AgentCaptureStatus,
    AgentCost,
    AgentInvocationReceipt,
    AgentResult,
    AgentSemanticEvent,
    AgentTerminationKind,
    AgentUsage,
    BOUNDARY_QUIESCENCE_POLICY,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)


def test_agent_result_defaults_preserve_minimal_fake_backends() -> None:
    result = AgentResult(
        AgentBackendName.CODEX,
        None,
        0,
        False,
        (),
        "",
        "",
        0.1,
    )

    assert result.semantic_events == ()
    assert result.usage is None
    assert result.cost is None
    assert result.effort is None


def test_normalized_agent_evidence_validates_lineage_and_numeric_domain() -> None:
    usage = AgentUsage(input_tokens=10, output_tokens=5, source_event_indices=(2,))
    cost = AgentCost("0.0100", "usd", 2, "total_cost_usd")
    event = AgentSemanticEvent(0, 1, "assistant", "agent_message", text="done")

    assert usage.to_dict()["source_event_indices"] == [2]
    assert cost.amount == "0.01" and cost.currency == "USD"
    assert event.to_dict()["source_event_index"] == 1

    with pytest.raises(ContractError):
        AgentUsage(input_tokens=-1)
    with pytest.raises(ContractError):
        AgentCost("NaN", "USD", 0, "cost")
    with pytest.raises(ContractError):
        AgentSemanticEvent(0, 0, "assistant", "unknown")


def _invocation(max_turns: int = 2) -> AgentInvocationReceipt:
    return AgentInvocationReceipt(
        cli_name="codex",
        cli_version="test",
        executable_path="/usr/bin/codex",
        resolved_executable_path="/usr/bin/codex",
        entrypoint_sha256="a" * 64,
        argv=("codex", "exec"),
        workspace="/tmp/workspace",
        prompt_transport="stdin",
        requested_allowed_files=("kernel.py",),
        allowed_files_enforced_by_cli=False,
        max_turns=max_turns,
        turn_policy=STRUCTURED_TURN_CHECKPOINT_POLICY,
        boundary_quiescence_policy_id=BOUNDARY_QUIESCENCE_POLICY,
        isolation=(("sandbox", "workspace-write"),),
    )


def test_exact_boundary_is_typed_and_candidate_capture_is_fail_closed() -> None:
    result = AgentResult(
        AgentBackendName.CODEX,
        None,
        0,
        False,
        (),
        "",
        "",
        0.1,
        invocation=_invocation(),
        termination_kind=AgentTerminationKind.EXACT_TURN_BOUNDARY,
        termination_reason="max_turns_exact_boundary",
        observed_turns=2,
        observer_stop_sent=True,
        observer_suspend_sent=True,
        suspension_verified=True,
    )

    assert not result.succeeded
    assert result.candidate_capture_allowed
    assert result.candidate_rejection_reason is None
    with pytest.raises(ContractError):
        AgentResult(
            AgentBackendName.CODEX,
            None,
            0,
            False,
            (),
            "",
            "",
            0.1,
            invocation=_invocation(),
            termination_kind=AgentTerminationKind.EXACT_TURN_BOUNDARY,
            termination_reason="max_turns_exact_boundary",
            observed_turns=1,
        )
    unsuspended = AgentResult(
        AgentBackendName.CODEX,
        None,
        0,
        False,
        (),
        "",
        "",
        0.1,
        invocation=_invocation(),
        termination_kind=AgentTerminationKind.EXACT_TURN_BOUNDARY,
        termination_reason="max_turns_exact_boundary",
        observed_turns=2,
    )
    assert not unsuspended.candidate_capture_allowed
    assert (
        unsuspended.candidate_rejection_reason
        == "agent_boundary_suspension_unverified"
    )

    truncated = AgentResult(
        AgentBackendName.CODEX,
        None,
        -15,
        False,
        (),
        "",
        "",
        0.1,
        invocation=_invocation(),
        termination_kind=AgentTerminationKind.EXACT_TURN_BOUNDARY,
        capture_status=AgentCaptureStatus.OUTPUT_TRUNCATED,
        termination_reason="max_turns_exact_boundary",
        observed_turns=2,
        observer_stop_sent=True,
        observer_suspend_sent=True,
        suspension_verified=True,
    )
    assert not truncated.candidate_capture_allowed
    assert truncated.candidate_rejection_reason == "agent_output_truncated"
