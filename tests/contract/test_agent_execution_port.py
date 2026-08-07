from __future__ import annotations

import pytest

from apex.core import AgentBackendName, ContractError
from apex.ports import AgentCost, AgentResult, AgentSemanticEvent, AgentUsage


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


def test_budget_failure_is_typed_and_cannot_report_success() -> None:
    result = AgentResult(
        AgentBackendName.CODEX,
        None,
        0,
        False,
        (),
        "",
        "",
        0.1,
        budget_exceeded=True,
        budget_reason="max_turns_exceeded",
        observed_turns=2,
    )

    assert not result.succeeded
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
            budget_exceeded=True,
        )
