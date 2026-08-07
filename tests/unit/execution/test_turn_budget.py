from __future__ import annotations

from apex.execution.turn_budget import AgentTurnBudget


def test_claude_tool_request_at_limit_stops_before_follow_up_turn() -> None:
    budget = AgentTurnBudget(2)

    assert not budget.observe(
        '{"type":"assistant","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"inspect"}]}}\n'
    )
    assert budget.observe(
        '{"type":"assistant","message":{"role":"assistant",'
        '"content":[{"type":"tool_use","name":"edit"}]}}\n'
    )
    assert budget.observed_turns == 2


def test_final_assistant_message_at_exact_limit_can_finish() -> None:
    budget = AgentTurnBudget(1)

    assert not budget.observe(
        '{"type":"assistant","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"done"}]}}\n'
    )
    assert not budget.observe('{"type":"result","num_turns":1}\n')
    assert budget.observed_turns == 1


def test_codex_structured_tool_action_consumes_a_turn() -> None:
    budget = AgentTurnBudget(1)

    assert budget.observe(
        '{"type":"item.started","item":{"id":"call-1",'
        '"type":"command_execution"}}\n'
    )
    assert budget.observed_turns == 1


def test_explicit_summary_over_limit_fails_closed() -> None:
    budget = AgentTurnBudget(3)

    assert budget.observe('{"type":"result","num_turns":4}\n')
    assert budget.observed_turns == 4


def test_non_json_diagnostics_do_not_invent_turns() -> None:
    budget = AgentTurnBudget(1)

    assert not budget.observe("warning: retrying\n")
    assert budget.observe("{malformed\n")
    assert budget.observed_turns == 0
    assert budget.enforcement_failed
    assert budget.stop_reason == "unparseable_structured_event"


def test_nominal_success_without_turn_evidence_is_unverifiable() -> None:
    budget = AgentTurnBudget(1)

    assert not budget.observe("warning only\n")
    budget.finalize(process_succeeded=True, observer_stopped=False)

    assert budget.enforcement_failed
    assert budget.stop_reason == "missing_structured_turn_evidence"
