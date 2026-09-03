from __future__ import annotations

from apex.execution.transcript import parse_agent_output


def test_codex_jsonl_normalizes_messages_tools_and_usage() -> None:
    output = "\n".join(
        (
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"id":"msg-1","type":"agent_message","text":"Inspecting the kernel"}}',
            '{"type":"item.started","item":{"id":"call-1","type":"command_execution","command":"pytest -q"}}',
            '{"type":"item.completed","item":{"id":"call-1","type":"command_execution","status":"completed","exit_code":0}}',
            '{"type":"turn.completed","usage":{"input_tokens":120,"cached_input_tokens":20,"output_tokens":30,"output_tokens_details":{"reasoning_tokens":7}}}',
        )
    )

    parsed = parse_agent_output(output)

    assert [event.kind for event in parsed.semantic_events] == [
        "agent_message",
        "tool_called",
        "tool_result",
    ]
    assert parsed.semantic_events[1].tool_name == "command_execution"
    assert parsed.semantic_events[2].succeeded is True
    assert parsed.usage is not None
    assert parsed.usage.input_tokens == 120
    assert parsed.usage.cached_input_tokens == 20
    assert parsed.usage.output_tokens == 30
    assert parsed.usage.reasoning_tokens == 7
    assert parsed.usage.total_tokens == 150
    assert parsed.usage.turn_count == 1
    assert parsed.usage.tool_call_count == 1
    assert parsed.cost is None


def test_claude_jsonl_uses_final_structured_summary_without_double_counting() -> None:
    output = "\n".join(
        (
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Profiling"},{"type":"tool_use","id":"tool-1","name":"profile","input":{"path":"kernel.py"}}],"usage":{"input_tokens":10,"output_tokens":2}}}',
            '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tool-1","content":"ok","is_error":false}]}}',
            '{"type":"result","subtype":"success","num_turns":3,"total_cost_usd":0.012300,"usage":{"input_tokens":100,"cache_read_input_tokens":40,"cache_creation_input_tokens":5,"output_tokens":25}}',
        )
    )

    parsed = parse_agent_output(output)

    assert [event.kind for event in parsed.semantic_events] == [
        "agent_message",
        "tool_called",
        "tool_result",
    ]
    assert parsed.semantic_events[2].tool_call_id == "tool-1"
    assert parsed.semantic_events[2].succeeded is True
    assert parsed.usage is not None
    assert parsed.usage.input_tokens == 100
    assert parsed.usage.output_tokens == 25
    assert parsed.usage.cached_input_tokens == 40
    assert parsed.usage.cache_creation_input_tokens == 5
    assert parsed.usage.turn_count == 3
    assert parsed.usage.tool_call_count == 1
    assert parsed.cost is not None
    assert parsed.cost.amount == "0.0123"
    assert parsed.cost.currency == "USD"
    assert parsed.cost.source_key == "total_cost_usd"


def test_human_text_and_malformed_json_never_create_usage_or_cost() -> None:
    output = "\n".join(
        (
            'usage={"input_tokens":999,"total_cost_usd":999}',
            '{"type":"agent_message","text":"I used 777 tokens and cost $42"}',
            '{"type":"turn.completed","usage":',
        )
    )

    parsed = parse_agent_output(output)

    assert [event.kind for event in parsed.events] == [
        "agent_message",
        "malformed_json",
    ]
    assert len(parsed.semantic_events) == 1
    assert parsed.usage is not None
    assert parsed.usage.input_tokens is None
    assert parsed.usage.turn_count == 1
    assert parsed.cost is None


def test_explicit_structured_non_usd_cost_remains_exact() -> None:
    parsed = parse_agent_output(
        '{"type":"result","usage":{"input_tokens":1},'
        '"cost":{"amount":"1.2300","currency":"EUR"}}'
    )

    assert parsed.cost is not None
    assert parsed.cost.to_dict() == {
        "amount": "1.23",
        "currency": "EUR",
        "source_event_index": 0,
        "source_key": "cost",
    }


def test_cursor_tool_call_envelope_preserves_name_phase_and_outcome() -> None:
    output = "\n".join(
        (
            '{"type":"tool_call","subtype":"started","tool_call":{"shell":{"args":{"command":"pytest"}}},"tool_call_id":"call-7"}',
            '{"type":"tool_call","subtype":"completed","tool_call":{"shell":{"result":"ok"}},"tool_call_id":"call-7","success":true}',
        )
    )

    parsed = parse_agent_output(output)

    assert [event.kind for event in parsed.semantic_events] == [
        "tool_called",
        "tool_result",
    ]
    assert all(event.tool_name == "shell" for event in parsed.semantic_events)
    assert all(event.tool_call_id == "call-7" for event in parsed.semantic_events)
    assert parsed.semantic_events[1].succeeded is True
    assert parsed.usage is not None
    assert parsed.usage.tool_call_count == 1
