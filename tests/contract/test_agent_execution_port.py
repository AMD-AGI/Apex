from __future__ import annotations

from dataclasses import replace

import pytest

from apex.core import AgentBackendName, ContractError
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentCaptureStatus,
    AgentCost,
    AgentInvocationReceipt,
    AgentProcessContainmentReceipt,
    AgentResult,
    AgentSemanticEvent,
    AgentTerminationKind,
    AgentUsage,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)


def _containment() -> AgentProcessContainmentReceipt:
    return AgentProcessContainmentReceipt(
        policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        launcher_path="/usr/bin/bwrap",
        launcher_sha256="b" * 64,
        namespace_init_host_pid=100,
        namespace_init_starttime=200,
        namespace_init_inner_pid=1,
        pid_namespace_inode=300,
        mount_namespace_inode=301,
        ipc_namespace_inode=302,
        user_namespace_inode=303,
        private_procfs_verified=True,
        pidfd_opened=True,
        termination_reason="stdout_budget_boundary",
        teardown_mode="pidfd_sigkill",
        pidfd_sigkill_sent=True,
        namespace_init_exit_verified=True,
        wrapper_exit_verified=True,
        wrapper_force_killed=False,
        terminal_status_verified=False,
        terminal_status_absent_after_sigkill=True,
        status_eof_verified=True,
        namespace_membership_scan_complete=True,
        live_namespace_members_after=(),
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
        process_containment_policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
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
        process_containment=_containment(),
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
    uncontained = AgentResult(
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
    assert not uncontained.candidate_capture_allowed
    assert (
        uncontained.candidate_rejection_reason
        == "agent_process_containment_unverified"
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
        process_containment=_containment(),
    )
    assert not truncated.candidate_capture_allowed
    assert truncated.candidate_rejection_reason == "agent_output_truncated"


@pytest.mark.parametrize(
    "receipt",
    (
        replace(_containment(), pidfd_opened=False),
        replace(_containment(), private_procfs_verified=False),
        replace(_containment(), namespace_init_exit_verified=False),
        replace(_containment(), wrapper_exit_verified=False),
        replace(_containment(), wrapper_force_killed=True),
        replace(_containment(), status_eof_verified=False),
        replace(_containment(), namespace_membership_scan_complete=False),
        replace(_containment(), live_namespace_members_after=(321,)),
        replace(
            _containment(),
            terminal_status_absent_after_sigkill=False,
        ),
    ),
)
def test_incomplete_process_containment_blocks_candidate_capture(
    receipt: AgentProcessContainmentReceipt,
) -> None:
    result = AgentResult(
        AgentBackendName.CODEX,
        None,
        137,
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
        process_containment=receipt,
    )

    assert not result.candidate_capture_allowed
    assert result.candidate_rejection_reason == "agent_process_containment_unverified"


def test_normal_completion_requires_verified_process_containment() -> None:
    contained = AgentResult(
        AgentBackendName.CODEX,
        None,
        0,
        False,
        (),
        "",
        "",
        0.1,
        process_containment=replace(
            _containment(),
            termination_reason="natural_exit",
            teardown_mode="natural_exit",
            pidfd_sigkill_sent=False,
            terminal_status_verified=True,
            terminal_status_absent_after_sigkill=False,
        ),
    )
    uncontained = replace(contained, process_containment=None)

    assert contained.succeeded and contained.candidate_capture_allowed
    assert not uncontained.succeeded and not uncontained.candidate_capture_allowed
