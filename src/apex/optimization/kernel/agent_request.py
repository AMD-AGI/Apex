"""Formal agent request construction for standalone kernel attempts."""

from __future__ import annotations

from apex.core import AgentBackendName, ContractError, TaskStatus
from apex.ports import (
    AgentExecutionAuthorityReceipt,
    AgentRequest,
    AgentTerminationKind,
)

from .attempts import AttemptSession


def build_agent_request(
    attempt: AttemptSession, backend: AgentBackendName
) -> AgentRequest:
    request = attempt.run.request
    task = request.task
    contract = attempt.run.evaluation_contract
    if not contract.verified or contract.authority is None:
        raise ContractError(
            "Formal agent execution requires an authorized evaluation contract",
            "agent_execution_authority_missing",
        )
    allowed_files = tuple(sorted(task.editable_files))
    authority = AgentExecutionAuthorityReceipt(
        authority_id="apex-kernel-controller-v1",
        authority_kind="evaluation_contract",
        run_id=attempt.run.run_id,
        attempt_id=attempt.attempt_id,
        backend=backend.value,
        workspace=str(attempt.candidate.root),
        allowed_files=allowed_files,
        requested_environment_keys=(),
        parent_receipt_sha256=contract.digest,
        source_anchor_sha256=contract.draft.digest,
    )
    return AgentRequest(
        run_id=attempt.run.run_id,
        attempt_id=attempt.attempt_id,
        backend=backend,
        prompt=attempt.context.prompt,
        workspace=attempt.candidate.root,
        allowed_files=allowed_files,
        execution_authority=authority,
        model=request.model_override or task.agent_options.model,
        effort=request.effort_override or task.agent_options.effort,
        max_turns=task.budget.max_turns,
        timeout_seconds=task.budget.timeout_seconds,
        runtime_closure_sha256=task.agent_options.runtime_closure_sha256,
    )


def agent_failure_status(kind: AgentTerminationKind) -> TaskStatus:
    if kind is AgentTerminationKind.TIMEOUT:
        return TaskStatus.TIMEOUT
    if kind is AgentTerminationKind.TURN_OVERRUN:
        return TaskStatus.BUDGET_EXHAUSTED
    return TaskStatus.INFRASTRUCTURE_ERROR


__all__ = ["agent_failure_status", "build_agent_request"]
