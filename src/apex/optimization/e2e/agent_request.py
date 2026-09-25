"""Formal controller authority binding for one E2E candidate invocation."""

from __future__ import annotations

from apex.intake import E2EOptimizeSpec

from .candidate import E2ECandidateRequest
from .kernel_lane import KernelOpportunity
from .run_record import E2ERunRecord


def build_e2e_candidate_request(
    *,
    spec: E2EOptimizeSpec,
    record: E2ERunRecord,
    attempt_id: str,
    opportunity: KernelOpportunity,
    prompt: str,
    context_packet_sha256: str,
) -> E2ECandidateRequest:
    """Issue permission from exact run and durable context controller state."""

    return E2ECandidateRequest(
        run_id=record.run_id,
        attempt_id=attempt_id,
        opportunity=opportunity,
        prompt=prompt,
        destination=record.root / "worktrees" / attempt_id,
        backend=spec.agent_backend,
        model=spec.agent_model,
        effort=spec.agent_effort,
        max_turns=spec.max_turns,
        timeout_seconds=spec.agent_timeout_seconds,
        controller_context_sha256=context_packet_sha256,
    )


__all__ = ["build_e2e_candidate_request"]
