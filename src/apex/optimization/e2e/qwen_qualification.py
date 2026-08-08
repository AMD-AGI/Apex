"""Composite micro-qualification policy for the reviewed Qwen workload."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from apex.core import ContractError

from .kernel_lane import KernelOpportunity
from .services import (
    MicroQualification,
    MicroQualificationPort,
    MicroQualificationRequest,
)


class QwenCompositeMicroQualifier:
    """Route each Qwen source library to its honest qualification authority.

    Reviewed vLLM sources have a tests-only Docker oracle. AITER does not have
    an equivalent trusted micro harness in the reviewed workload, so it may
    only prove frozen-source integrity before the unchanged safety, quality,
    and E2E performance gates make the promotion decision.
    """

    qualification_mode = "e2e_quality_deferred"

    def __init__(
        self,
        *,
        vllm: MicroQualificationPort,
        aiter: MicroQualificationPort,
    ) -> None:
        self._vllm = vllm
        self._aiter = aiter

    def supports(self, opportunity: KernelOpportunity) -> bool:
        qualifier = self._qualifier(opportunity)
        return qualifier is not None and qualifier.supports(opportunity)

    def verify(self, request: MicroQualificationRequest) -> MicroQualification:
        qualifier = self._qualifier(request.opportunity)
        if qualifier is None or not qualifier.supports(request.opportunity):
            raise ContractError(
                "No Qwen qualification lane supports this opportunity",
                "micro_qualification_unsupported",
                {"origin_library": request.opportunity.origin_library},
            )
        result = qualifier.verify(request)
        expected = {
            "vllm": "reviewed_vllm_docker_oracle",
            "aiter": "frozen_source_deferred",
        }[request.opportunity.origin_library]
        evidence: Mapping[str, Any] = {
            **dict(result.evidence),
            "qwen_composite_qualification": {
                "schema": "apex.qwen-composite-qualification/v1",
                "origin_library": request.opportunity.origin_library,
                "route": expected,
                "downstream_authority": [
                    "evaluator_owned_safety_gate",
                    "unchanged_magpie_quality_gate",
                    "unchanged_magpie_e2e_performance_gate",
                ],
            },
        }
        return replace(result, evidence=evidence)

    def _qualifier(
        self, opportunity: KernelOpportunity
    ) -> MicroQualificationPort | None:
        if opportunity.origin_library == "vllm":
            return self._vllm
        if opportunity.origin_library == "aiter":
            return self._aiter
        return None


__all__ = ["QwenCompositeMicroQualifier"]
