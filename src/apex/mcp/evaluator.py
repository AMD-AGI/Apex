"""Thin MCP handlers for the evaluator-owned standalone kernel phases."""

from __future__ import annotations

from apex.optimization.kernel import (
    FormalKernelCampaign,
    KernelFormalCapabilityUseCase,
)
from apex.ports import (
    CapabilityGrantReceipt,
    CapabilityRequest,
    CapabilityResult,
)

from .scope import CapabilityScope
from .grants import granted_gpu_selector


class KernelEvaluatorHandler:
    """Resolve scoped state and delegate all evaluation policy downward."""

    def __init__(
        self,
        scope: CapabilityScope,
        evaluator: KernelFormalCapabilityUseCase,
    ) -> None:
        self._scope = scope
        self._evaluator = evaluator

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        arguments = request.arguments
        campaign = FormalKernelCampaign.load(
            self._scope.read_results(str(arguments["run_locator"])),
            workspace=self._scope.workspace,
            results=self._scope.results,
        )
        capability = request.capability_id
        grant = request.grant
        result = self._evaluator.invoke(
            campaign,
            capability,
            arguments,
            granted_gpu_devices=granted_gpu_selector(request),
            grant_receipt=(
                grant.to_receipt()
                if isinstance(grant, CapabilityGrantReceipt)
                else None
            ),
        )
        key = "verification" if capability == "bundle.build" else "receipt"
        return CapabilityResult(
            capability,
            {key: dict(result.receipt)},
            tuple(receipt.to_dict() for receipt in result.artifacts),
            result.reward_eligible,
        )
__all__ = ["KernelEvaluatorHandler"]
