"""Recorded dispatcher for evaluator-owned formal kernel capabilities."""

from __future__ import annotations

from typing import Mapping

from apex.orchestration import RunPhase

from .formal_campaign import FormalKernelCampaign
from .formal_capability_recording import (
    begin_formal_capability,
    complete_formal_capability,
    fail_formal_capability,
)
from .formal_evaluator import KernelFormalEvaluator
from .formal_result import FormalEvaluatorResult


class KernelFormalCapabilityUseCase:
    """Dispatch one typed formal request and seal its exact call/result pair."""

    def __init__(self, evaluator: KernelFormalEvaluator) -> None:
        self._evaluator = evaluator

    def invoke(
        self,
        campaign: FormalKernelCampaign,
        capability_id: str,
        arguments: Mapping[str, object],
        *,
        granted_gpu_devices: str | None = None,
        grant_receipt: Mapping[str, object] | None = None,
    ) -> FormalEvaluatorResult:
        invocation = begin_formal_capability(
            campaign.record,
            capability_id,
            arguments,
            grant_receipt=grant_receipt,
        )
        try:
            result = self._dispatch(
                campaign,
                capability_id,
                arguments,
                granted_gpu_devices=granted_gpu_devices,
            )
        except Exception as error:
            fail_formal_capability(campaign.record, invocation, error)
            raise
        complete_formal_capability(
            campaign.record,
            invocation,
            _capability_content(capability_id, result),
        )
        if capability_id == "bundle.build":
            campaign.record.finish(
                RunPhase.SUCCEEDED, "verified_candidate_delivered"
            )
        return result

    def _dispatch(
        self,
        campaign,
        capability_id,
        arguments,
        *,
        granted_gpu_devices,
    ):
        if capability_id == "kernel.compile":
            return self._evaluator.compile(
                campaign,
                confirmed_draft_digest=str(arguments["confirmed_draft_digest"]),
                requested_devices=granted_gpu_devices,
            )
        if capability_id == "kernel.correctness":
            return self._evaluator.correctness(
                campaign,
                **_attempt_arguments(arguments),
                requested_devices=granted_gpu_devices,
            )
        if capability_id == "kernel.measure":
            return self._evaluator.measure(
                campaign,
                **_attempt_arguments(arguments),
                requested_devices=granted_gpu_devices,
            )
        if capability_id == "kernel.grade":
            return self._evaluator.grade(
                campaign,
                attempt_id=_optional_string(arguments, "attempt_id"),
                contract_digest=_optional_string(arguments, "contract_digest"),
                candidate_digest=_optional_string(arguments, "candidate_digest"),
            )
        if capability_id == "bundle.build":
            return self._evaluator.build_bundle(
                campaign, **_attempt_arguments(arguments), finish=False
            )
        raise AssertionError(f"unsupported evaluator capability: {capability_id}")


def _attempt_arguments(arguments) -> dict[str, str]:
    return {
        "attempt_id": str(arguments["attempt_id"]),
        "contract_digest": str(arguments["contract_digest"]),
        "candidate_digest": str(arguments["candidate_digest"]),
    }


def _optional_string(arguments, key: str) -> str | None:
    value = arguments.get(key)
    return str(value) if value is not None else None


def _capability_content(
    capability_id: str, result: FormalEvaluatorResult
) -> dict[str, object]:
    key = "verification" if capability_id == "bundle.build" else "receipt"
    return {key: dict(result.receipt)}


__all__ = ["KernelFormalCapabilityUseCase"]
