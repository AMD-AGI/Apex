"""Narrow host authority for an explicitly requested kernel discovery session."""

from __future__ import annotations

from typing import Mapping

from apex.core import ContractError, new_identifier
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityGrantReceipt,
    CapabilityRewardRole,
    CapabilitySideEffect,
)


class KernelDraftSessionGrantAuthority:
    """Permit only an unverified campaign draft, never formal execution."""

    def __init__(self) -> None:
        self._consumed = False

    def supports(self, descriptor: CapabilityDescriptor) -> bool:
        return bool(
            descriptor.capability_id == "campaign.start"
            and descriptor.required_authority is CapabilityAuthority.WORKSPACE_USER
            and descriptor.gpu_requirement is CapabilityGpuRequirement.NONE
            and set(descriptor.side_effects)
            == {
                CapabilitySideEffect.READ_WORKSPACE,
                CapabilitySideEffect.WRITE_RESULTS,
            }
            and descriptor.reward_role is CapabilityRewardRole.EVIDENCE_ONLY
        )

    def authorize(
        self,
        *,
        session_id: str,
        descriptor: CapabilityDescriptor,
        arguments: Mapping[str, object],
        arguments_sha256: str,
    ) -> CapabilityGrantReceipt:
        del arguments
        if not self.supports(descriptor):
            raise AssertionError("unsupported descriptor reached kernel draft authority")
        if self._consumed:
            raise ContractError(
                "Kernel draft session authority was already consumed",
                "capability_grant_replayed",
            )
        self._consumed = True
        return CapabilityGrantReceipt(
            grant_id=new_identifier("draft-grant"),
            session_id=session_id,
            capability_id=descriptor.capability_id,
            descriptor_sha256=descriptor.digest,
            arguments_sha256=arguments_sha256,
            authority=descriptor.required_authority,
            side_effects=tuple(
                sorted(item.value for item in descriptor.side_effects)
            ),
            gpu_devices=(),
            timeout_seconds=descriptor.timeout_seconds,
            artifact_classes=tuple(sorted(descriptor.artifact_classes)),
            reward_role=descriptor.reward_role.value,
            cost_ceiling_microusd=descriptor.estimated_cost_microusd,
        )


__all__ = ["KernelDraftSessionGrantAuthority"]
