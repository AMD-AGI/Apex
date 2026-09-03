"""One-shot grant enforcement for calls entering through the MCP server."""

from __future__ import annotations

from typing import Mapping

from apex.core import ContractError, new_identifier, sha256_json
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityGrantAuthority,
    CapabilityGrantReceipt,
    CapabilityRewardRole,
    CapabilityRequest,
    CapabilitySideEffect,
    validate_capability_grant,
)


class CapabilityGrantGate:
    """Resolve and consume external approval before a handler is invoked."""

    def __init__(
        self,
        authority: CapabilityGrantAuthority | None,
        *,
        session_id: str | None = None,
    ) -> None:
        self.authority = authority
        self.session_id = session_id or new_identifier("mcp-session")
        self._consumed: set[str] = set()

    def authorize(
        self,
        descriptor: CapabilityDescriptor,
        arguments: Mapping[str, object],
    ) -> CapabilityGrantReceipt | None:
        """Return no grant only for an inert, unauthorised, zero-GPU call."""

        if not capability_grant_required(descriptor):
            return None
        if self.authority is None:
            raise ContractError(
                "Capability invocation requires explicit caller approval",
                "capability_grant_required",
            )
        if not self.authority.supports(descriptor):
            raise ContractError(
                "No injected authority can approve this capability role",
                "capability_grant_unavailable",
            )
        arguments_sha256 = sha256_json(arguments)
        grant = self.authority.authorize(
            session_id=self.session_id,
            descriptor=descriptor,
            arguments=arguments,
            arguments_sha256=arguments_sha256,
        )
        if not isinstance(grant, CapabilityGrantReceipt):
            raise ContractError(
                "Capability grant authority returned an invalid receipt",
                "capability_grant_mismatch",
            )
        validate_capability_grant(
            grant,
            session_id=self.session_id,
            descriptor=descriptor,
            arguments=arguments,
        )
        if grant.grant_id in self._consumed:
            raise ContractError(
                "Capability grant was already consumed",
                "capability_grant_replayed",
            )
        self._consumed.add(grant.grant_id)
        return grant

    def available(self, descriptor: CapabilityDescriptor) -> bool:
        """Return whether this server instance can authorize the descriptor."""

        if not capability_grant_required(descriptor):
            return True
        return self.authority is not None and self.authority.supports(descriptor)


def capability_grant_required(descriptor: CapabilityDescriptor) -> bool:
    """Classify every non-inert or authority-bearing tool as approval-gated."""

    active_effect = any(
        item is not CapabilitySideEffect.NONE for item in descriptor.side_effects
    )
    return bool(
        active_effect
        or descriptor.required_authority is not CapabilityAuthority.NONE
        or descriptor.gpu_requirement is not CapabilityGpuRequirement.NONE
        or descriptor.reward_role is not CapabilityRewardRole.INELIGIBLE
    )


def granted_gpu_selector(request: CapabilityRequest) -> str | None:
    """Return the exact ordered selector approved for this invocation."""

    if isinstance(request.grant, CapabilityGrantReceipt):
        return ",".join(request.grant.gpu_devices) or None
    value = request.arguments.get("gpu_devices")
    return str(value) if value is not None else None


__all__ = [
    "CapabilityGrantGate",
    "capability_grant_required",
    "granted_gpu_selector",
]
