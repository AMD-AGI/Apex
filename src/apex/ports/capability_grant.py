"""Explicit one-call authorization contract for exposed capabilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Protocol

from apex.core import ContractError, sha256_json, validate_identifier

from .capability import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class CapabilityGrantReceipt:
    """One argument-bound, bounded permission minted outside the MCP channel."""

    grant_id: str
    session_id: str
    capability_id: str
    descriptor_sha256: str
    arguments_sha256: str
    authority: CapabilityAuthority
    side_effects: tuple[str, ...]
    gpu_devices: tuple[str, ...]
    timeout_seconds: int
    artifact_classes: tuple[str, ...]
    reward_role: str
    cost_ceiling_microusd: int

    SCHEMA = "apex.capability-grant/v1"

    def __post_init__(self) -> None:
        validate_identifier(self.grant_id, field_name="capability grant ID")
        validate_identifier(self.session_id, field_name="capability grant session ID")
        if not self.capability_id:
            _invalid("Capability grant identity is incomplete")
        for value in (self.descriptor_sha256, self.arguments_sha256):
            if _SHA256.fullmatch(value) is None:
                _invalid("Capability grant digest is invalid")
        if (
            tuple(sorted(set(self.side_effects))) != self.side_effects
            or len(set(self.gpu_devices)) != len(self.gpu_devices)
            or tuple(sorted(set(self.artifact_classes))) != self.artifact_classes
        ):
            _invalid("Capability grant scope is not canonical")
        if (
            type(self.timeout_seconds) is not int
            or self.timeout_seconds <= 0
            or type(self.cost_ceiling_microusd) is not int
            or self.cost_ceiling_microusd < 0
        ):
            _invalid("Capability grant budget is invalid")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "grant_id": self.grant_id,
            "session_id": self.session_id,
            "capability_id": self.capability_id,
            "descriptor_sha256": self.descriptor_sha256,
            "arguments_sha256": self.arguments_sha256,
            "authority": self.authority.value,
            "side_effects": list(self.side_effects),
            "gpu_devices": list(self.gpu_devices),
            "timeout_seconds": self.timeout_seconds,
            "artifact_classes": list(self.artifact_classes),
            "reward_role": self.reward_role,
            "cost_ceiling_microusd": self.cost_ceiling_microusd,
        }

    def to_receipt(self) -> dict[str, object]:
        """Return the transcript-safe projection plus its canonical digest."""

        return {**self.to_dict(), "receipt_sha256": self.digest}


class CapabilityGrantAuthority(Protocol):
    """Obtain caller approval without trusting tool arguments or model text."""

    def supports(self, descriptor: CapabilityDescriptor) -> bool:
        """Return whether this authority can mint the descriptor's exact role."""

    def authorize(
        self,
        *,
        session_id: str,
        descriptor: CapabilityDescriptor,
        arguments: Mapping[str, object],
        arguments_sha256: str,
    ) -> CapabilityGrantReceipt: ...


def validate_capability_grant(
    grant: CapabilityGrantReceipt,
    *,
    session_id: str,
    descriptor: CapabilityDescriptor,
    arguments: Mapping[str, object],
) -> None:
    """Require an exact descriptor/argument/budget binding before dispatch."""

    arguments_sha256 = sha256_json(arguments)
    gpu_valid = _gpu_scope_valid(grant, descriptor, arguments)
    valid = (
        grant.session_id == session_id
        and grant.capability_id == descriptor.capability_id
        and grant.descriptor_sha256 == descriptor.digest
        and grant.arguments_sha256 == arguments_sha256
        and grant.authority is descriptor.required_authority
        and grant.side_effects
        == tuple(sorted(item.value for item in descriptor.side_effects))
        and gpu_valid
        and grant.timeout_seconds <= descriptor.timeout_seconds
        and grant.artifact_classes == tuple(sorted(descriptor.artifact_classes))
        and grant.reward_role == descriptor.reward_role.value
        and grant.cost_ceiling_microusd >= descriptor.estimated_cost_microusd
    )
    if not valid:
        _invalid("Capability grant does not bind the requested invocation")


def _gpu_scope_valid(
    grant: CapabilityGrantReceipt,
    descriptor: CapabilityDescriptor,
    arguments: Mapping[str, object],
) -> bool:
    requested = _requested_gpu_devices(arguments)
    granted = grant.gpu_devices
    if descriptor.gpu_requirement is CapabilityGpuRequirement.NONE:
        return not granted and requested is None
    if requested is not None and requested != granted:
        return False
    if descriptor.gpu_requirement is CapabilityGpuRequirement.REQUIRED:
        return bool(granted)
    return requested is None or bool(granted)


def _requested_gpu_devices(
    arguments: Mapping[str, object],
) -> tuple[str, ...] | None:
    value = arguments.get("gpu_devices")
    if value is None:
        return None
    if not isinstance(value, str):
        return ()
    result = tuple(item.strip() for item in value.split(","))
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        return ()
    return result


def _invalid(message: str) -> None:
    raise ContractError(message, "capability_grant_mismatch")


__all__ = [
    "CapabilityGrantAuthority",
    "CapabilityGrantReceipt",
    "validate_capability_grant",
]
