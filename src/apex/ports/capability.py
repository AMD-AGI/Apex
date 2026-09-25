"""Backend-neutral contracts for lazily exposed Apex capabilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from apex.core import ContractError, sha256_json


class CapabilityKind(str, Enum):
    SKILL = "skill"
    TOOL = "tool"
    CAMPAIGN = "campaign"
    DELIVERY = "delivery"


class CapabilityAuthority(str, Enum):
    NONE = "none"
    WORKSPACE_USER = "workspace_user"
    FORMAL_EVALUATOR = "formal_evaluator"
    EXTERNAL_EVALUATOR = "external_evaluator"


class CapabilitySideEffect(str, Enum):
    NONE = "none"
    READ_WORKSPACE = "read_workspace"
    READ_RESULTS = "read_results"
    WRITE_WORKSPACE = "write_workspace"
    WRITE_RESULTS = "write_results"
    RUN_PROCESS = "run_process"


class CapabilityGpuRequirement(str, Enum):
    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class CapabilityRewardRole(str, Enum):
    INELIGIBLE = "ineligible"
    EVIDENCE_ONLY = "evidence_only"
    EVALUATOR_OWNED = "evaluator_owned"


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Stable schema and authority declaration for one capability."""

    capability_id: str
    title: str
    summary: str
    kind: CapabilityKind
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    side_effects: tuple[CapabilitySideEffect, ...]
    required_authority: CapabilityAuthority
    gpu_requirement: CapabilityGpuRequirement
    timeout_seconds: int
    artifact_classes: tuple[str, ...]
    reward_role: CapabilityRewardRole
    estimated_cost_microusd: int = 0

    def __post_init__(self) -> None:
        if (
            not self.capability_id
            or not self.title.strip()
            or not self.summary.strip()
            or isinstance(self.timeout_seconds, bool)
            or self.timeout_seconds <= 0
        ):
            raise ContractError("Capability descriptor is incomplete", "invalid_capability")
        if any(not value for value in self.artifact_classes):
            raise ContractError("Capability artifact class is invalid", "invalid_capability")
        if (
            type(self.estimated_cost_microusd) is not int
            or self.estimated_cost_microusd < 0
        ):
            raise ContractError("Capability cost estimate is invalid", "invalid_capability")
        if len(set(self.side_effects)) != len(self.side_effects):
            raise ContractError("Capability side effects are duplicated", "invalid_capability")
        _validate_schema(self.input_schema, "input")
        _validate_schema(self.output_schema, "output")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "title": self.title,
            "summary": self.summary,
            "kind": self.kind.value,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "side_effects": [item.value for item in self.side_effects],
            "required_authority": self.required_authority.value,
            "gpu_requirement": self.gpu_requirement.value,
            "timeout_seconds": self.timeout_seconds,
            "artifact_classes": list(self.artifact_classes),
            "reward_role": self.reward_role.value,
            "estimated_cost_microusd": self.estimated_cost_microusd,
        }


@dataclass(frozen=True, slots=True)
class CapabilityAvailability:
    descriptor: CapabilityDescriptor
    available: bool
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if self.available == bool(self.unavailable_reason):
            raise ContractError("Capability availability is incoherent", "invalid_capability")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.descriptor.to_dict(),
            "descriptor_digest": self.descriptor.digest,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class CapabilityRequest:
    capability_id: str
    arguments: Mapping[str, Any]
    authorities: frozenset[CapabilityAuthority] = field(default_factory=frozenset)
    grant: object | None = None


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    capability_id: str
    content: Mapping[str, Any]
    artifact_receipts: tuple[Mapping[str, Any], ...] = ()
    reward_eligible: bool = False
    cost_microusd: int = 0

    def __post_init__(self) -> None:
        if type(self.cost_microusd) is not int or self.cost_microusd < 0:
            raise ContractError(
                "Capability result cost is invalid", "capability_result_mismatch"
            )


class CapabilityHandler(Protocol):
    def invoke(self, request: CapabilityRequest) -> CapabilityResult: ...


class CapabilityPort(Protocol):
    def inventory(self) -> tuple[CapabilityAvailability, ...]: ...

    def invoke(self, request: CapabilityRequest) -> CapabilityResult: ...


def _validate_schema(value: Mapping[str, Any], label: str) -> None:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"Capability {label} schema is not JSON", "invalid_capability"
        ) from error
    if not isinstance(decoded, dict) or decoded.get("type") != "object":
        raise ContractError(
            f"Capability {label} schema must describe an object", "invalid_capability"
        )


__all__ = [
    "CapabilityAuthority",
    "CapabilityAvailability",
    "CapabilityDescriptor",
    "CapabilityGpuRequirement",
    "CapabilityHandler",
    "CapabilityKind",
    "CapabilityPort",
    "CapabilityRequest",
    "CapabilityResult",
    "CapabilityRewardRole",
    "CapabilitySideEffect",
]
