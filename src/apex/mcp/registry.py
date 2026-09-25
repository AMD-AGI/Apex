"""One run-scoped registry for capability schemas, authority, and handlers."""

from __future__ import annotations

from dataclasses import dataclass

from apex.core import ContractError
from apex.ports import (
    CapabilityAuthority,
    CapabilityAvailability,
    CapabilityDescriptor,
    CapabilityHandler,
    CapabilityKind,
    CapabilityGrantReceipt,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRewardRole,
    validate_capability_grant,
)


@dataclass(frozen=True, slots=True)
class _Entry:
    availability: CapabilityAvailability
    handler: CapabilityHandler | None


class CapabilityRegistry:
    """Fail-closed dispatch over the same descriptors presented to every backend."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(
        self,
        descriptor: CapabilityDescriptor,
        handler: CapabilityHandler | None,
        *,
        unavailable_reason: str | None = None,
    ) -> None:
        if descriptor.capability_id in self._entries:
            raise ContractError("Capability is already registered", "duplicate_capability")
        available = handler is not None
        if available == bool(unavailable_reason):
            raise ContractError("Capability registration is incoherent", "invalid_capability")
        availability = CapabilityAvailability(
            descriptor,
            available,
            unavailable_reason,
        )
        self._entries[descriptor.capability_id] = _Entry(availability, handler)

    def register_presentation(self, descriptor: CapabilityDescriptor) -> None:
        """Register an available non-tool surface mounted by the native host."""

        if descriptor.kind is not CapabilityKind.SKILL:
            raise ContractError("Presented capability is not a skill", "invalid_capability")
        if descriptor.capability_id in self._entries:
            raise ContractError("Capability is already registered", "duplicate_capability")
        self._entries[descriptor.capability_id] = _Entry(
            CapabilityAvailability(descriptor, True, None),
            None,
        )

    def inventory(self) -> tuple[CapabilityAvailability, ...]:
        return tuple(
            self._entries[name].availability for name in sorted(self._entries)
        )

    def validate_arguments(
        self, capability_id: str, arguments: object
    ) -> CapabilityDescriptor:
        """Validate identity, availability and schema before seeking approval."""

        entry = self._entry(capability_id)
        availability = entry.availability
        if not availability.available:
            raise ContractError(
                "Capability is unavailable",
                availability.unavailable_reason or "capability_unavailable",
            )
        if entry.handler is None:
            raise ContractError(
                "Capability is mounted by the native host, not callable",
                "capability_not_callable",
            )
        _validate(arguments, availability.descriptor.input_schema, "input")
        return availability.descriptor

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        descriptor = self.validate_arguments(
            request.capability_id, request.arguments
        )
        entry = self._entry(request.capability_id)
        assert entry.handler is not None
        authority = descriptor.required_authority
        if authority is not CapabilityAuthority.NONE and authority not in request.authorities:
            raise ContractError("Capability authority is missing", "capability_authority_missing")
        if request.grant is not None and not isinstance(
            request.grant, CapabilityGrantReceipt
        ):
            raise ContractError("Capability grant is invalid", "capability_grant_mismatch")
        if isinstance(request.grant, CapabilityGrantReceipt):
            validate_capability_grant(
                request.grant,
                session_id=request.grant.session_id,
                descriptor=descriptor,
                arguments=request.arguments,
            )
        result = entry.handler.invoke(request)
        if result.capability_id != request.capability_id:
            raise ContractError("Capability result identity differs", "capability_result_mismatch")
        evaluator_owned = descriptor.reward_role is CapabilityRewardRole.EVALUATOR_OWNED
        if result.reward_eligible and not evaluator_owned:
            raise ContractError("Capability reward role differs", "capability_result_mismatch")
        if request.grant is not None and (
            result.cost_microusd > request.grant.cost_ceiling_microusd
        ):
            raise ContractError("Capability cost exceeds its grant", "capability_cost_exceeded")
        _validate(result.content, descriptor.output_schema, "output")
        return result

    def _entry(self, capability_id: str) -> _Entry:
        try:
            return self._entries[capability_id]
        except KeyError as error:
            raise ContractError("Capability is not registered", "capability_not_registered") from error


def _validate(value, schema, label: str) -> None:
    from jsonschema import ValidationError, validate

    try:
        validate(instance=value, schema=schema)
    except ValidationError as error:
        raise ContractError(
            f"Capability {label} does not match its schema",
            "invalid_capability_arguments" if label == "input" else "capability_result_mismatch",
        ) from error


__all__ = ["CapabilityRegistry"]
