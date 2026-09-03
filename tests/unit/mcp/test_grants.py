from __future__ import annotations

from dataclasses import dataclass

import pytest

from apex.core import ContractError
from apex.mcp import CapabilityGrantGate, CapabilityRegistry
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityGrantReceipt,
    CapabilityKind,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRewardRole,
    CapabilitySideEffect,
)


def _descriptor(*, inert: bool = False) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="test.inert" if inert else "test.formal",
        title="Test capability",
        summary="Exercise the explicit capability grant boundary.",
        kind=CapabilityKind.TOOL,
        input_schema={"type": "object", "additionalProperties": False},
        output_schema={"type": "object"},
        side_effects=(
            (CapabilitySideEffect.NONE,)
            if inert
            else (CapabilitySideEffect.RUN_PROCESS, CapabilitySideEffect.WRITE_RESULTS)
        ),
        required_authority=(
            CapabilityAuthority.NONE
            if inert
            else CapabilityAuthority.FORMAL_EVALUATOR
        ),
        gpu_requirement=(
            CapabilityGpuRequirement.NONE
            if inert
            else CapabilityGpuRequirement.REQUIRED
        ),
        timeout_seconds=30,
        artifact_classes=() if inert else ("measurement",),
        reward_role=(
            CapabilityRewardRole.INELIGIBLE
            if inert
            else CapabilityRewardRole.EVALUATOR_OWNED
        ),
        estimated_cost_microusd=0 if inert else 5,
    )


@dataclass
class _Authority:
    grant_id: str = "grant-1"
    calls: int = 0

    def supports(self, descriptor) -> bool:
        del descriptor
        return True

    def authorize(
        self, *, session_id, descriptor, arguments, arguments_sha256
    ):
        del arguments
        self.calls += 1
        return CapabilityGrantReceipt(
            grant_id=self.grant_id,
            session_id=session_id,
            capability_id=descriptor.capability_id,
            descriptor_sha256=descriptor.digest,
            arguments_sha256=arguments_sha256,
            authority=descriptor.required_authority,
            side_effects=tuple(sorted(item.value for item in descriptor.side_effects)),
            gpu_devices=("0",),
            timeout_seconds=20,
            artifact_classes=tuple(sorted(descriptor.artifact_classes)),
            reward_role=descriptor.reward_role.value,
            cost_ceiling_microusd=7,
        )


def test_non_inert_capability_requires_external_one_shot_grant() -> None:
    descriptor = _descriptor()
    with pytest.raises(ContractError) as missing:
        CapabilityGrantGate(None, session_id="session-1").authorize(descriptor, {})
    assert missing.value.reason_code == "capability_grant_required"

    authority = _Authority()
    gate = CapabilityGrantGate(authority, session_id="session-1")
    grant = gate.authorize(descriptor, {})
    assert grant is not None
    assert grant.authority is CapabilityAuthority.FORMAL_EVALUATOR
    assert grant.gpu_devices == ("0",)
    assert authority.calls == 1

    with pytest.raises(ContractError) as replayed:
        gate.authorize(descriptor, {})
    assert replayed.value.reason_code == "capability_grant_replayed"


def test_inert_capability_never_asks_authority_for_a_grant() -> None:
    authority = _Authority()
    grant = CapabilityGrantGate(authority, session_id="session-1").authorize(
        _descriptor(inert=True), {}
    )

    assert grant is None
    assert authority.calls == 0


def test_grant_binding_and_cost_ceiling_fail_before_success() -> None:
    descriptor = _descriptor()
    authority = _Authority()
    grant = CapabilityGrantGate(authority, session_id="session-1").authorize(
        descriptor, {}
    )
    assert grant is not None

    class Handler:
        calls = 0

        def invoke(self, request):
            self.calls += 1
            return CapabilityResult(
                request.capability_id,
                {"ok": True},
                reward_eligible=True,
                cost_microusd=8,
            )

    handler = Handler()
    registry = CapabilityRegistry()
    registry.register(descriptor, handler)
    request = CapabilityRequest(
        descriptor.capability_id,
        {},
        frozenset({CapabilityAuthority.FORMAL_EVALUATOR}),
        grant,
    )

    with pytest.raises(ContractError) as cost:
        registry.invoke(request)
    assert cost.value.reason_code == "capability_cost_exceeded"
    assert handler.calls == 1

    drifted = CapabilityRequest(
        descriptor.capability_id,
        {"unexpected": True},
        request.authorities,
        grant,
    )
    with pytest.raises(ContractError) as invalid:
        registry.invoke(drifted)
    assert invalid.value.reason_code == "invalid_capability_arguments"
    assert handler.calls == 1


def test_grant_cost_ceiling_rejects_before_handler_dispatch() -> None:
    descriptor = _descriptor()
    authority = _Authority()
    authority_cost = authority

    class LowCeilingAuthority:
        def supports(self, descriptor):
            return authority_cost.supports(descriptor)

        def authorize(
            self, *, session_id, descriptor, arguments, arguments_sha256
        ):
            receipt = authority_cost.authorize(
                session_id=session_id,
                descriptor=descriptor,
                arguments=arguments,
                arguments_sha256=arguments_sha256,
            )
            return CapabilityGrantReceipt(
                grant_id=receipt.grant_id,
                session_id=receipt.session_id,
                capability_id=receipt.capability_id,
                descriptor_sha256=receipt.descriptor_sha256,
                arguments_sha256=receipt.arguments_sha256,
                authority=receipt.authority,
                side_effects=receipt.side_effects,
                gpu_devices=receipt.gpu_devices,
                timeout_seconds=receipt.timeout_seconds,
                artifact_classes=receipt.artifact_classes,
                reward_role=receipt.reward_role,
                cost_ceiling_microusd=4,
            )

    with pytest.raises(ContractError) as bounded:
        CapabilityGrantGate(
            LowCeilingAuthority(), session_id="session-1"
        ).authorize(descriptor, {})
    assert bounded.value.reason_code == "capability_grant_mismatch"


def test_authority_role_unavailable_is_not_presented_or_called() -> None:
    descriptor = _descriptor()

    class UserOnlyAuthority(_Authority):
        def supports(self, descriptor) -> bool:
            return descriptor.required_authority is CapabilityAuthority.WORKSPACE_USER

    authority = UserOnlyAuthority()
    gate = CapabilityGrantGate(authority, session_id="session-1")
    assert gate.available(descriptor) is False
    with pytest.raises(ContractError) as unavailable:
        gate.authorize(descriptor, {})
    assert unavailable.value.reason_code == "capability_grant_unavailable"
    assert authority.calls == 0


def test_grant_preserves_gpu_order_and_rejects_argument_drift() -> None:
    descriptor = _descriptor()

    class OrderedAuthority(_Authority):
        def authorize(
            self, *, session_id, descriptor, arguments, arguments_sha256
        ):
            receipt = super().authorize(
                session_id=session_id,
                descriptor=descriptor,
                arguments=arguments,
                arguments_sha256=arguments_sha256,
            )
            return CapabilityGrantReceipt(
                grant_id=receipt.grant_id,
                session_id=receipt.session_id,
                capability_id=receipt.capability_id,
                descriptor_sha256=receipt.descriptor_sha256,
                arguments_sha256=receipt.arguments_sha256,
                authority=receipt.authority,
                side_effects=receipt.side_effects,
                gpu_devices=("4", "2"),
                timeout_seconds=receipt.timeout_seconds,
                artifact_classes=receipt.artifact_classes,
                reward_role=receipt.reward_role,
                cost_ceiling_microusd=receipt.cost_ceiling_microusd,
            )

    arguments = {"gpu_devices": "4,2"}
    grant = CapabilityGrantGate(
        OrderedAuthority(), session_id="session-1"
    ).authorize(descriptor, arguments)
    assert grant is not None
    assert grant.gpu_devices == ("4", "2")

    with pytest.raises(ContractError) as mismatch:
        CapabilityGrantGate(
            OrderedAuthority(), session_id="session-1"
        ).authorize(descriptor, {"gpu_devices": "2,4"})
    assert mismatch.value.reason_code == "capability_grant_mismatch"
