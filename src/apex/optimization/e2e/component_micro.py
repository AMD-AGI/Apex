"""Source-component routing for E2E micro-qualification authorities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from apex.core import ContractError, validate_identifier

from .kernel_lane import KernelOpportunity
from .services import (
    MicroQualification,
    MicroQualificationPort,
    MicroQualificationRequest,
)


@dataclass(frozen=True, slots=True)
class ComponentMicroBinding:
    """One component's reviewed qualifier and honest downstream authorities."""

    component: str
    route_id: str
    qualifier: MicroQualificationPort
    downstream_authorities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.component, field_name="source component")
        validate_identifier(self.route_id, field_name="micro route ID")
        if any(not item.strip() for item in self.downstream_authorities):
            raise ContractError(
                "Micro authority is empty", "invalid_component_micro_binding"
            )


class ComponentMicroQualifierRegistry:
    """Route qualification by source component without workload/model branches."""

    qualification_mode = "component_routed"

    def __init__(self, bindings: tuple[ComponentMicroBinding, ...]) -> None:
        if not bindings:
            raise ContractError(
                "At least one micro binding is required",
                "invalid_component_micro_binding",
            )
        routes: dict[str, ComponentMicroBinding] = {}
        route_ids: set[str] = set()
        for binding in bindings:
            if binding.component in routes or binding.route_id in route_ids:
                raise ContractError(
                    "Micro component or route has multiple owners",
                    "duplicate_component_micro_binding",
                )
            routes[binding.component] = binding
            route_ids.add(binding.route_id)
        self._bindings = bindings
        self._routes = routes

    @property
    def supported_components(self) -> frozenset[str]:
        return frozenset(self._routes)

    def supports(self, opportunity: KernelOpportunity) -> bool:
        binding = self._routes.get(opportunity.origin_library)
        return bool(binding and binding.qualifier.supports(opportunity))

    def verify(self, request: MicroQualificationRequest) -> MicroQualification:
        component = request.opportunity.origin_library
        binding = self._routes.get(component)
        if binding is None or not binding.qualifier.supports(request.opportunity):
            raise ContractError(
                "No micro-qualification adapter supports this source component",
                "micro_qualification_unsupported",
                {"source_component": component},
            )
        result = binding.qualifier.verify(request)
        evidence: Mapping[str, Any] = {
            **dict(result.evidence),
            "component_micro_qualification": {
                "schema": "apex.component-micro-qualification/v1",
                "source_component": component,
                "route_id": binding.route_id,
                "downstream_authorities": list(binding.downstream_authorities),
            },
        }
        return replace(result, evidence=evidence)

    def capability_receipt(self) -> dict[str, Any]:
        return {
            "schema": "apex.component-micro-registry/v1",
            "qualification_mode": self.qualification_mode,
            "bindings": [
                {
                    "source_component": binding.component,
                    "route_id": binding.route_id,
                    "downstream_authorities": list(
                        binding.downstream_authorities
                    ),
                }
                for binding in self._bindings
            ],
        }


__all__ = ["ComponentMicroBinding", "ComponentMicroQualifierRegistry"]
