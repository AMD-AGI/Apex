"""Source-component routing for E2E candidate deployment adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from apex.core import ContractError, IntegrityError, validate_identifier
from apex.runtime import RunProvenance

from .kernel_lane import KernelOpportunity
from .services import (
    CandidateDeployment,
    CandidateDeploymentPort,
    CandidateDeploymentRequest,
)


@dataclass(frozen=True, slots=True)
class ComponentDeploymentBinding:
    """One adapter's exact source-component and run-mode capability claim."""

    adapter_id: str
    components: frozenset[str]
    run_modes: frozenset[str]
    adapter: CandidateDeploymentPort

    def __post_init__(self) -> None:
        validate_identifier(self.adapter_id, field_name="deployment adapter ID")
        if (
            not self.components
            or not self.run_modes
            or any(not value.strip() for value in (*self.components, *self.run_modes))
        ):
            raise ContractError(
                "Component deployment binding is empty",
                "invalid_component_deployment_binding",
            )
        declared = getattr(self.adapter, "adapter_id", self.adapter_id)
        if declared != self.adapter_id:
            raise ContractError(
                "Deployment adapter identity differs from its binding",
                "invalid_component_deployment_binding",
            )


class CandidateDeploymentRegistry:
    """Route by source component and run mode, never model or config filename."""

    adapter_id = "e2e-component-deployment-registry-v2"

    def __init__(self, bindings: tuple[ComponentDeploymentBinding, ...]) -> None:
        if not bindings:
            raise ContractError(
                "At least one component deployment binding is required",
                "invalid_component_deployment_binding",
            )
        routes: dict[tuple[str, str], ComponentDeploymentBinding] = {}
        ids: set[str] = set()
        for binding in bindings:
            if binding.adapter_id in ids:
                raise ContractError(
                    "Deployment adapter ID is duplicated",
                    "duplicate_component_deployment_binding",
                )
            ids.add(binding.adapter_id)
            for component in binding.components:
                for run_mode in binding.run_modes:
                    route = (component, run_mode)
                    if route in routes:
                        raise ContractError(
                            "Source component and run mode have multiple deployment owners",
                            "duplicate_component_deployment_binding",
                        )
                    routes[route] = binding
        self._bindings = bindings
        self._routes = routes

    @property
    def supported_components(self) -> frozenset[str]:
        return frozenset(component for component, _ in self._routes)

    @property
    def supported_run_modes(self) -> frozenset[str]:
        return frozenset(
            mode for binding in self._bindings for mode in binding.run_modes
        )

    def supports(
        self, opportunity: KernelOpportunity, provenance: RunProvenance
    ) -> bool:
        binding = self._binding(opportunity.origin_library, provenance.run_mode)
        return bool(binding and binding.adapter.supports(opportunity, provenance))

    def deploy(self, request: CandidateDeploymentRequest) -> CandidateDeployment:
        binding = self._binding(
            request.opportunity.origin_library, request.provenance.run_mode
        )
        if binding is None or not binding.adapter.supports(
            request.opportunity, request.provenance
        ):
            raise ContractError(
                "No deployment adapter supports this source component",
                "delivery_adapter_unavailable",
                {
                    "source_component": request.opportunity.origin_library,
                    "run_mode": request.provenance.run_mode,
                },
            )
        result = binding.adapter.deploy(request)
        evidence = dict(result.evidence)
        existing = evidence.get("deployment_adapter_id")
        if existing is not None and existing != binding.adapter_id:
            raise IntegrityError(
                "Deployment result claims another adapter",
                "deployment_adapter_mismatch",
            )
        evidence.update(
            {
                "deployment_adapter_id": binding.adapter_id,
                "source_component": request.opportunity.origin_library,
                "run_mode": request.provenance.run_mode,
            }
        )
        return replace(result, evidence=evidence)

    def rollback(self, deployment: CandidateDeployment) -> None:
        adapter_id = deployment.evidence.get("deployment_adapter_id")
        matches = tuple(
            binding for binding in self._bindings if binding.adapter_id == adapter_id
        )
        if len(matches) != 1:
            raise IntegrityError(
                "Deployment rollback adapter is unresolved",
                "deployment_adapter_mismatch",
            )
        matches[0].adapter.rollback(deployment)

    def capability_receipt(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-component-deployment-registry/v2",
            "adapter_id": self.adapter_id,
            "bindings": [
                {
                    "adapter_id": binding.adapter_id,
                    "components": sorted(binding.components),
                    "run_modes": sorted(binding.run_modes),
                }
                for binding in self._bindings
            ],
        }

    def _binding(
        self, component: str, run_mode: str
    ) -> ComponentDeploymentBinding | None:
        if run_mode not in {"docker", "local", "ray"}:
            raise ContractError(
                "Run provenance has an unsupported run mode",
                "invalid_provenance",
                {"run_mode": run_mode},
            )
        return self._routes.get((component, run_mode))


__all__ = ["CandidateDeploymentRegistry", "ComponentDeploymentBinding"]
