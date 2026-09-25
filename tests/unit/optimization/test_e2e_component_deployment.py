from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.core import ContractError
from apex.core import ValidationLevel
from apex.optimization.e2e import (
    CandidateDeploymentRegistry,
    ComponentDeploymentBinding,
)
from apex.optimization.e2e.services import CandidateDeployment


class FakeDeployment:
    def __init__(self, adapter_id: str, component: str) -> None:
        self.adapter_id = adapter_id
        self.component = component
        self.deployments = 0
        self.rollbacks = 0

    def supports(self, opportunity, _provenance) -> bool:
        return opportunity.origin_library == self.component

    def deploy(self, request) -> CandidateDeployment:
        self.deployments += 1
        return _failed_deployment(request.candidate.candidate_id)

    def rollback(self, deployment) -> None:
        assert deployment.evidence["deployment_adapter_id"] == self.adapter_id
        self.rollbacks += 1


def _failed_deployment(candidate_id: str) -> CandidateDeployment:
    path = Path("/tmp/config.yaml")
    return CandidateDeployment(
        candidate_id,
        False,
        "not_deployed",
        path,
        path,
        path,
        "a" * 64,
        "b" * 64,
        None,
        ValidationLevel.NONE,
        False,
        {},
    )


def _binding(
    adapter: FakeDeployment, run_modes: frozenset[str] = frozenset({"docker"})
) -> ComponentDeploymentBinding:
    return ComponentDeploymentBinding(
        adapter.adapter_id,
        frozenset({adapter.component}),
        run_modes,
        adapter,
    )


def test_registry_routes_deploy_and_rollback_by_source_component() -> None:
    vllm = FakeDeployment("vllm-overlay-v1", "vllm")
    aiter = FakeDeployment("aiter-overlay-v1", "aiter")
    registry = CandidateDeploymentRegistry((_binding(vllm), _binding(aiter)))
    opportunity = SimpleNamespace(origin_library="aiter")
    provenance = SimpleNamespace(run_mode="docker")
    request = SimpleNamespace(
        opportunity=opportunity,
        provenance=provenance,
        candidate=SimpleNamespace(candidate_id="candidate-1"),
    )

    assert registry.supports(opportunity, provenance) is True
    result = registry.deploy(request)
    registry.rollback(result)

    assert (vllm.deployments, vllm.rollbacks) == (0, 0)
    assert (aiter.deployments, aiter.rollbacks) == (1, 1)
    assert result.evidence == {
        "deployment_adapter_id": "aiter-overlay-v1",
        "source_component": "aiter",
        "run_mode": "docker",
    }
    assert registry.supported_components == frozenset({"vllm", "aiter"})
    assert registry.supported_run_modes == frozenset({"docker"})


def test_registry_rejects_unowned_or_duplicate_components() -> None:
    first = FakeDeployment("first-overlay-v1", "vllm")
    second = FakeDeployment("second-overlay-v1", "vllm")

    with pytest.raises(ContractError) as duplicate:
        CandidateDeploymentRegistry((_binding(first), _binding(second)))
    assert duplicate.value.reason_code == "duplicate_component_deployment_binding"

    registry = CandidateDeploymentRegistry((_binding(first),))
    request = SimpleNamespace(
        opportunity=SimpleNamespace(origin_library="sglang"),
        provenance=SimpleNamespace(run_mode="docker"),
        candidate=SimpleNamespace(candidate_id="candidate-1"),
    )
    with pytest.raises(ContractError) as unavailable:
        registry.deploy(request)
    assert unavailable.value.reason_code == "delivery_adapter_unavailable"


def test_registry_allows_distinct_run_mode_owners_for_one_component() -> None:
    docker = FakeDeployment("vllm-docker-v1", "vllm")
    ray = FakeDeployment("vllm-ray-v1", "vllm")
    registry = CandidateDeploymentRegistry(
        (
            _binding(docker, frozenset({"docker"})),
            _binding(ray, frozenset({"ray"})),
        )
    )
    opportunity = SimpleNamespace(origin_library="vllm")

    assert registry.supports(
        opportunity, SimpleNamespace(run_mode="docker")
    ) is True
    request = SimpleNamespace(
        opportunity=opportunity,
        provenance=SimpleNamespace(run_mode="ray"),
        candidate=SimpleNamespace(candidate_id="candidate-1"),
    )
    result = registry.deploy(request)

    assert (docker.deployments, ray.deployments) == (0, 1)
    assert result.evidence["deployment_adapter_id"] == "vllm-ray-v1"
    assert result.evidence["run_mode"] == "ray"


def test_registry_capability_receipt_contains_no_model_identity() -> None:
    registry = CandidateDeploymentRegistry(
        (_binding(FakeDeployment("vllm-overlay-v1", "vllm")),)
    )

    assert registry.capability_receipt() == {
        "schema": "apex.e2e-component-deployment-registry/v2",
        "adapter_id": "e2e-component-deployment-registry-v2",
        "bindings": [
            {
                "adapter_id": "vllm-overlay-v1",
                "components": ["vllm"],
                "run_modes": ["docker"],
            }
        ],
    }
