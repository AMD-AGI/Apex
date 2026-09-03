from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import AgentBackendName, ContractError
from apex.optimization.e2e import (
    ComponentMicroBinding,
    ComponentMicroQualifierRegistry,
)
from apex.optimization.e2e.candidate import E2ECandidate
from apex.optimization.e2e.kernel_lane import KernelOpportunity
from apex.optimization.e2e.services import MicroQualification, MicroQualificationRequest
from apex.ports import AgentResult


class _Qualifier:
    qualification_mode = "e2e_quality_deferred"

    def __init__(self, route: str, library: str) -> None:
        self.route = route
        self.library = library
        self.calls: list[str] = []

    def supports(self, opportunity: KernelOpportunity) -> bool:
        return opportunity.origin_library == self.library

    def verify(self, request: MicroQualificationRequest) -> MicroQualification:
        self.calls.append(request.opportunity.origin_library)
        return MicroQualification(
            candidate_id="candidate-1",
            grade=None,
            evidence={"delegate": self.route},
            qualification_mode="e2e_quality_deferred",
            deferred_candidate_valid=True,
        )


def _opportunity(library: str) -> KernelOpportunity:
    return KernelOpportunity(
        "kernel-1",
        "evidence-1",
        "kernel",
        "operator",
        "decode",
        0,
        "triton",
        library,
        (),
        (),
        "eager",
        "high",
        10.0,
        10.0,
        Path("/source/kernel.py"),
        Path("/source"),
        Path("/source/test.py"),
        "pytest -q",
        "eligible",
        "eligible",
    )


def _request(tmp_path: Path, library: str) -> MicroQualificationRequest:
    candidate = E2ECandidate(
        "attempt-1",
        "candidate-1",
        True,
        "candidate_frozen",
        tmp_path,
        ("kernel.py",),
        ("kernel.py",),
        "a" * 64,
        "b" * 64,
        AgentResult(AgentBackendName.CODEX, None, 0, False, (), "", "", 0.1),
    )
    return MicroQualificationRequest(
        "run-1", candidate, _opportunity(library), tmp_path / "artifacts", 0, "gpu=0"
    )


def _registry(vllm: _Qualifier, aiter: _Qualifier) -> ComponentMicroQualifierRegistry:
    downstream = ("quality", "performance")
    return ComponentMicroQualifierRegistry(
        (
            ComponentMicroBinding("vllm", "strict-oracle", vllm, downstream),
            ComponentMicroBinding("aiter", "source-deferred", aiter, downstream),
        )
    )


def test_registry_routes_by_component_and_preserves_delegate_truth(tmp_path: Path) -> None:
    vllm = _Qualifier("oracle", "vllm")
    aiter = _Qualifier("deferred", "aiter")
    qualifier = _registry(vllm, aiter)

    result = qualifier.verify(_request(tmp_path, "vllm"))

    assert vllm.calls == ["vllm"] and aiter.calls == []
    assert result.evidence["delegate"] == "oracle"
    assert result.evidence["component_micro_qualification"] == {
        "schema": "apex.component-micro-qualification/v1",
        "source_component": "vllm",
        "route_id": "strict-oracle",
        "downstream_authorities": ["quality", "performance"],
    }
    assert qualifier.supported_components == frozenset({"vllm", "aiter"})


def test_registry_keeps_deferred_result_rewardless(tmp_path: Path) -> None:
    qualifier = _registry(
        _Qualifier("oracle", "vllm"), _Qualifier("deferred", "aiter")
    )

    result = qualifier.verify(_request(tmp_path, "aiter"))

    assert result.grade is None and result.kernel_reward_available is False
    assert result.evidence["component_micro_qualification"]["route_id"] == (
        "source-deferred"
    )


def test_registry_rejects_unowned_or_duplicate_components(tmp_path: Path) -> None:
    vllm = _Qualifier("oracle", "vllm")
    qualifier = ComponentMicroQualifierRegistry(
        (ComponentMicroBinding("vllm", "strict-oracle", vllm),)
    )

    with pytest.raises(ContractError) as unsupported:
        qualifier.verify(_request(tmp_path, "other"))
    assert unsupported.value.reason_code == "micro_qualification_unsupported"

    with pytest.raises(ContractError) as duplicate:
        ComponentMicroQualifierRegistry(
            (
                ComponentMicroBinding("vllm", "strict-oracle", vllm),
                ComponentMicroBinding(
                    "vllm", "second-oracle", _Qualifier("other", "vllm")
                ),
            )
        )
    assert duplicate.value.reason_code == "duplicate_component_micro_binding"


def test_registry_capability_receipt_contains_no_model_identity() -> None:
    qualifier = _registry(
        _Qualifier("oracle", "vllm"), _Qualifier("deferred", "aiter")
    )

    receipt = qualifier.capability_receipt()

    assert receipt["schema"] == "apex.component-micro-registry/v1"
    assert "model" not in str(receipt).lower()
    assert [item["source_component"] for item in receipt["bindings"]] == [
        "vllm",
        "aiter",
    ]
