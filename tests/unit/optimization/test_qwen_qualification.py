from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import AgentBackendName, ContractError
from apex.optimization.e2e.candidate import E2ECandidate
from apex.optimization.e2e.kernel_lane import KernelOpportunity
from apex.optimization.e2e.qwen_qualification import QwenCompositeMicroQualifier
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


def test_composite_routes_vllm_to_strict_oracle_lane(tmp_path: Path) -> None:
    vllm = _Qualifier("oracle", "vllm")
    aiter = _Qualifier("deferred", "aiter")
    qualifier = QwenCompositeMicroQualifier(vllm=vllm, aiter=aiter)

    result = qualifier.verify(_request(tmp_path, "vllm"))

    assert vllm.calls == ["vllm"] and aiter.calls == []
    assert result.evidence["delegate"] == "oracle"
    assert result.evidence["qwen_composite_qualification"]["route"] == (
        "reviewed_vllm_docker_oracle"
    )


def test_composite_routes_aiter_to_deferred_lane_without_reward(tmp_path: Path) -> None:
    vllm = _Qualifier("oracle", "vllm")
    aiter = _Qualifier("deferred", "aiter")
    qualifier = QwenCompositeMicroQualifier(vllm=vllm, aiter=aiter)

    result = qualifier.verify(_request(tmp_path, "aiter"))

    assert aiter.calls == ["aiter"] and vllm.calls == []
    assert result.grade is None and result.kernel_reward_available is False
    assert result.evidence["qwen_composite_qualification"]["route"] == (
        "frozen_source_deferred"
    )


def test_composite_rejects_unreviewed_source_library(tmp_path: Path) -> None:
    qualifier = QwenCompositeMicroQualifier(
        vllm=_Qualifier("oracle", "vllm"),
        aiter=_Qualifier("deferred", "aiter"),
    )

    with pytest.raises(ContractError, match="No Qwen qualification lane"):
        qualifier.verify(_request(tmp_path, "other"))
