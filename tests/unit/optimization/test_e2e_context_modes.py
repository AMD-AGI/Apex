from __future__ import annotations

from types import SimpleNamespace

from apex.optimization.e2e.context import (
    _agent_contract,
    _hard_constraints,
    _harness_contract,
    _hypothesis,
)
from apex.optimization.e2e.kernel_lane import KernelOpportunity


def _opportunity() -> KernelOpportunity:
    return KernelOpportunity(
        opportunity_id="kernel-1",
        evidence_id="evidence-1",
        runtime_name="kernel_fn",
        operation_name="attention",
        phase="decode",
        rank=1,
        language="triton",
        origin_library="vllm",
        shape_summary=("[1,128]",),
        dtypes=("float16",),
        graph_mode="eager",
        match_confidence="exact",
        measured_gpu_pct=10.0,
        roi_prior=1.0,
        source_path=None,
        source_root=None,
        test_file=None,
        test_command="pytest test_kernel.py",
        eligibility="eligible",
        reason_code="eligible",
    )


def test_deferred_context_does_not_claim_strict_micro_evidence() -> None:
    opportunity = _opportunity()
    harness = _harness_contract(
        opportunity,
        "vllm/kernel.py",
        "e2e_quality_deferred",
    )
    hypothesis = _hypothesis(opportunity, 0, "e2e_quality_deferred")
    contract = _agent_contract("vllm/kernel.py", "e2e_quality_deferred")

    policy = harness["promotion_policy"]
    assert isinstance(policy, dict)
    assert policy["qualification_mode"] == "e2e_quality_deferred"
    assert "explicitly unmeasured" in str(policy["micro"])
    assert "No kernel-level compile" in hypothesis.falsification_condition
    assert "Kernel reward is unavailable" in contract.acceptance_policy
    spec = SimpleNamespace(
        goal=SimpleNamespace(
            gates=SimpleNamespace(
                ttft_p99_regression_pct=5.0,
                tpot_p99_regression_pct=2.0,
            )
        )
    )
    constraints = _hard_constraints(
        spec,
        "vllm/kernel.py",
        "e2e_quality_deferred",
    )
    assert any("No trusted micro timing harness" in item for item in constraints)
    assert all("Micro promotion requires" not in item for item in constraints)


def test_strict_context_retains_canonical_kernel_grade_contract() -> None:
    opportunity = _opportunity()
    harness = _harness_contract(opportunity, "vllm/kernel.py", "strict_micro")
    hypothesis = _hypothesis(opportunity, 0, "strict_micro")
    contract = _agent_contract("vllm/kernel.py", "strict_micro")

    policy = harness["promotion_policy"]
    assert isinstance(policy, dict)
    assert policy["qualification_mode"] == "strict_micro"
    assert "compile+correct+integrity" in str(policy["micro"])
    assert "independent micro grader" in hypothesis.falsification_condition
    assert "evaluator-owned micro" in contract.acceptance_policy
