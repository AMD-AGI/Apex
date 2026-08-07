from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from apex.diagnostics import (
    AcquisitionCoverage,
    EvidenceArtifactReceipt,
    EvidenceArtifacts,
    KernelEvidence,
    KernelVolume,
    OperationEvidence,
    PerformanceModelEvidence,
    ShapeEvidence,
    TraceEvidence,
    derive_candidate_id,
)
from apex.core import IntegrityError
from apex.optimization.e2e import (
    CorrectnessOracleBinding,
    CorrectnessOracleRegistry,
    build_kernel_opportunity_plan,
)


def _evidence(tmp_path: Path, name: str, share: float, *, resolved: bool) -> TraceEvidence:
    root = tmp_path / name
    root.mkdir()
    source = root / "kernel.py"
    test = root / "test_kernel.py"
    if resolved:
        source.write_text("pass\n", encoding="utf-8")
        test.write_text("def test_ok(): pass\n", encoding="utf-8")
    kernel = KernelEvidence(
        runtime_name=name,
        language="triton",
        origin_library="aiter",
        source_path=str(source) if resolved else None,
        source_confidence="active_finder" if resolved else "unknown",
        patchable=resolved,
        source_root=str(root) if resolved else None,
        test_file=str(test) if resolved else None,
        test_command="pytest test_kernel.py" if resolved else None,
    )
    shape = ShapeEvidence(concrete_inputs=("[16, 128]",))
    candidate = derive_candidate_id(
        provenance_hash="a" * 64, phase="decode", rank=0, kernel=kernel, shape=shape
    )
    return TraceEvidence(
        1,
        candidate,
        "a" * 64,
        "decode",
        0,
        OperationEvidence("attention", name),
        kernel,
        shape,
        KernelVolume(10, share, share),
        PerformanceModelEvidence(),
        EvidenceArtifacts(
            "TargetedKernelTrace",
            AcquisitionCoverage(10, 10, 10, 0),
            (
                EvidenceArtifactReceipt(
                    "targeted_manifest",
                    "targeted_trace/manifest.json",
                    "b" * 64,
                    1,
                    "application/json",
                ),
                EvidenceArtifactReceipt(
                    "targeted_shard",
                    "targeted_trace/shards/rank-0.jsonl",
                    "d" * 64,
                    1,
                    "application/x-ndjson",
                ),
            ),
            "c" * 64,
        ),
    )


def test_dynamic_plan_preserves_unresolved_and_selects_by_measured_share(tmp_path: Path) -> None:
    unresolved = _evidence(tmp_path, "large_unknown", 40, resolved=False)
    eligible = _evidence(tmp_path, "smaller", 20, resolved=True)
    plan = build_kernel_opportunity_plan((eligible, unresolved), max_kernels=10)
    assert plan.opportunities[0].runtime_name == "large_unknown"
    assert plan.opportunities[0].reason_code == "source_unresolved"
    assert [item.runtime_name for item in plan.eligible] == ["smaller"]
    assert all(item.opportunity_id.startswith("kernel-") for item in plan.opportunities)


def test_config_or_non_source_candidates_cannot_enter_lane(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "hip_monolith", 50, resolved=True)
    kernel = replace(evidence.kernel, language="hip")
    # Changing a candidate identity-relevant field requires a new identity.
    candidate = derive_candidate_id(
        provenance_hash=evidence.provenance_hash,
        phase=evidence.phase,
        rank=evidence.rank,
        kernel=kernel,
        shape=evidence.shape,
    )
    evidence = replace(evidence, kernel=kernel, candidate_id=candidate)
    plan = build_kernel_opportunity_plan((evidence,), max_kernels=1)
    assert plan.eligible == ()
    assert plan.opportunities[0].reason_code == "unsupported_kernel_language"


def test_source_locked_oracle_makes_dynamically_ranked_source_eligible(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path, "runtime_symbol_can_change", 20, resolved=True)
    kernel = replace(evidence.kernel, test_file=None, test_command=None)
    candidate = derive_candidate_id(
        provenance_hash=evidence.provenance_hash,
        phase=evidence.phase,
        rank=evidence.rank,
        kernel=kernel,
        shape=evidence.shape,
    )
    evidence = replace(evidence, kernel=kernel, candidate_id=candidate)
    root = Path(kernel.source_root or "")
    registry = CorrectnessOracleRegistry(
        source_roots={"aiter": root},
        source_lock_sha256="e" * 64,
        bindings=(
            CorrectnessOracleBinding(
                "aiter",
                "kernel.py",
                "test_kernel.py",
                ("python", "-m", "pytest", "test_kernel.py", "-q"),
            ),
        ),
    )

    plan = build_kernel_opportunity_plan(
        (evidence,), max_kernels=1, correctness_oracles=registry
    )

    assert [item.runtime_name for item in plan.eligible] == [
        "runtime_symbol_can_change"
    ]
    assert plan.eligible[0].test_file == root / "test_kernel.py"
    assert plan.eligible[0].test_command == "python -m pytest test_kernel.py -q"
    assert len(plan.eligible[0].correctness_oracle_sha256 or "") == 64
    assert plan.correctness_oracle_policy_sha256 == registry.policy_sha256


def test_oracle_rejects_a_different_source_root(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, "kernel", 20, resolved=True)
    root = Path(evidence.kernel.source_root or "")
    registry = CorrectnessOracleRegistry(
        source_roots={"aiter": root},
        source_lock_sha256="e" * 64,
        bindings=(
            CorrectnessOracleBinding(
                "aiter",
                "kernel.py",
                "test_kernel.py",
                ("python", "-m", "pytest", "test_kernel.py"),
            ),
        ),
    )
    other = tmp_path / "other"
    other.mkdir()
    (other / "kernel.py").write_text("pass\n", encoding="utf-8")

    with pytest.raises(IntegrityError, match="differs"):
        registry.resolve(
            repository_id="aiter",
            source_root=other,
            source_path=other / "kernel.py",
        )
