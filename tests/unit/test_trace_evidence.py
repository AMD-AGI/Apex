from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError, sha256_bytes
from apex.diagnostics import (
    AcquisitionCoverage,
    EvidenceArtifactReceipt,
    EvidenceArtifacts,
    KernelEvidence,
    KernelVolume,
    MagpieTraceEvidenceAdapter,
    OperationEvidence,
    PerformanceModelEvidence,
    ShapeEvidence,
    TraceEvidence,
    TraceEvidenceNormalizer,
    derive_candidate_id,
    predicted_e2e_gain_pct,
    rank_evidence,
)
from apex.ports import DiagnosticsRequest


PROVENANCE = "a" * 64


def _record(
    name: str,
    share: float,
    *,
    shape: str = "[16, 128]",
    roofline: float | None = None,
    confidence: str = "high",
) -> TraceEvidence:
    kernel = KernelEvidence(name, "triton", "aiter", "ops/kernel.py", 12, "exact_launch", True)
    shape_evidence = ShapeEvidence(concrete_inputs=(shape,), graph_mode="eager")
    candidate = derive_candidate_id(
        provenance_hash=PROVENANCE,
        phase="decode",
        rank=0,
        kernel=kernel,
        shape=shape_evidence,
    )
    model = (
        PerformanceModelEvidence(True, 1.0, 2.0, 0.5, "gfx950", "memory", roofline, confidence)
        if roofline is not None
        else PerformanceModelEvidence()
    )
    return TraceEvidence(
        1,
        candidate,
        PROVENANCE,
        "decode",
        0,
        OperationEvidence("attention", name),
        kernel,
        shape_evidence,
        KernelVolume(10, share, share),
        model,
        EvidenceArtifacts(
            "TargetedKernelTrace",
            AcquisitionCoverage(
                10,
                8,
                7,
                3,
                (("cap", 1), ("sampling", 2)),
            ),
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
        "exact",
    )


def test_candidate_identity_includes_shape_but_not_timing() -> None:
    first = _record("kernel", 20, shape="[16, 128]")
    same_identity = _record("kernel", 30, shape="[16, 128]")
    other_shape = _record("kernel", 20, shape="[32, 128]")
    assert first.candidate_id == same_identity.candidate_id
    assert first.candidate_id != other_shape.candidate_id


def test_trace_evidence_round_trip_revalidates_identity() -> None:
    record = _record("kernel", 20)
    assert TraceEvidence.from_mapping(record.to_dict()) == record


def test_candidate_identity_tampering_is_rejected() -> None:
    valid = _record("kernel", 20)
    with pytest.raises(IntegrityError, match="identity"):
        TraceEvidence(
            valid.schema_version,
            "0" * 64,
            valid.provenance_hash,
            valid.phase,
            valid.rank,
            valid.op,
            valid.kernel,
            valid.shape,
            valid.volume,
            valid.perf_model,
            valid.evidence,
        )


@pytest.mark.parametrize(
    "coverage",
    [
        (-1, 0, 0, 0, ()),
        (1, 2, 1, 0, ()),
        (2, 2, 1, 1, (("sampling", 1),)),
        (2, 1, 1, 1, (("cap", 1),)),
    ],
)
def test_coverage_accounting_fails_closed(
    coverage: tuple[int, int, int, int, tuple[tuple[str, int], ...]]
) -> None:
    with pytest.raises(ContractError):
        AcquisitionCoverage(*coverage)


def test_dual_ranking_preserves_measured_order_and_uses_trusted_headroom() -> None:
    saturated = _record("large", 40, roofline=95)
    headroom = _record("smaller", 20, roofline=10)
    low_confidence = _record("uncertain", 30, roofline=99, confidence="low")
    rankings = rank_evidence((saturated, headroom, low_confidence), min_gpu_pct=1)
    assert [item.candidate_id for item in rankings.measured] == [
        saturated.candidate_id,
        low_confidence.candidate_id,
        headroom.candidate_id,
    ]
    assert rankings.recoverable[0].candidate_id == low_confidence.candidate_id
    assert rankings.recoverable[0].perf_model_used is False
    assert rankings.recoverable[1].candidate_id == headroom.candidate_id


def test_amdahl_prediction() -> None:
    assert predicted_e2e_gain_pct(20, 2) == pytest.approx(11.111111, rel=1e-5)
    with pytest.raises(ContractError):
        predicted_e2e_gain_pct(20, 0)


def _write_report(workspace: Path) -> Path:
    repository = workspace / "repos" / "aiter"
    (repository / "ops").mkdir(parents=True)
    (repository / "ops" / "kernel.py").write_text("pass\n", encoding="utf-8")
    gap_dir = workspace / "gap_analysis"
    gap_dir.mkdir(parents=True)
    gap = gap_dir / "gap_analysis.csv"
    with gap.open("w", newline="", encoding="utf-8") as handle:
        handle.write(f"# Base directory: {workspace / 'repos'}\n")
        handle.write("# $AITER_DIR=./aiter\n")
        handle.write("#\n")
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Name",
                "Calls",
                "Self CUDA total (us)",
                "Avg time (us)",
                "% Total",
                "Input Shapes",
                "kind",
                "category",
                "source_repo",
                "source_file",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Name": "triton_kernel",
                "Calls": "20",
                "Self CUDA total (us)": "1000",
                "Avg time (us)": "50",
                "% Total": "25",
                "Input Shapes": "[16, 128]",
                "kind": "triton_jit",
                "category": "attention",
                "source_repo": "aiter",
                "source_file": "$AITER_DIR/ops/kernel.py",
            }
        )
    report = workspace / "benchmark_report.json"
    report.write_text(
        json.dumps(
            {
                "success": True,
                "kernel_summary": [
                    {"name": "triton_kernel", "time_ms": 1.0, "percent": 25.0, "calls": 20}
                ],
                "gap_analysis": {"csv_path": str(gap)},
            }
        ),
        encoding="utf-8",
    )
    return report


def test_magpie_report_without_targeted_trace_is_rejected(tmp_path: Path) -> None:
    report = _write_report(tmp_path)
    with pytest.raises(IntegrityError) as captured:
        TraceEvidenceNormalizer().from_benchmark_report(
            report, provenance_hash=PROVENANCE
        )
    assert captured.value.reason_code == "missing_targeted_trace"


def test_diagnostics_adapter_does_not_fallback_to_aggregate_trace(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    _write_report(benchmark)
    output = tmp_path / "normalized"
    result = MagpieTraceEvidenceAdapter().analyze(
        DiagnosticsRequest("run-1", benchmark, output, PROVENANCE)
    )
    assert not result.succeeded
    assert result.error == "Diagnostic report has no TargetedKernelTrace evidence"


def test_normalizer_rejects_invalid_report(tmp_path: Path) -> None:
    report = tmp_path / "benchmark_report.json"
    report.write_text("[]", encoding="utf-8")
    with pytest.raises(IntegrityError):
        TraceEvidenceNormalizer().from_benchmark_report(report, provenance_hash=PROVENANCE)
