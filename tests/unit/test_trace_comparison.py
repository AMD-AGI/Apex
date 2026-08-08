from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apex.core import IntegrityError
from apex.diagnostics import PinnedTraceLensComparisonAdapter
from apex.ports import (
    TraceComparisonArtifact,
    TraceComparisonRequest,
    TraceComparisonStatus,
    TraceDiagnosticEvidence,
)
from apex.storage import ArtifactReceipt, ArtifactStore


class _Comparator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        output_csvs = Path(kwargs["output_csvs_dir"])
        output_csvs.mkdir(parents=True)
        (output_csvs / "comparison.csv").write_text(
            "metric,baseline,terminal\ngpu_time,10,9\n", encoding="utf-8"
        )
        output = Path(kwargs["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"minimal-test-workbook")
        return {"gpu_timeline": object()}


def _adapter(
    tmp_path: Path, comparator: _Comparator
) -> PinnedTraceLensComparisonAdapter:
    module = (
        tmp_path / "TraceLens" / "Reporting" / "compare_perf_reports_pytorch.py"
    )
    module.parent.mkdir(parents=True)
    module.write_text("# pinned report comparison API\n", encoding="utf-8")
    return PinnedTraceLensComparisonAdapter(
        root=tmp_path,
        commit="a" * 40,
        report_comparator=comparator,
    )


def _artifact(
    receipt: ArtifactReceipt, *, role: str, logical_path: str
) -> TraceComparisonArtifact:
    return TraceComparisonArtifact(
        role,
        logical_path,
        receipt.digest,
        receipt.size,
        receipt.media_type,
        receipt.relative_path,
    )


def _diagnostic(
    store: ArtifactStore,
    *,
    marker: str,
    report_name: str = "gpu_timeline.csv",
) -> TraceDiagnosticEvidence:
    benchmark = store.put_bytes(
        f'{{"marker":"{marker}"}}'.encode(), media_type="application/json"
    )
    raw = store.put_bytes(
        f"raw-trace-{marker}".encode(), media_type="application/gzip"
    )
    report = store.put_bytes(
        f"type,time ms\nkernel-{marker},1.0\n".encode(), media_type="text/csv"
    )
    return TraceDiagnosticEvidence(
        marker * 64,
        store.root.resolve(),
        (
            _artifact(
                benchmark,
                role="diagnostic_benchmark_report",
                logical_path="metadata/benchmark_report.json",
            ),
            _artifact(
                raw,
                role="diagnostic_raw_trace",
                logical_path="raw/rank0.pt.trace.json.gz",
            ),
            _artifact(
                report,
                role="diagnostic_tracelens_report",
                logical_path=f"reports/decode/{report_name}",
            ),
        ),
    )


def _request(
    tmp_path: Path,
    *,
    report_name: str = "gpu_timeline.csv",
) -> TraceComparisonRequest:
    store = ArtifactStore(tmp_path / "cas")
    return TraceComparisonRequest(
        "run-1",
        "gfx950",
        _diagnostic(store, marker="b", report_name=report_name),
        _diagnostic(store, marker="c", report_name=report_name),
        "d" * 64,
        (tmp_path / "comparison").resolve(),
    )


def test_pinned_adapter_runs_report_diff_without_claiming_attribution(
    tmp_path: Path,
) -> None:
    comparator = _Comparator()
    request = _request(tmp_path)
    result = _adapter(tmp_path / "dependency", comparator).compare(request)

    assert result.status is TraceComparisonStatus.PARTIAL
    assert result.reward_eligible is False
    assert result.reason_code == (
        "tracelens_perf_report_comparison_succeeded_full_attribution_unavailable"
    )
    assert result.summary["claims"] == {
        "comparison_performed": True,
        "attribution_performed": False,
        "performance_grade_emitted": False,
        "reward_emitted": False,
    }
    assert result.summary["full_attribution"] == {
        "status": "unavailable",
        "reason_code": "pinned_tracelens_full_attribution_contract_unavailable",
    }
    assert comparator.calls[0]["sheets"] == ["gpu_timeline"]
    assert len(result.artifacts) == 2
    assert {path.suffix for path in result.artifacts} == {".csv", ".xlsx"}
    assert all(path.is_file() for path in result.artifacts)
    baseline = result.summary["inputs"]["baseline"]
    assert baseline["artifacts"][0]["receipt"]["relative_path"].startswith(
        "sha256/"
    )


def test_grouped_ops_selector_accepts_one_present_member(tmp_path: Path) -> None:
    comparator = _Comparator()
    request = _request(tmp_path, report_name="ops_unique_args.csv")

    result = _adapter(tmp_path / "dependency", comparator).compare(request)

    assert result.status is TraceComparisonStatus.PARTIAL
    assert comparator.calls[0]["sheets"] == ["ops_all"]
    assert result.summary["groups"][0]["input_files"] == ["ops_unique_args.csv"]


def test_pinned_adapter_marks_missing_terminal_trace_as_failed(tmp_path: Path) -> None:
    comparator = _Comparator()
    request = _request(tmp_path)
    terminal = TraceDiagnosticEvidence(
        request.terminal.trace_evidence_sha256,
        request.terminal.cas_root,
        tuple(
            item
            for item in request.terminal.artifacts
            if item.role != "diagnostic_raw_trace"
        ),
    )

    result = _adapter(tmp_path / "dependency", comparator).compare(
        TraceComparisonRequest(
            request.run_id,
            request.gpu_arch,
            request.baseline,
            terminal,
            request.terminal_benchmark_sha256,
            request.output_dir,
        )
    )

    assert result.status is TraceComparisonStatus.FAILED
    assert result.reason_code == "trace_comparison_evidence_incomplete"
    assert result.reward_eligible is False
    assert comparator.calls == []


def test_pinned_adapter_rejects_tampered_cas_input(tmp_path: Path) -> None:
    comparator = _Comparator()
    request = _request(tmp_path)
    artifact = request.terminal.artifacts[-1]
    path = request.terminal.cas_root / artifact.receipt_relative_path
    path.write_text("tampered", encoding="utf-8")

    with pytest.raises(IntegrityError, match="failed verification"):
        _adapter(tmp_path / "dependency", comparator).compare(request)
