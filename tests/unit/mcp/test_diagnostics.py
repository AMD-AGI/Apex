from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_file
from apex.mcp import (
    CapabilityRegistry,
    CapabilityScope,
    HotspotRankHandler,
    TraceAnalyzeHandler,
    TraceCompareHandler,
    hotspot_rank_descriptor,
    trace_analyze_descriptor,
    trace_compare_descriptor,
)
from apex.ports import (
    CapabilityAuthority,
    CapabilityRequest,
    DiagnosticsRequest,
    DiagnosticsResult,
    TraceComparisonRequest,
    TraceComparisonResult,
    TraceComparisonStatus,
)


class _Diagnostics:
    def __init__(self) -> None:
        self.request: DiagnosticsRequest | None = None

    def analyze(self, request: DiagnosticsRequest) -> DiagnosticsResult:
        self.request = request
        request.output_dir.mkdir(parents=True)
        evidence = request.output_dir / "trace_evidence.json"
        evidence.write_text('{"records": []}\n', encoding="utf-8")
        report = request.benchmark_workspace / "benchmark_report.json"
        return DiagnosticsResult(
            run_id=request.run_id,
            succeeded=True,
            artifacts=(report, evidence),
            summary={"record_count": 0},
            artifact_roles={
                str(report.resolve()): "diagnostic_benchmark_report",
                str(evidence.resolve()): "diagnostic_trace_evidence",
            },
            benchmark_workspace=request.benchmark_workspace,
        )


class _Comparison:
    def __init__(self) -> None:
        self.request: TraceComparisonRequest | None = None

    def compare(self, request: TraceComparisonRequest) -> TraceComparisonResult:
        self.request = request
        request.output_dir.mkdir(parents=True)
        artifact = request.output_dir / "comparison.csv"
        artifact.write_text("metric,delta\nlatency,-1\n", encoding="utf-8")
        return TraceComparisonResult(
            TraceComparisonStatus.PARTIAL,
            "comparison_complete_attribution_unavailable",
            {
                "tracelens": {
                    "root": "/private/dependency/path",
                    "commit": "a" * 40,
                },
                "claims": {"reward_emitted": False},
            },
            artifacts=(artifact,),
            artifact_roles={
                str(artifact.resolve()): "tracelens_perf_report_comparison_csv"
            },
            output_root=request.output_dir.resolve(),
        )


def _registry(workspace: Path, results: Path, diagnostics: _Diagnostics):
    registry = CapabilityRegistry()
    registry.register(
        trace_analyze_descriptor(),
        TraceAnalyzeHandler(CapabilityScope(workspace, results), diagnostics),
    )
    return registry


def test_trace_analysis_is_scoped_receipted_and_reward_ineligible(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    benchmark = workspace / "diagnostic"
    benchmark.mkdir(parents=True)
    report = benchmark / "benchmark_report.json"
    report.write_text('{"schema": "fixture"}\n', encoding="utf-8")
    results = tmp_path / "results"
    diagnostics = _Diagnostics()
    registry = _registry(workspace, results, diagnostics)

    result = registry.invoke(
        CapabilityRequest(
            "trace.analyze",
            {"run_id": "run-1", "benchmark_workspace": "diagnostic"},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    assert result.reward_eligible is False
    assert result.content["reward_eligible"] is False
    assert diagnostics.request is not None
    assert diagnostics.request.provenance_hash == sha256_file(report)
    assert diagnostics.request.output_dir == results / "trace-analysis" / "run-1"
    receipts = {item["role"]: item for item in result.artifact_receipts}
    assert receipts["diagnostic_benchmark_report"]["scope"] == "workspace"
    assert receipts["diagnostic_benchmark_report"]["path"] == (
        "diagnostic/benchmark_report.json"
    )
    assert receipts["diagnostic_trace_evidence"]["scope"] == "results"
    assert receipts["diagnostic_trace_evidence"]["path"] == (
        "trace-analysis/run-1/trace_evidence.json"
    )


def test_trace_analysis_rejects_path_escape_and_output_reuse(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    diagnostics = _Diagnostics()
    registry = _registry(workspace, tmp_path / "results", diagnostics)
    authority = frozenset({CapabilityAuthority.WORKSPACE_USER})

    with pytest.raises(ContractError) as escaped:
        registry.invoke(
            CapabilityRequest(
                "trace.analyze",
                {"run_id": "run-1", "benchmark_workspace": "../outside"},
                authority,
            )
        )
    assert escaped.value.reason_code == "unsafe_capability_path"

    benchmark = workspace / "diagnostic"
    benchmark.mkdir()
    (benchmark / "benchmark_report.json").write_text("{}\n", encoding="utf-8")
    request = CapabilityRequest(
        "trace.analyze",
        {"run_id": "run-1", "benchmark_workspace": "diagnostic"},
        authority,
    )
    registry.invoke(request)
    with pytest.raises(ContractError) as reused:
        registry.invoke(request)
    assert reused.value.reason_code == "capability_output_exists"


def test_capability_scope_rejects_symlink_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "results-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ContractError) as error:
        CapabilityScope(workspace, link)
    assert error.value.reason_code == "unsafe_capability_path"


def test_hotspot_rank_projects_digest_bound_existing_rankings(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    path = results / "trace-analysis" / "run-1" / "trace_evidence.json"
    workspace.mkdir()
    path.parent.mkdir(parents=True)
    document = {
        "schema_version": 1,
        "run_id": "run-1",
        "artifact_receipts": [],
        "records": [],
        "rankings": {
            "measured": [{"candidate_id": "measured-1"}, {"candidate_id": "measured-2"}],
            "recoverable": [{"candidate_id": "modeled-1"}],
        },
    }
    path.write_bytes(canonical_json_bytes(document) + b"\n")
    registry = CapabilityRegistry()
    registry.register(
        hotspot_rank_descriptor(),
        HotspotRankHandler(CapabilityScope(workspace, results)),
    )
    authority = frozenset({CapabilityAuthority.WORKSPACE_USER})

    result = registry.invoke(
        CapabilityRequest(
            "hotspot.rank",
            {
                "trace_evidence_path": "trace-analysis/run-1/trace_evidence.json",
                "expected_sha256": sha256_file(path),
                "limit": 1,
            },
            authority,
        )
    )

    assert result.content["measured"] == [{"candidate_id": "measured-1"}]
    assert result.content["recoverable"] == [{"candidate_id": "modeled-1"}]
    assert result.content["source_receipt"]["role"] == "diagnostic_trace_evidence"
    assert result.content["reward_eligible"] is False

    with pytest.raises(IntegrityError) as drift:
        registry.invoke(
            CapabilityRequest(
                "hotspot.rank",
                {
                    "trace_evidence_path": "trace-analysis/run-1/trace_evidence.json",
                    "expected_sha256": "0" * 64,
                },
                authority,
            )
        )
    assert drift.value.reason_code == "diagnostic_artifact_drift"


def test_trace_compare_is_scoped_portable_and_reward_ineligible(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    baseline = results / "baseline-cas"
    terminal = results / "terminal-cas"
    workspace.mkdir()
    baseline.mkdir(parents=True)
    terminal.mkdir()
    comparison = _Comparison()
    registry = CapabilityRegistry()
    registry.register(
        trace_compare_descriptor(),
        TraceCompareHandler(
            CapabilityScope(workspace, results), lambda: comparison
        ),
    )
    evidence = {"trace_evidence_sha256": "b" * 64, "artifacts": []}

    result = registry.invoke(
        CapabilityRequest(
            "trace.compare",
            {
                "run_id": "compare-1",
                "gpu_arch": "gfx950",
                "baseline_cas_root": "baseline-cas",
                "terminal_cas_root": "terminal-cas",
                "baseline": evidence,
                "terminal": evidence,
                "terminal_benchmark_sha256": "c" * 64,
            },
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )

    assert comparison.request is not None
    assert comparison.request.baseline.cas_root == baseline.resolve()
    assert comparison.request.terminal.cas_root == terminal.resolve()
    assert result.content["status"] == "partial"
    assert result.content["reward_eligible"] is False
    assert result.content["summary"]["tracelens"] == {"commit": "a" * 40}
    assert result.artifact_receipts == (
        {
            "role": "tracelens_perf_report_comparison_csv",
            "scope": "results",
            "path": "trace-comparison/compare-1/comparison.csv",
            "sha256": sha256_file(
                results / "trace-comparison" / "compare-1" / "comparison.csv"
            ),
            "byte_count": 24,
        },
    )


def test_trace_compare_rejects_cas_escape_before_dependency_probe(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    workspace.mkdir()
    results.mkdir()
    calls = 0

    def comparator():
        nonlocal calls
        calls += 1
        return _Comparison()

    registry = CapabilityRegistry()
    registry.register(
        trace_compare_descriptor(),
        TraceCompareHandler(CapabilityScope(workspace, results), comparator),
    )
    evidence = {"trace_evidence_sha256": "b" * 64, "artifacts": []}

    with pytest.raises(ContractError) as error:
        registry.invoke(
            CapabilityRequest(
                "trace.compare",
                {
                    "run_id": "compare-1",
                    "gpu_arch": "gfx950",
                    "baseline_cas_root": "../outside",
                    "terminal_cas_root": "terminal",
                    "baseline": evidence,
                    "terminal": evidence,
                    "terminal_benchmark_sha256": "c" * 64,
                },
                frozenset({CapabilityAuthority.WORKSPACE_USER}),
            )
        )

    assert error.value.reason_code == "unsafe_capability_path"
    assert calls == 0
