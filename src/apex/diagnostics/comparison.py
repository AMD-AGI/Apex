"""Pinned TraceLens report comparison over receipt-verified diagnostic inputs."""

from __future__ import annotations

import importlib.util
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable, Mapping

from apex.core import ContractError, IntegrityError, sha256_file, sha256_json
from apex.ports import (
    TraceComparisonArtifact,
    TraceComparisonRequest,
    TraceComparisonResult,
    TraceComparisonStatus,
    TraceDiagnosticEvidence,
)


_COMMIT_LENGTH = 40
_REPORT_API = "TraceLens/Reporting/compare_perf_reports_pytorch.py"
_ROOFLINE_SHEETS = {
    "BinaryElementwise.csv",
    "CONV_bwd.csv",
    "CONV_fwd.csv",
    "GEMM.csv",
    "SDPA_bwd.csv",
    "SDPA_fwd.csv",
    "UnaryElementwise.csv",
}
_OPS_SHEETS = {
    "ops_all.csv",
    "ops_unique_args.csv",
    "unified_perf_summary.csv",
}
_MAX_OUTPUT_FILES = 256
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
ReportComparator = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class _BoundArtifact:
    artifact: TraceComparisonArtifact
    path: Path


@dataclass(frozen=True, slots=True)
class _ValidatedDiagnostic:
    raw_trace: _BoundArtifact
    benchmark_report: _BoundArtifact
    report_groups: Mapping[str, Mapping[str, _BoundArtifact]]


@dataclass(frozen=True, slots=True)
class _ComparableGroup:
    name: str
    baseline: Mapping[str, _BoundArtifact]
    terminal: Mapping[str, _BoundArtifact]
    sheets: tuple[str, ...]


class PinnedTraceLensComparisonAdapter:
    """Run the pinned report-diff API without claiming full LCA attribution."""

    adapter_id = "pinned_tracelens_terminal_comparison_v1"

    def __init__(
        self,
        *,
        root: Path,
        commit: str,
        report_comparator: ReportComparator | None = None,
    ) -> None:
        resolved = root.expanduser().resolve()
        if (
            not resolved.is_dir()
            or len(commit) != _COMMIT_LENGTH
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ContractError(
                "Pinned TraceLens dependency is invalid",
                "dependency_receipt_invalid",
            )
        self._root = resolved
        self._commit = commit
        self._api = resolved / _REPORT_API
        self._api_sha256 = sha256_file(self._api) if self._api.is_file() else None
        self._injected_comparator = report_comparator

    def compare(self, request: TraceComparisonRequest) -> TraceComparisonResult:
        common = self._summary(request)
        if self._api_sha256 is None:
            return _result(
                TraceComparisonStatus.UNAVAILABLE,
                "tracelens_perf_report_comparison_api_unavailable",
                common,
            )
        if sha256_file(self._api) != self._api_sha256:
            raise IntegrityError(
                "Pinned TraceLens comparison API changed after composition",
                "tracelens_comparison_api_drift",
            )
        baseline = _validated_diagnostic(request.baseline)
        terminal = _validated_diagnostic(request.terminal)
        if baseline is None or terminal is None:
            return _result(
                TraceComparisonStatus.FAILED,
                "trace_comparison_evidence_incomplete",
                common,
            )
        groups = _comparable_groups(baseline, terminal)
        if not groups:
            return _result(
                TraceComparisonStatus.UNAVAILABLE,
                "tracelens_perf_report_inputs_inapplicable",
                common,
            )
        return self._run_groups(request, groups, common)

    def _run_groups(
        self,
        request: TraceComparisonRequest,
        groups: tuple[_ComparableGroup, ...],
        common: dict[str, object],
    ) -> TraceComparisonResult:
        root = _prepare_output_root(request.output_dir)
        comparator = self._comparator()
        outcomes: list[dict[str, object]] = []
        artifacts: list[Path] = []
        roles: dict[str, str] = {}
        for group in groups:
            try:
                outcome, produced = _compare_group(root, group, comparator)
            except (ImportError, OSError, TypeError, ValueError) as error:
                outcomes.append(
                    {
                        "group": group.name,
                        "status": "failed",
                        "error_type": type(error).__name__,
                    }
                )
                continue
            outcomes.append(outcome)
            for path in produced:
                artifacts.append(path)
                roles[str(path.resolve())] = (
                    "tracelens_perf_report_comparison_workbook"
                    if path.suffix.lower() == ".xlsx"
                    else "tracelens_perf_report_comparison_csv"
                )
        succeeded = sum(item["status"] == "succeeded" for item in outcomes)
        summary = {
            **common,
            "groups": outcomes,
            "claims": {
                "comparison_performed": succeeded > 0,
                "attribution_performed": False,
                "performance_grade_emitted": False,
                "reward_emitted": False,
            },
            "full_attribution": {
                "status": "unavailable",
                "reason_code": "pinned_tracelens_full_attribution_contract_unavailable",
            },
            "outputs": [_output_manifest(path, root) for path in artifacts],
        }
        if succeeded == 0:
            return _result(
                TraceComparisonStatus.FAILED,
                "tracelens_perf_report_comparison_failed",
                summary,
            )
        reason = (
            "tracelens_perf_report_comparison_succeeded_full_attribution_unavailable"
            if succeeded == len(outcomes)
            else "tracelens_perf_report_comparison_partially_succeeded"
        )
        return TraceComparisonResult(
            TraceComparisonStatus.PARTIAL,
            reason,
            summary,
            False,
            tuple(artifacts),
            roles,
            root,
        )

    def _comparator(self) -> ReportComparator:
        if self._injected_comparator is not None:
            return self._injected_comparator
        name = f"_apex_tracelens_compare_{self._commit}"
        spec = importlib.util.spec_from_file_location(name, self._api)
        if spec is None or spec.loader is None:
            raise ImportError("TraceLens report comparison module is unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return _comparison_function(module)

    def _summary(self, request: TraceComparisonRequest) -> dict[str, object]:
        platform = "MI355X" if request.gpu_arch == "gfx950" else request.gpu_arch
        profile = (
            self._root
            / "TraceLens"
            / "Agent"
            / "Analysis"
            / "utils"
            / "arch"
            / f"{platform}.json"
        )
        return {
            "schema": "apex.tracelens-terminal-comparison/v2",
            "adapter_id": self.adapter_id,
            "tracelens": {
                "root": str(self._root),
                "commit": self._commit,
                "report_comparison_api_sha256": self._api_sha256,
            },
            "gpu_arch": request.gpu_arch,
            "platform_candidate": platform,
            "inputs": {
                "baseline": request.baseline.to_dict(),
                "terminal": request.terminal.to_dict(),
                "terminal_benchmark_sha256": request.terminal_benchmark_sha256,
            },
            "capabilities": {
                "documented_report_comparison_api_detected": self._api.is_file(),
                "mi355x_attribution_profile_detected": profile.is_file(),
                "stable_full_attribution_contract_detected": False,
            },
            "claims": {
                "comparison_performed": False,
                "attribution_performed": False,
                "performance_grade_emitted": False,
                "reward_emitted": False,
            },
        }


def _comparison_function(module: ModuleType) -> ReportComparator:
    value = getattr(module, "generate_compare_perf_reports_pytorch", None)
    if not callable(value):
        raise ImportError("TraceLens report comparison function is unavailable")
    # The public TraceLens helper prints progress. This module is loaded under a
    # run-unique private name, so replacing only its global `print` prevents
    # stdio MCP protocol corruption without redirecting process-global stdout.
    module.__dict__["print"] = _discard_dependency_output
    return value


def _discard_dependency_output(*_args: object, **_kwargs: object) -> None:
    return None


def _validated_diagnostic(
    evidence: TraceDiagnosticEvidence,
) -> _ValidatedDiagnostic | None:
    raw: list[_BoundArtifact] = []
    benchmark: list[_BoundArtifact] = []
    groups: dict[str, dict[str, _BoundArtifact]] = {}
    for artifact in evidence.artifacts:
        bound = _BoundArtifact(artifact, _verify_cas_artifact(evidence, artifact))
        logical = PurePosixPath(artifact.logical_path)
        if artifact.role == "diagnostic_raw_trace":
            raw.append(bound)
        elif artifact.role == "diagnostic_benchmark_report":
            benchmark.append(bound)
        elif artifact.role == "diagnostic_tracelens_report":
            if len(logical.parts) < 3 or logical.parts[0] != "reports":
                raise IntegrityError(
                    "TraceLens report has no comparison group",
                    "invalid_trace_comparison_artifact",
                )
            group = "/".join(logical.parts[1:-1])
            groups.setdefault(group, {})[logical.name] = bound
    if len(raw) != 1 or len(benchmark) != 1 or not groups:
        return None
    return _ValidatedDiagnostic(raw[0], benchmark[0], groups)


def _verify_cas_artifact(
    evidence: TraceDiagnosticEvidence, artifact: TraceComparisonArtifact
) -> Path:
    root = evidence.cas_root.resolve()
    expected = root / "sha256" / artifact.digest[:2] / artifact.digest
    supplied = (root / artifact.receipt_relative_path).resolve()
    try:
        metadata = expected.lstat()
    except OSError as error:
        raise IntegrityError(
            "Trace comparison CAS artifact is missing", "artifact_missing"
        ) from error
    if (
        supplied != expected
        or expected.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != artifact.size
        or sha256_file(expected) != artifact.digest
    ):
        raise IntegrityError(
            "Trace comparison CAS artifact failed verification",
            "artifact_digest_mismatch",
        )
    return expected


def _comparable_groups(
    baseline: _ValidatedDiagnostic, terminal: _ValidatedDiagnostic
) -> tuple[_ComparableGroup, ...]:
    result: list[_ComparableGroup] = []
    for name in sorted(set(baseline.report_groups).intersection(terminal.report_groups)):
        left = baseline.report_groups[name]
        right = terminal.report_groups[name]
        common = sorted(set(left).intersection(right))
        sheets = _supported_sheets(common)
        if not sheets:
            continue
        result.append(
            _ComparableGroup(
                name,
                {key: left[key] for key in common},
                {key: right[key] for key in common},
                sheets,
            )
        )
    return tuple(result)


def _supported_sheets(names: list[str]) -> tuple[str, ...]:
    present = set(names)
    sheets: list[str] = []
    if "gpu_timeline.csv" in present:
        sheets.append("gpu_timeline")
    if "ops_summary.csv" in present:
        sheets.append("ops_summary")
    if "kernel_summary.csv" in present:
        sheets.append("kernel_summary")
    # TraceLens exposes the three operation tables as one public `ops_all`
    # selector, then compares only the member tables present in the report.
    if present.intersection(_OPS_SHEETS):
        sheets.append("ops_all")
    # The public `roofline` selector has the same grouped-table semantics.
    if present.intersection(_ROOFLINE_SHEETS):
        sheets.append("roofline")
    return tuple(sheets)


def _prepare_output_root(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise IntegrityError(
            "Trace comparison output already exists", "stale_trace_comparison_output"
        )
    path.mkdir(parents=True, mode=0o700)
    return path.resolve(strict=True)


def _compare_group(
    root: Path, group: _ComparableGroup, comparator: ReportComparator
) -> tuple[dict[str, object], tuple[Path, ...]]:
    identity = sha256_json({"group": group.name})[:16]
    group_root = root / f"group-{identity}"
    baseline = group_root / "inputs" / "baseline"
    terminal = group_root / "inputs" / "terminal"
    _materialize_report(baseline, group.baseline)
    _materialize_report(terminal, group.terminal)
    output_csvs = group_root / "outputs" / "csv"
    workbook = group_root / "outputs" / "comparison.xlsx"
    result = comparator(
        reports=[str(baseline), str(terminal)],
        output=str(workbook),
        names=["baseline", "terminal"],
        sheets=list(group.sheets),
        output_csvs_dir=str(output_csvs),
    )
    if not isinstance(result, Mapping):
        raise TypeError("TraceLens comparison returned a non-mapping result")
    produced = _validated_outputs(group_root / "outputs")
    return (
        {
            "group": group.name,
            "status": "succeeded",
            "sheets": list(group.sheets),
            "input_files": sorted(group.baseline),
            "output_files": [path.relative_to(root).as_posix() for path in produced],
        },
        produced,
    )


def _materialize_report(
    destination: Path, artifacts: Mapping[str, _BoundArtifact]
) -> None:
    destination.mkdir(parents=True, mode=0o700)
    for name, bound in artifacts.items():
        target = destination / name
        shutil.copyfile(bound.path, target)
        target.chmod(0o400)
        if sha256_file(target) != bound.artifact.digest:
            raise IntegrityError(
                "Materialized TraceLens input changed bytes",
                "artifact_digest_mismatch",
            )


def _validated_outputs(root: Path) -> tuple[Path, ...]:
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    if (
        not files
        or len(files) > _MAX_OUTPUT_FILES
        or any(path.is_symlink() for path in files)
        or sum(path.stat().st_size for path in files) > _MAX_OUTPUT_BYTES
        or not any(path.suffix.lower() == ".csv" for path in files)
    ):
        raise ValueError("TraceLens comparison output set is invalid")
    return files


def _output_manifest(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "media_type": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if path.suffix.lower() == ".xlsx"
            else "text/csv"
        ),
    }


def _result(
    status: TraceComparisonStatus,
    reason: str,
    summary: Mapping[str, object],
) -> TraceComparisonResult:
    return TraceComparisonResult(status, reason, summary)


__all__ = ["PinnedTraceLensComparisonAdapter"]
