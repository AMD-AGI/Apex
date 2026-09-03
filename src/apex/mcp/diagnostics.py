"""Reward-ineligible TraceLens evidence normalization capability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityKind,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRewardRole,
    CapabilitySideEffect,
    DiagnosticsPort,
    DiagnosticsRequest,
    TraceComparisonPort,
    TraceComparisonRequest,
    TraceDiagnosticEvidence,
)

from .scope import CapabilityScope


_ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "minLength": 1},
        "scope": {"enum": ["workspace", "results"]},
        "path": {"type": "string", "minLength": 1},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "byte_count": {"type": "integer", "minimum": 0},
    },
    "required": ["role", "scope", "path", "sha256", "byte_count"],
    "additionalProperties": False,
}

_TRACE_INPUT_ARTIFACT_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "minLength": 1},
        "logical_path": {"type": "string", "minLength": 1},
        "receipt": {
            "type": "object",
            "properties": {
                "digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "size": {"type": "integer", "minimum": 0},
                "media_type": {"type": "string", "minLength": 1},
                "relative_path": {"type": "string", "minLength": 1},
            },
            "required": ["digest", "size", "media_type", "relative_path"],
            "additionalProperties": False,
        },
    },
    "required": ["role", "logical_path", "receipt"],
    "additionalProperties": False,
}

_TRACE_INPUT_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "trace_evidence_sha256": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "artifacts": {"type": "array", "items": _TRACE_INPUT_ARTIFACT_SCHEMA},
    },
    "required": ["trace_evidence_sha256", "artifacts"],
    "additionalProperties": False,
}


def trace_analyze_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="trace.analyze",
        title="Analyze an existing Magpie trace",
        summary=(
            "Validate and normalize profiler artifacts already produced by Magpie; "
            "the output is diagnostic evidence and never a performance grade."
        ),
        kind=CapabilityKind.TOOL,
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"},
                "benchmark_workspace": {"type": "string", "minLength": 1},
                "preserve_raw_trace": {"type": "boolean"},
            },
            "required": ["run_id", "benchmark_workspace"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "succeeded": {"type": "boolean"},
                "summary": {"type": "object"},
                "error": {"type": ["string", "null"]},
                "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA},
                "reward_eligible": {"const": False},
            },
            "required": ["succeeded", "summary", "error", "artifacts", "reward_eligible"],
            "additionalProperties": False,
        },
        side_effects=(
            CapabilitySideEffect.READ_WORKSPACE,
            CapabilitySideEffect.WRITE_RESULTS,
        ),
        required_authority=CapabilityAuthority.WORKSPACE_USER,
        gpu_requirement=CapabilityGpuRequirement.NONE,
        timeout_seconds=60,
        artifact_classes=("diagnostic_trace_evidence",),
        reward_role=CapabilityRewardRole.INELIGIBLE,
    )


def hotspot_rank_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="hotspot.rank",
        title="Read ranked kernel hotspots",
        summary=(
            "Verify one normalized trace-evidence artifact and project its existing "
            "measured and recoverable rankings without rerunning or regrading it."
        ),
        kind=CapabilityKind.TOOL,
        input_schema={
            "type": "object",
            "properties": {
                "trace_evidence_path": {"type": "string", "minLength": 1},
                "expected_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["trace_evidence_path", "expected_sha256"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "trace_run_id": {"type": "string"},
                "measured": {"type": "array", "items": {"type": "object"}},
                "recoverable": {"type": "array", "items": {"type": "object"}},
                "source_receipt": _ARTIFACT_SCHEMA,
                "reward_eligible": {"const": False},
            },
            "required": [
                "trace_run_id",
                "measured",
                "recoverable",
                "source_receipt",
                "reward_eligible",
            ],
            "additionalProperties": False,
        },
        side_effects=(CapabilitySideEffect.READ_RESULTS,),
        required_authority=CapabilityAuthority.WORKSPACE_USER,
        gpu_requirement=CapabilityGpuRequirement.NONE,
        timeout_seconds=5,
        artifact_classes=("diagnostic_trace_evidence",),
        reward_role=CapabilityRewardRole.INELIGIBLE,
    )


def trace_compare_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id="trace.compare",
        title="Compare two diagnostic traces",
        summary=(
            "Compare two receipt-bound TraceLens report sets using the pinned "
            "public comparison API; outputs are diagnostic and reward-ineligible."
        ),
        kind=CapabilityKind.TOOL,
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                },
                "gpu_arch": {"type": "string", "minLength": 1},
                "baseline_cas_root": {"type": "string", "minLength": 1},
                "terminal_cas_root": {"type": "string", "minLength": 1},
                "baseline": _TRACE_INPUT_EVIDENCE_SCHEMA,
                "terminal": _TRACE_INPUT_EVIDENCE_SCHEMA,
                "terminal_benchmark_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
            },
            "required": [
                "run_id",
                "gpu_arch",
                "baseline_cas_root",
                "terminal_cas_root",
                "baseline",
                "terminal",
                "terminal_benchmark_sha256",
            ],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {
                    "enum": ["succeeded", "partial", "unavailable", "failed"]
                },
                "reason_code": {"type": "string", "minLength": 1},
                "summary": {"type": "object"},
                "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA},
                "reward_eligible": {"const": False},
            },
            "required": [
                "status",
                "reason_code",
                "summary",
                "artifacts",
                "reward_eligible",
            ],
            "additionalProperties": False,
        },
        side_effects=(
            CapabilitySideEffect.READ_RESULTS,
            CapabilitySideEffect.WRITE_RESULTS,
        ),
        required_authority=CapabilityAuthority.WORKSPACE_USER,
        gpu_requirement=CapabilityGpuRequirement.NONE,
        timeout_seconds=60,
        artifact_classes=("diagnostic_analysis",),
        reward_role=CapabilityRewardRole.INELIGIBLE,
    )


class TraceAnalyzeHandler:
    def __init__(self, scope: CapabilityScope, diagnostics: DiagnosticsPort) -> None:
        self._scope = scope
        self._diagnostics = diagnostics

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        run_id = str(request.arguments["run_id"])
        workspace = self._scope.read_workspace(
            str(request.arguments["benchmark_workspace"])
        )
        output = self._scope.claim_output("trace-analysis", run_id)
        result = self._diagnostics.analyze(
            DiagnosticsRequest(
                run_id=run_id,
                benchmark_workspace=workspace,
                output_dir=output,
                provenance_hash=_diagnostic_provenance(workspace),
                preserve_raw_trace=bool(
                    request.arguments.get("preserve_raw_trace", False)
                ),
            )
        )
        receipts = tuple(
            _artifact_receipt(self._scope, path, result.artifact_roles)
            for path in result.artifacts
        )
        return CapabilityResult(
            request.capability_id,
            {
                "succeeded": result.succeeded,
                "summary": dict(result.summary),
                "error": result.error,
                "artifacts": list(receipts),
                "reward_eligible": False,
            },
            artifact_receipts=receipts,
            reward_eligible=False,
        )


class HotspotRankHandler:
    def __init__(self, scope: CapabilityScope) -> None:
        self._scope = scope

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        path = self._scope.read_results(
            str(request.arguments["trace_evidence_path"])
        )
        expected = str(request.arguments["expected_sha256"])
        if sha256_file(path) != expected:
            raise IntegrityError(
                "Trace evidence digest differs", "diagnostic_artifact_drift"
            )
        document = _read_trace_evidence(path)
        limit = request.arguments.get("limit", 20)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ContractError("Hotspot limit is invalid", "invalid_capability_arguments")
        rankings = document["rankings"]
        receipt = _artifact_receipt(
            self._scope,
            path,
            {str(path.resolve()): "diagnostic_trace_evidence"},
        )
        return CapabilityResult(
            request.capability_id,
            {
                "trace_run_id": str(document["run_id"]),
                "measured": list(rankings["measured"][:limit]),
                "recoverable": list(rankings["recoverable"][:limit]),
                "source_receipt": receipt,
                "reward_eligible": False,
            },
            artifact_receipts=(receipt,),
        )


class TraceCompareHandler:
    def __init__(
        self,
        scope: CapabilityScope,
        comparator: Callable[[], TraceComparisonPort],
    ) -> None:
        self._scope = scope
        self._comparator = comparator

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        arguments = request.arguments
        baseline_root = self._scope.read_results(str(arguments["baseline_cas_root"]))
        terminal_root = self._scope.read_results(str(arguments["terminal_cas_root"]))
        baseline = _trace_input(arguments["baseline"], baseline_root)
        terminal = _trace_input(arguments["terminal"], terminal_root)
        output = self._scope.claim_output(
            "trace-comparison", str(arguments["run_id"])
        )
        result = self._comparator().compare(
            TraceComparisonRequest(
                run_id=str(arguments["run_id"]),
                gpu_arch=str(arguments["gpu_arch"]),
                baseline=baseline,
                terminal=terminal,
                terminal_benchmark_sha256=str(
                    arguments["terminal_benchmark_sha256"]
                ),
                output_dir=output,
            )
        )
        receipts = tuple(
            _artifact_receipt(self._scope, path, result.artifact_roles)
            for path in result.artifacts
        )
        return CapabilityResult(
            request.capability_id,
            {
                "status": result.status.value,
                "reason_code": result.reason_code,
                "summary": _portable_comparison_summary(result.summary),
                "artifacts": list(receipts),
                "reward_eligible": False,
            },
            artifact_receipts=receipts,
        )


def _trace_input(value: object, cas_root: Path) -> TraceDiagnosticEvidence:
    if not isinstance(value, Mapping):
        raise ContractError(
            "Trace comparison evidence is invalid", "invalid_capability_arguments"
        )
    return TraceDiagnosticEvidence.from_mapping(value, cas_root=cas_root)


def _portable_comparison_summary(value: Mapping[str, object]) -> dict[str, object]:
    summary = dict(value)
    dependency = summary.get("tracelens")
    if isinstance(dependency, Mapping):
        summary["tracelens"] = {
            key: item for key, item in dependency.items() if key != "root"
        }
    return summary


def _diagnostic_provenance(workspace: Path) -> str:
    reports = tuple(sorted(workspace.rglob("benchmark_report.json")))
    if len(reports) == 1 and reports[0].is_file() and not reports[0].is_symlink():
        return sha256_file(reports[0])
    return sha256_json(
        {
            "status": "benchmark_report_unresolved",
            "report_count": len(reports),
        }
    )


def _read_trace_evidence(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            "Trace evidence cannot be decoded", "invalid_diagnostic_evidence"
        ) from error
    if payload != canonical_json_bytes(document) + b"\n" or not isinstance(
        document, Mapping
    ):
        raise IntegrityError(
            "Trace evidence is not canonical", "invalid_diagnostic_evidence"
        )
    rankings = document.get("rankings")
    if (
        document.get("schema_version") != 1
        or not str(document.get("run_id", "")).strip()
        or not isinstance(rankings, Mapping)
        or any(
            not isinstance(rankings.get(name), list)
            or any(not isinstance(item, Mapping) for item in rankings[name])
            for name in ("measured", "recoverable")
        )
    ):
        raise IntegrityError(
            "Trace rankings are malformed", "invalid_diagnostic_evidence"
        )
    return document


def _artifact_receipt(
    scope: CapabilityScope,
    path: Path,
    roles: Mapping[str, str],
) -> Mapping[str, object]:
    label, relative = scope.locator(path)
    return {
        "role": roles.get(str(path.resolve()), "diagnostic_artifact"),
        "scope": label,
        "path": relative,
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
    }


__all__ = [
    "HotspotRankHandler",
    "TraceCompareHandler",
    "TraceAnalyzeHandler",
    "hotspot_rank_descriptor",
    "trace_compare_descriptor",
    "trace_analyze_descriptor",
]
