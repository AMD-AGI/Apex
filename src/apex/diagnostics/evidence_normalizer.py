"""Exact-symbol normalization from validated Magpie diagnostic artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError, sha256_file, sha256_json

from .evidence_models import (
    EvidenceArtifacts,
    KernelEvidence,
    KernelVolume,
    OperationEvidence,
    PerformanceModelEvidence,
    ShapeEvidence,
    TraceEvidence,
    derive_candidate_id,
    require_digest,
)
from .evidence_support import (
    EventGroup,
    allocate_aggregate,
    event_group_signature,
    evidence_receipt,
    expand_repo_path,
    language,
    mapping,
    optional_text,
    phase,
    resolve_launch_source,
    shape_from_payload,
)
from .targeted_trace import TargetedTraceValidator
from .targeted_trace_models import (
    AcquisitionCoverage,
    EvidenceArtifactReceipt,
    ValidatedTargetedEvent,
    ValidatedTargetedTrace,
)


class TraceEvidenceNormalizer:
    """Normalize exact-symbol joins from independently validated artifacts."""

    def __init__(self, *, validator: TargetedTraceValidator | None = None) -> None:
        self._validator = validator or TargetedTraceValidator()

    def from_benchmark_report(
        self, report_path: Path, *, provenance_hash: str
    ) -> tuple[TraceEvidence, ...]:
        require_digest(provenance_hash, "provenance_hash")
        report, workspace = _load_report(report_path)
        gap_rows, gap_path = _gap_rows(report, workspace)
        artifacts = [_report_receipt(Path(report_path).resolve(), workspace)]
        if gap_path is not None:
            artifacts.append(
                evidence_receipt("gap_analysis_csv", gap_path, workspace, "text/csv")
            )
        rows_by_name = _index_kernel_rows(report)
        groups, validated = self._validate_targeted(report, workspace)
        all_artifacts = tuple(artifacts) + validated.artifacts
        semantic_warnings = tuple(
            f"semantic_coverage_unresolved:{reason}"
            for reason in validated.semantic_unresolved_reasons
        )
        warnings = _coverage_warnings(
            validated.coverage, validated.warnings + semantic_warnings
        )
        normalized, matched = self._normalize_groups(
            groups,
            rows_by_name,
            gap_rows,
            provenance_hash,
            validated.coverage,
            all_artifacts,
            warnings,
        )
        normalized.extend(
            self._unmatched_rows(
                rows_by_name,
                matched,
                gap_rows,
                provenance_hash,
                validated.coverage,
                all_artifacts,
                warnings,
            )
        )
        return tuple(sorted(normalized, key=lambda item: item.candidate_id))

    def _validate_targeted(
        self, report: Mapping[str, Any], workspace: Path
    ) -> tuple[dict[str, EventGroup], ValidatedTargetedTrace]:
        targeted = report.get("targeted_trace")
        if not isinstance(targeted, Mapping):
            raise IntegrityError(
                "Diagnostic report has no TargetedKernelTrace evidence",
                "missing_targeted_trace",
            )
        groups: dict[str, EventGroup] = {}

        def collect(event: ValidatedTargetedEvent) -> None:
            signature = event_group_signature(event.payload)
            group = groups.get(signature)
            if group is None:
                groups[signature] = EventGroup.from_event(event)
            else:
                group.add(event)

        validated = self._validator.validate(
            targeted, workspace=workspace, on_event=collect
        )
        if not groups or validated.coverage.written == 0:
            raise IntegrityError(
                "TargetedKernelTrace contains no usable events", "empty_targeted_trace"
            )
        return groups, validated

    def _normalize_groups(
        self,
        groups: Mapping[str, EventGroup],
        rows_by_name: Mapping[str, Mapping[str, Any]],
        gap_rows: Mapping[str, Mapping[str, str]],
        provenance_hash: str,
        coverage: AcquisitionCoverage,
        artifacts: tuple[EvidenceArtifactReceipt, ...],
        warnings: tuple[str, ...],
    ) -> tuple[list[TraceEvidence], set[str]]:
        by_symbol: dict[str, list[EventGroup]] = defaultdict(list)
        for group in groups.values():
            by_symbol[group.runtime_symbol].append(group)
        normalized: list[TraceEvidence] = []
        matched: set[str] = set()
        for symbol, symbol_groups in sorted(by_symbol.items()):
            symbol_groups.sort(key=lambda item: item.signature)
            aggregate = rows_by_name.get(symbol) if symbol else None
            source = gap_rows.get(symbol, {}) if symbol else {}
            if aggregate is not None:
                matched.add(symbol)
            normalized.extend(
                self._normalize_symbol(
                    symbol_groups,
                    aggregate,
                    source,
                    provenance_hash,
                    coverage,
                    artifacts,
                    warnings,
                )
            )
        return normalized, matched

    def _normalize_symbol(
        self,
        groups: list[EventGroup],
        aggregate: Mapping[str, Any] | None,
        source: Mapping[str, str],
        provenance_hash: str,
        coverage: AcquisitionCoverage,
        artifacts: tuple[EvidenceArtifactReceipt, ...],
        warnings: tuple[str, ...],
    ) -> list[TraceEvidence]:
        records: list[TraceEvidence] = []
        volumes = allocate_aggregate(aggregate, groups)
        for group, volume in zip(groups, volumes, strict=True):
            local_warnings = list(warnings)
            if len(groups) > 1 and aggregate is not None:
                local_warnings.append(
                    "aggregate_volume_partitioned_by_targeted_samples"
                )
            if aggregate is None:
                local_warnings.append("aggregate_profiler_row_unmatched")
            records.append(
                self._from_targeted_group(
                    group,
                    aggregate,
                    source,
                    volume,
                    provenance_hash,
                    coverage,
                    artifacts,
                    tuple(sorted(set(local_warnings))),
                )
            )
        return records

    @staticmethod
    def _from_targeted_group(
        group: EventGroup,
        aggregate: Mapping[str, Any] | None,
        gap_source: Mapping[str, str],
        volume: KernelVolume,
        provenance_hash: str,
        coverage: AcquisitionCoverage,
        artifacts: tuple[EvidenceArtifactReceipt, ...],
        warnings: tuple[str, ...],
    ) -> TraceEvidence:
        payload = group.payload
        identity = mapping(payload, "identity")
        context = mapping(payload, "context")
        semantics = mapping(payload, "semantics")
        runtime = mapping(payload, "runtime")
        kernel, warnings = _targeted_kernel(
            group, identity, semantics, gap_source, warnings
        )
        shape = shape_from_payload(identity, context, semantics, runtime)
        event_phase = phase(str(context.get("stage", "unknown")))
        rank = int(context.get("rank", 0))
        evidence = _targeted_artifacts(
            group, aggregate, gap_source, coverage, artifacts, warnings
        )
        candidate_id = derive_candidate_id(
            provenance_hash=provenance_hash,
            phase=event_phase,
            rank=rank,
            kernel=kernel,
            shape=shape,
        )
        symbol = group.runtime_symbol or str(identity.get("target_id", "unknown"))
        return TraceEvidence(
            1,
            candidate_id,
            provenance_hash,
            event_phase,
            rank,
            OperationEvidence(
                category=gap_source.get(
                    "category", str(payload.get("kind", "unknown"))
                ),
                name=str(identity.get("target_id", symbol)),
            ),
            kernel,
            shape,
            volume,
            PerformanceModelEvidence(),
            evidence,
            "exact" if aggregate is not None else "unknown",
        )

    @staticmethod
    def _unmatched_rows(
        rows: Mapping[str, Mapping[str, Any]],
        matched: set[str],
        gaps: Mapping[str, Mapping[str, str]],
        provenance_hash: str,
        coverage: AcquisitionCoverage,
        artifacts: tuple[EvidenceArtifactReceipt, ...],
        warnings: tuple[str, ...],
    ) -> list[TraceEvidence]:
        return [
            _from_kernel_row(
                row,
                gaps.get(name, {}),
                provenance_hash,
                artifacts,
                coverage,
                tuple(sorted(set(warnings + ("targeted_record_unmatched",)))),
            )
            for name, row in sorted(rows.items())
            if name not in matched
        ]


def _load_report(report_path: Path) -> tuple[Mapping[str, Any], Path]:
    resolved = Path(report_path).resolve()
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(
            "Cannot read Magpie benchmark report", "invalid_benchmark_report"
        ) from error
    if not isinstance(report, Mapping):
        raise IntegrityError(
            "Magpie benchmark report is not an object", "invalid_benchmark_report"
        )
    if report.get("success") is not True:
        raise IntegrityError(
            "Magpie diagnostic benchmark did not succeed",
            "failed_diagnostic_benchmark",
        )
    return report, resolved.parent.resolve()


def _report_receipt(path: Path, workspace: Path) -> EvidenceArtifactReceipt:
    return evidence_receipt(
        "benchmark_report", path, workspace, "application/json"
    )


def _index_kernel_rows(
    report: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    kernel_rows = report.get("kernel_summary", ())
    if not isinstance(kernel_rows, list):
        raise IntegrityError(
            "kernel_summary is not a list", "invalid_benchmark_report"
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in kernel_rows:
        if not isinstance(raw, Mapping):
            raise IntegrityError(
                "kernel_summary contains a non-object", "invalid_benchmark_report"
            )
        name = str(raw.get("name", "")).strip()
        if not name:
            raise IntegrityError(
                "kernel_summary contains an unnamed row", "invalid_benchmark_report"
            )
        if name in indexed:
            raise IntegrityError(
                f"kernel_summary contains ambiguous duplicate symbol {name!r}",
                "ambiguous_kernel_symbol",
            )
        indexed[name] = raw
    return indexed


def _gap_rows(
    report: Mapping[str, Any], workspace: Path
) -> tuple[dict[str, dict[str, str]], Path | None]:
    gap = report.get("gap_analysis")
    if not isinstance(gap, Mapping) or not gap.get("csv_path"):
        return {}, None
    raw_path = Path(str(gap["csv_path"]))
    path = raw_path if raw_path.is_absolute() else workspace / raw_path
    if not path.is_file():
        return {}, None
    path = path.resolve()
    try:
        path.relative_to(workspace)
    except ValueError as error:
        raise IntegrityError(
            "Gap-analysis artifact escapes benchmark workspace",
            "invalid_artifact_path",
        ) from error
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    mappings = _repository_mappings(raw_lines, workspace)
    lines = [line for line in raw_lines if not line.startswith("#")]
    rows: dict[str, dict[str, str]] = {}
    for raw in csv.DictReader(lines):
        if not raw.get("Name"):
            continue
        row = dict(raw)
        source, source_root = expand_repo_path(row.get("source_file", ""), mappings)
        test_file, _ = expand_repo_path(row.get("test_file", ""), mappings)
        row["_resolved_source_path"] = source or ""
        row["_resolved_source_root"] = source_root or ""
        row["_resolved_test_file"] = test_file or ""
        rows[str(row["Name"])] = row
    return rows, path


def _repository_mappings(
    lines: list[str], workspace: Path
) -> dict[str, Path]:
    base: Path | None = None
    mappings: dict[str, Path] = {}
    for line in lines:
        if line.startswith("# Base directory:"):
            base = Path(line.split(":", 1)[1].strip()).expanduser()
        elif line.startswith("# $") and "=" in line:
            key, raw_value = line[2:].split("=", 1)
            value = Path(raw_value.strip()).expanduser()
            mappings[key.strip()] = (
                value if value.is_absolute() else (base or workspace) / value
            )
    return mappings


def _coverage_warnings(
    coverage: AcquisitionCoverage, warnings: tuple[str, ...]
) -> tuple[str, ...]:
    dropped = tuple(
        f"targeted_drop:{reason}:{count}"
        for reason, count in coverage.dropped_by_reason
        if count
    )
    return tuple(sorted(set(warnings + dropped)))


def _targeted_kernel(
    group: EventGroup,
    identity: Mapping[str, Any],
    semantics: Mapping[str, Any],
    gap_source: Mapping[str, str],
    warnings: tuple[str, ...],
) -> tuple[KernelEvidence, tuple[str, ...]]:
    semantic_source = semantics.get("source")
    semantic_source = semantic_source if isinstance(semantic_source, Mapping) else {}
    launch_path = optional_text(semantic_source.get("path"))
    gap_path = gap_source.get("_resolved_source_path") or gap_source.get("source_file") or None
    source_path = resolve_launch_source(launch_path, gap_path)
    warnings = _validate_source_receipt(
        source_path, optional_text(semantic_source.get("sha256")), warnings
    )
    if launch_path and gap_path and Path(launch_path).name != Path(gap_path).name:
        warnings = tuple(sorted(set(warnings + ("source_evidence_disagreement",))))
    symbol = group.runtime_symbol or str(identity.get("target_id", "unknown"))
    kernel_language = language(
        f"{gap_source.get('kind', '')} {source_path or launch_path or ''}", symbol
    )
    source_line = semantic_source.get("line")
    return KernelEvidence(
        runtime_name=symbol,
        language=kernel_language,
        origin_library=str(
            identity.get("package") or gap_source.get("source_repo") or "unknown"
        ),
        source_path=source_path or launch_path,
        source_line=int(source_line) if source_line is not None else None,
        source_confidence="exact_launch" if launch_path else (
            "active_finder" if gap_path else "unknown"
        ),
        patchable=bool(
            source_path
            and Path(source_path).is_absolute()
            and Path(source_path).is_file()
            and kernel_language in {"python", "triton", "hip"}
        ),
        source_root=gap_source.get("_resolved_source_root") or None,
        test_file=gap_source.get("_resolved_test_file") or gap_source.get("test_file") or None,
        test_command=gap_source.get("test_cmd") or None,
    ), warnings


def _validate_source_receipt(
    source_path: str | None, source_digest: str | None, warnings: tuple[str, ...]
) -> tuple[str, ...]:
    path = Path(source_path) if source_path else None
    if source_digest and path and path.is_absolute() and path.is_file():
        if sha256_file(path) != source_digest:
            raise IntegrityError(
                "Targeted launch source differs from its semantic receipt",
                "targeted_source_digest_mismatch",
            )
    elif source_digest:
        return tuple(sorted(set(warnings + ("source_digest_unresolved",))))
    return warnings


def _targeted_artifacts(
    group: EventGroup,
    aggregate: Mapping[str, Any] | None,
    gap_source: Mapping[str, str],
    coverage: AcquisitionCoverage,
    artifacts: tuple[EvidenceArtifactReceipt, ...],
    warnings: tuple[str, ...],
) -> EvidenceArtifacts:
    row_hash = sha256_json(
        {
            "targeted_event_chain": group.payload_hash_chain,
            "event_count": group.count,
            "kernel_summary": dict(aggregate or {}),
            "gap_row": dict(gap_source),
        }
    )
    return EvidenceArtifacts(
        "TargetedKernelTrace",
        coverage,
        artifacts,
        row_hash,
        tuple(sorted(set(warnings + group.warnings))),
    )


def _from_kernel_row(
    row: Mapping[str, Any],
    source: Mapping[str, str],
    provenance_hash: str,
    artifacts: tuple[EvidenceArtifactReceipt, ...],
    coverage: AcquisitionCoverage,
    warnings: tuple[str, ...],
) -> TraceEvidence:
    name = str(row["name"])
    source_path = source.get("_resolved_source_path") or source.get("source_file") or None
    kernel_language = language(source.get("kind", ""), name)
    kernel = _aggregate_kernel(name, source_path, kernel_language, source)
    shape_text = source.get("Input Shapes", "")
    shape = ShapeEvidence(concrete_inputs=(shape_text,) if shape_text else ())
    evidence = EvidenceArtifacts(
        "torch_profiler_summary",
        coverage,
        artifacts,
        sha256_json({"kernel_summary": dict(row), "gap_row": dict(source)}),
        warnings,
    )
    candidate_id = derive_candidate_id(
        provenance_hash=provenance_hash,
        phase="unknown",
        rank=0,
        kernel=kernel,
        shape=shape,
    )
    return TraceEvidence(
        1,
        candidate_id,
        provenance_hash,
        "unknown",
        0,
        OperationEvidence(source.get("category", "unknown"), name),
        kernel,
        shape,
        KernelVolume(
            int(row.get("calls", row.get("Calls", 0)) or 0),
            float(row.get("time_ms", 0) or 0),
            float(row.get("percent", row.get("% Total", 0)) or 0),
        ),
        PerformanceModelEvidence(),
        evidence,
        "probable" if source_path else "unknown",
    )


def _aggregate_kernel(
    name: str,
    source_path: str | None,
    kernel_language: str,
    source: Mapping[str, str],
) -> KernelEvidence:
    return KernelEvidence(
        runtime_name=name,
        language=kernel_language,
        origin_library=source.get("source_repo") or "unknown",
        source_path=source_path,
        source_confidence="active_finder" if source_path else "unknown",
        patchable=bool(
            source_path
            and Path(source_path).is_absolute()
            and Path(source_path).is_file()
            and kernel_language in {"python", "triton", "hip"}
        ),
        source_root=source.get("_resolved_source_root") or None,
        test_file=source.get("_resolved_test_file") or source.get("test_file") or None,
        test_command=source.get("test_cmd") or None,
    )


__all__ = ["TraceEvidenceNormalizer"]
