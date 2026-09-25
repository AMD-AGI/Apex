"""Strict validation of Magpie targeted-trace semantic-quality summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from apex.core import IntegrityError

from .targeted_trace_models import AcquisitionCoverage, ValidatedTargetedEvent


_QUALITY_FIELDS = {
    "evidence_class", "resolution_status", "semantic_coverage_claimed",
    "record_coverage_fraction", "lossless_record_coverage", "records_evaluated",
    "records_with_complete_semantics", "missing_by_field", "cross_event_join",
    "join_eligible_records", "unresolved_reasons",
}
_MISSING_FIELDS = {"phase", "source", "grid", "shape", "correlation"}


@dataclass(slots=True)
class SemanticQualityAccumulator:
    records: int = 0
    complete_records: int = 0
    torch_profiler_records: int = 0
    missing: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _MISSING_FIELDS}
    )

    def observe(self, event: ValidatedTargetedEvent) -> None:
        payload = event.payload
        context = _mapping(payload.get("context"))
        semantics = _mapping(payload.get("semantics"))
        runtime = _mapping(payload.get("runtime"))
        missing = []
        stage = context.get("stage")
        if not isinstance(stage, str) or not stage or stage.lower() == "unknown":
            missing.append("phase")
        if not isinstance(semantics.get("source"), Mapping):
            missing.append("source")
        if runtime.get("grid") is None and semantics.get("python_grid") is None:
            missing.append("grid")
        tensors = semantics.get("tensors")
        if not isinstance(tensors, list) or not tensors:
            missing.append("shape")
        if payload.get("kind") == "torch_profiler_kernel":
            self.torch_profiler_records += 1
            if runtime.get("correlation_id") is None:
                missing.append("correlation")
        self.records += 1
        for name in missing:
            self.missing[name] += 1
        if not missing:
            self.complete_records += 1

    @property
    def join_eligible_records(self) -> int:
        return self.torch_profiler_records - self.missing["correlation"]


def validate_semantic_quality(
    value: object,
    coverage: AcquisitionCoverage,
    observed: SemanticQualityAccumulator,
) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != _QUALITY_FIELDS:
        raise IntegrityError(
            "Targeted semantic-quality shape is invalid", "invalid_targeted_summary"
        )
    missing = _missing_counts(value.get("missing_by_field"), coverage.written)
    reasons = _reasons(value.get("unresolved_reasons"))
    fraction = _finite_fraction(value.get("record_coverage_fraction"))
    expected_fraction = coverage.written / coverage.seen if coverage.seen else 0.0
    lossless = coverage.seen > 0 and coverage.dropped == 0
    complete = _bounded_int(
        value.get("records_with_complete_semantics"),
        "records_with_complete_semantics",
        coverage.written,
    )
    claimed = bool(lossless and coverage.written > 0 and complete == coverage.written)
    expected_reasons = _expected_reasons(coverage, observed.missing)
    conflicts = (
        value.get("evidence_class") != "diagnostic_only"
        or value.get("cross_event_join") != "not_performed"
        or value.get("records_evaluated") != coverage.written
        or observed.records != coverage.written
        or value.get("lossless_record_coverage") is not lossless
        or abs(fraction - expected_fraction) > 1e-12
        or missing != observed.missing
        or complete != observed.complete_records
        or value.get("semantic_coverage_claimed") is not claimed
        or value.get("resolution_status") != ("resolved" if claimed else "unresolved")
        or reasons != expected_reasons
    )
    if conflicts:
        raise IntegrityError(
            "Targeted semantic-quality evidence conflicts with coverage",
            "invalid_targeted_summary",
        )
    join_eligible = _bounded_int(
        value.get("join_eligible_records"), "join_eligible_records", coverage.written
    )
    if join_eligible != observed.join_eligible_records:
        raise IntegrityError(
            "Targeted join eligibility differs from events", "invalid_targeted_summary"
        )
    return claimed, reasons


def _expected_reasons(
    coverage: AcquisitionCoverage, missing: Mapping[str, int]
) -> tuple[str, ...]:
    reasons = []
    if not coverage.seen:
        reasons.append("no_records")
    reasons.extend(
        f"dropped:{reason}" for reason, count in coverage.dropped_by_reason if count
    )
    reasons.extend(
        f"missing:{name}"
        for name in ("phase", "source", "grid", "shape", "correlation")
        if missing[name]
    )
    return tuple(reasons)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _missing_counts(value: object, limit: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _MISSING_FIELDS:
        raise IntegrityError(
            "Targeted missing-field counts are invalid", "invalid_targeted_summary"
        )
    return {
        name: _bounded_int(value[name], f"missing:{name}", limit)
        for name in sorted(_MISSING_FIELDS)
    }


def _reasons(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise IntegrityError(
            "Targeted unresolved reasons are invalid", "invalid_targeted_summary"
        )
    reasons = tuple(value)
    if len(reasons) != len(set(reasons)):
        raise IntegrityError(
            "Targeted unresolved reasons repeat", "invalid_targeted_summary"
        )
    return reasons


def _finite_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntegrityError(
            "Targeted coverage fraction is invalid", "invalid_targeted_summary"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 1:
        raise IntegrityError(
            "Targeted coverage fraction is invalid", "invalid_targeted_summary"
        )
    return result


def _bounded_int(value: object, name: str, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= limit:
        raise IntegrityError(
            f"Targeted {name} is invalid", "invalid_targeted_summary"
        )
    return value


__all__ = ["SemanticQualityAccumulator", "validate_semantic_quality"]
