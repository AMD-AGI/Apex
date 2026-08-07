"""Machine-readable terminal result for an E2E optimization run."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from apex.benchmark import BenchmarkConfigViews
from apex.core import TaskStatus, ValidationLevel, canonical_json_bytes
from apex.evaluation import E2EMeasurement
from apex.runtime import RunProvenance

from .benchmarking import measurement_metrics
from .kernel_lane import KernelOpportunityPlan
from .run_record import E2ERunRecord


@dataclass(frozen=True, slots=True)
class E2EOptimizationResult:
    schema_version: int
    run_id: str
    status: TaskStatus
    reason_code: str
    validation_level: ValidationLevel
    intake_provenance_status: str
    intake_missing_evidence: tuple[str, ...]
    formal_delivery_verified: bool
    provenance_hash: str
    baseline_metrics: Mapping[str, float]
    final_metrics: Mapping[str, float]
    accepted_patch_ids: tuple[str, ...]
    opportunity_count: int
    eligible_opportunity_count: int
    event_journal: str
    artifact_store: str
    benchmark_original: str
    benchmark_measurement: str
    benchmark_diagnostic: str
    benchmark_replay: str
    diagnostic_evidence: str | None
    no_regression: bool | None
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["validation_level"] = self.validation_level.value
        value["intake_missing_evidence"] = list(self.intake_missing_evidence)
        value["accepted_patch_ids"] = list(self.accepted_patch_ids)
        return value


def build_e2e_result(
    *,
    record: E2ERunRecord,
    views: BenchmarkConfigViews,
    provenance: RunProvenance,
    status: TaskStatus,
    reason: str,
    validation_level: ValidationLevel,
    baseline: E2EMeasurement | None,
    final: E2EMeasurement | None,
    plan: KernelOpportunityPlan | None,
    evidence_path: str | None,
    no_regression: bool | None,
    details: Mapping[str, Any],
) -> E2EOptimizationResult:
    """Assemble the sole machine-readable terminal result shape."""

    return E2EOptimizationResult(
        schema_version=1,
        run_id=record.run_id,
        status=status,
        reason_code=reason,
        validation_level=validation_level,
        intake_provenance_status=provenance.status,
        intake_missing_evidence=provenance.missing_evidence,
        formal_delivery_verified=(
            status is TaskStatus.SUCCEEDED
            and validation_level is ValidationLevel.SOURCE_REBUILD_VERIFIED
        ),
        provenance_hash=provenance.digest,
        baseline_metrics=measurement_metrics(baseline) if baseline else {},
        final_metrics=measurement_metrics(final) if final else {},
        accepted_patch_ids=record.controller.state.accepted_patch_ids,
        opportunity_count=len(plan.opportunities) if plan else 0,
        eligible_opportunity_count=len(plan.eligible) if plan else 0,
        event_journal=str(record.root / "events" / "run.db"),
        artifact_store=str(record.root / "artifacts"),
        benchmark_original=str(views.original),
        benchmark_measurement=str(views.measurement),
        benchmark_diagnostic=str(views.diagnostic),
        benchmark_replay=str(views.replay),
        diagnostic_evidence=evidence_path,
        no_regression=no_regression,
        details=dict(details),
    )


def write_e2e_result(result: E2EOptimizationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(result.to_dict()) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["E2EOptimizationResult", "build_e2e_result", "write_e2e_result"]
