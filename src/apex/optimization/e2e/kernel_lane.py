"""Dynamic kernel-only opportunity extraction and source routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apex.core import ContractError
from apex.diagnostics import OpportunityRankings, TraceEvidence, rank_evidence


@dataclass(frozen=True, slots=True)
class KernelOpportunity:
    opportunity_id: str
    evidence_id: str
    runtime_name: str
    operation_name: str
    phase: str
    rank: int
    language: str
    origin_library: str
    shape_summary: tuple[str, ...]
    dtypes: tuple[str, ...]
    graph_mode: str
    match_confidence: str
    measured_gpu_pct: float
    roi_prior: float
    source_path: Path | None
    source_root: Path | None
    test_file: Path | None
    test_command: str | None
    eligibility: str
    reason_code: str

    @property
    def eligible(self) -> bool:
        return self.eligibility == "eligible"


@dataclass(frozen=True, slots=True)
class KernelOpportunityPlan:
    opportunities: tuple[KernelOpportunity, ...]
    measured_order: tuple[str, ...]
    recoverable_order: tuple[str, ...]

    @property
    def eligible(self) -> tuple[KernelOpportunity, ...]:
        return tuple(item for item in self.opportunities if item.eligible)


def build_kernel_opportunity_plan(
    evidence: tuple[TraceEvidence, ...],
    *,
    max_kernels: int,
    min_gpu_pct: float = 0.1,
) -> KernelOpportunityPlan:
    """Keep unresolved kernels visible while selecting only source candidates."""

    if max_kernels < 1:
        raise ContractError("max_kernels must be positive", "invalid_kernel_budget")
    rankings = rank_evidence(evidence, min_gpu_pct=min_gpu_pct)
    by_id = {record.candidate_id: record for record in evidence}
    roi = {item.candidate_id: item.roi_prior for item in rankings.recoverable}
    planned: list[KernelOpportunity] = []
    eligible_count = 0
    for ranked in rankings.measured:
        record = by_id[ranked.candidate_id]
        source = Path(record.kernel.source_path) if record.kernel.source_path else None
        root = Path(record.kernel.source_root) if record.kernel.source_root else None
        test = Path(record.kernel.test_file) if record.kernel.test_file else None
        reason = _eligibility_reason(record, source, root)
        planned.append(
            KernelOpportunity(
                opportunity_id=f"kernel-{record.candidate_id[:24]}",
                evidence_id=record.candidate_id,
                runtime_name=record.kernel.runtime_name,
                operation_name=record.op.name,
                phase=record.phase,
                rank=record.rank,
                language=record.kernel.language,
                origin_library=record.kernel.origin_library,
                shape_summary=tuple(record.shape.concrete_inputs),
                dtypes=record.shape.dtypes,
                graph_mode=record.shape.graph_mode,
                match_confidence=record.match_confidence,
                measured_gpu_pct=record.volume.gpu_time_pct,
                roi_prior=roi[record.candidate_id],
                source_path=source,
                source_root=root,
                test_file=test,
                test_command=record.kernel.test_command,
                eligibility="eligible" if reason == "eligible" else "unresolved",
                reason_code=reason,
            )
        )
        if reason == "eligible":
            eligible_count += 1
        if eligible_count >= max_kernels:
            break
    return KernelOpportunityPlan(
        opportunities=tuple(planned),
        measured_order=tuple(item.opportunity_id for item in planned),
        recoverable_order=tuple(
            f"kernel-{item.candidate_id[:24]}"
            for item in rankings.recoverable
            if item.candidate_id in {planned_item.evidence_id for planned_item in planned}
        ),
    )


def _eligibility_reason(record: TraceEvidence, source: Path | None, root: Path | None) -> str:
    if record.kernel.language not in {"python", "triton"}:
        return "unsupported_kernel_language"
    if not record.kernel.patchable or source is None:
        return "source_unresolved"
    if not source.is_absolute() or not source.is_file() or source.is_symlink():
        return "source_not_regular"
    if root is None or not root.is_absolute() or not root.is_dir():
        return "source_root_unresolved"
    try:
        source.resolve().relative_to(root.resolve())
    except ValueError:
        return "source_outside_root"
    if not record.kernel.test_command and not (record.kernel.test_file and Path(record.kernel.test_file).is_file()):
        return "correctness_oracle_unresolved"
    return "eligible"


__all__ = [
    "KernelOpportunity",
    "KernelOpportunityPlan",
    "build_kernel_opportunity_plan",
]
