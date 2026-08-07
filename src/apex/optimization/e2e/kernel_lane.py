"""Dynamic kernel-only opportunity extraction and source routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apex.core import ContractError
from apex.diagnostics import OpportunityRankings, TraceEvidence, rank_evidence

from .oracles import CorrectnessOracleRegistry, ResolvedCorrectnessOracle


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
    correctness_oracle_sha256: str | None = None

    @property
    def eligible(self) -> bool:
        return self.eligibility == "eligible"


@dataclass(frozen=True, slots=True)
class KernelOpportunityPlan:
    opportunities: tuple[KernelOpportunity, ...]
    measured_order: tuple[str, ...]
    recoverable_order: tuple[str, ...]
    correctness_oracle_policy_sha256: str | None = None

    @property
    def eligible(self) -> tuple[KernelOpportunity, ...]:
        return tuple(item for item in self.opportunities if item.eligible)


def build_kernel_opportunity_plan(
    evidence: tuple[TraceEvidence, ...],
    *,
    max_kernels: int,
    min_gpu_pct: float = 0.1,
    correctness_oracles: CorrectnessOracleRegistry | None = None,
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
        command = record.kernel.test_command
        oracle = _resolve_oracle(
            correctness_oracles,
            record=record,
            source=source,
            root=root,
            observed_test=test,
            observed_command=command,
        )
        if oracle is not None:
            test = oracle.test_file
            command = oracle.test_command
        reason = _eligibility_reason(record, source, root, test, command)
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
                test_command=command,
                eligibility="eligible" if reason == "eligible" else "unresolved",
                reason_code=reason,
                correctness_oracle_sha256=(
                    oracle.binding_sha256 if oracle is not None else None
                ),
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
        correctness_oracle_policy_sha256=(
            correctness_oracles.policy_sha256 if correctness_oracles is not None else None
        ),
    )


def _eligibility_reason(
    record: TraceEvidence,
    source: Path | None,
    root: Path | None,
    test: Path | None,
    command: str | None,
) -> str:
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
    if not command and not (test and test.is_file()):
        return "correctness_oracle_unresolved"
    return "eligible"


def _resolve_oracle(
    registry: CorrectnessOracleRegistry | None,
    *,
    record: TraceEvidence,
    source: Path | None,
    root: Path | None,
    observed_test: Path | None,
    observed_command: str | None,
) -> ResolvedCorrectnessOracle | None:
    if registry is None or source is None or root is None:
        return None
    resolved = registry.resolve(
        repository_id=record.kernel.origin_library,
        source_root=root,
        source_path=source,
    )
    if resolved is None:
        return None
    if (
        observed_test is not None
        and observed_test.resolve() != resolved.test_file.resolve()
    ):
        raise ContractError(
            "Observed correctness test conflicts with reviewed oracle",
            "correctness_oracle_conflict",
        )
    if observed_command is not None and observed_command != resolved.test_command:
        raise ContractError(
            "Observed correctness command conflicts with reviewed oracle",
            "correctness_oracle_conflict",
        )
    return resolved


__all__ = [
    "KernelOpportunity",
    "KernelOpportunityPlan",
    "build_kernel_opportunity_plan",
]
