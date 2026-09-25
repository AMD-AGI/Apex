"""Dynamic kernel-only opportunity extraction and source routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    oracle_execution_mode: str | None = None

    @property
    def eligible(self) -> bool:
        return self.eligibility == "eligible"


@dataclass(frozen=True, slots=True)
class KernelPlanningCoverage:
    policy_id: str
    minimum_selected_gpu_pct: float
    maximum_unclassified_family_gpu_pct: float
    observed_gpu_pct: float
    selected_gpu_pct: float
    unclassified_families: tuple[tuple[str, float], ...]
    semantic_coverage_complete: bool
    semantic_unresolved_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.policy_id != "kernel_family_gpu_time_coverage_v1":
            raise ContractError("Planning coverage policy is invalid", "invalid_diagnosis")
        numeric = (
            self.minimum_selected_gpu_pct,
            self.maximum_unclassified_family_gpu_pct,
            self.observed_gpu_pct,
            self.selected_gpu_pct,
        )
        if any(not math.isfinite(value) or value < 0 or value > 100.000001 for value in numeric):
            raise ContractError("Planning coverage values are invalid", "invalid_diagnosis")
        if self.selected_gpu_pct > self.observed_gpu_pct + 0.000001:
            raise ContractError("Selected GPU time exceeds observed time", "invalid_diagnosis")
        names = tuple(name for name, _ in self.unclassified_families)
        if (
            tuple(sorted(self.unclassified_families)) != self.unclassified_families
            or len(names) != len(set(names))
            or any(not name.strip() or not math.isfinite(share) or share < 0 for name, share in self.unclassified_families)
        ):
            raise ContractError("Planning coverage families are invalid", "invalid_diagnosis")
        if (
            tuple(sorted(set(self.semantic_unresolved_reasons)))
            != self.semantic_unresolved_reasons
            or any(not reason for reason in self.semantic_unresolved_reasons)
            or self.semantic_coverage_complete == bool(self.semantic_unresolved_reasons)
        ):
            raise ContractError("Semantic coverage reasons are invalid", "invalid_diagnosis")

    @property
    def satisfied(self) -> bool:
        largest = max((share for _, share in self.unclassified_families), default=0.0)
        return (
            self.semantic_coverage_complete
            and
            self.selected_gpu_pct >= self.minimum_selected_gpu_pct
            and largest <= self.maximum_unclassified_family_gpu_pct
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "minimum_selected_gpu_pct": self.minimum_selected_gpu_pct,
            "maximum_unclassified_family_gpu_pct": self.maximum_unclassified_family_gpu_pct,
            "observed_gpu_pct": self.observed_gpu_pct,
            "selected_gpu_pct": self.selected_gpu_pct,
            "unclassified_families": [
                {"family": family, "gpu_time_pct": share}
                for family, share in self.unclassified_families
            ],
            "semantic_coverage_complete": self.semantic_coverage_complete,
            "semantic_unresolved_reasons": list(self.semantic_unresolved_reasons),
            "satisfied": self.satisfied,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KernelPlanningCoverage":
        raw = value.get("unclassified_families")
        if not isinstance(raw, list):
            raise ContractError("Planning coverage families are invalid", "invalid_diagnosis")
        try:
            result = cls(
                str(value["policy_id"]),
                float(value["minimum_selected_gpu_pct"]),
                float(value["maximum_unclassified_family_gpu_pct"]),
                float(value["observed_gpu_pct"]),
                float(value["selected_gpu_pct"]),
                tuple(
                    (str(item["family"]), float(item["gpu_time_pct"]))
                    for item in raw
                    if isinstance(item, Mapping)
                ),
                value["semantic_coverage_complete"],
                tuple(value["semantic_unresolved_reasons"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("Planning coverage is invalid", "invalid_diagnosis") from error
        if (
            not isinstance(value.get("semantic_coverage_complete"), bool)
            or not isinstance(value.get("semantic_unresolved_reasons"), list)
            or any(not isinstance(item, str) for item in value["semantic_unresolved_reasons"])
            or len(result.unclassified_families) != len(raw)
            or value.get("satisfied") is not result.satisfied
        ):
            raise ContractError("Planning coverage verdict differs", "invalid_diagnosis")
        return result


@dataclass(frozen=True, slots=True)
class KernelOpportunityPlan:
    opportunities: tuple[KernelOpportunity, ...]
    measured_order: tuple[str, ...]
    recoverable_order: tuple[str, ...]
    coverage: KernelPlanningCoverage
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
    require_coverage: bool = True,
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
        opportunity = _build_opportunity(
            record, roi[record.candidate_id], correctness_oracles
        )
        planned.append(opportunity)
        if opportunity.eligible:
            eligible_count += 1
        if eligible_count >= max_kernels:
            break
    coverage = _planning_coverage(evidence, tuple(planned))
    _require_planning_coverage(coverage, required=require_coverage)
    planned_ids = {item.evidence_id for item in planned}
    return KernelOpportunityPlan(
        opportunities=tuple(planned),
        measured_order=tuple(item.opportunity_id for item in planned),
        recoverable_order=tuple(
            f"kernel-{item.candidate_id[:24]}"
            for item in rankings.recoverable
            if item.candidate_id in planned_ids
        ),
        coverage=coverage,
        correctness_oracle_policy_sha256=(
            correctness_oracles.policy_sha256 if correctness_oracles is not None else None
        ),
    )


def _build_opportunity(
    record: TraceEvidence,
    roi_prior: float,
    correctness_oracles: CorrectnessOracleRegistry | None,
) -> KernelOpportunity:
    source = Path(record.kernel.source_path) if record.kernel.source_path else None
    root = Path(record.kernel.source_root) if record.kernel.source_root else None
    test = Path(record.kernel.test_file) if record.kernel.test_file else None
    oracle = _resolve_oracle(
        correctness_oracles,
        record=record,
        source=source,
        root=root,
        observed_test=test,
        observed_command=record.kernel.test_command,
    )
    command = oracle.test_command if oracle else record.kernel.test_command
    test = oracle.test_file if oracle else test
    reason = _eligibility_reason(record, source, root, test, command)
    return KernelOpportunity(
        f"kernel-{record.candidate_id[:24]}", record.candidate_id,
        record.kernel.runtime_name, record.op.name, record.phase, record.rank,
        record.kernel.language, record.kernel.origin_library,
        tuple(record.shape.concrete_inputs), record.shape.dtypes,
        record.shape.graph_mode, record.match_confidence, record.volume.gpu_time_pct,
        roi_prior, source, root, test, command,
        "eligible" if reason == "eligible" else "unresolved", reason,
        oracle.binding_sha256 if oracle else None,
        oracle.execution_mode if oracle else None,
    )


def _planning_coverage(
    evidence: tuple[TraceEvidence, ...],
    planned: tuple[KernelOpportunity, ...],
) -> KernelPlanningCoverage:
    observed = sum(item.volume.gpu_time_pct for item in evidence)
    if not math.isfinite(observed) or observed > 100.000001:
        raise ContractError("Trace GPU-time shares exceed 100%", "invalid_trace_coverage")
    selected_ids = {item.evidence_id for item in planned}
    selected = sum(
        item.volume.gpu_time_pct for item in evidence if item.candidate_id in selected_ids
    )
    families: dict[str, float] = {}
    for item in evidence:
        if (
            _unclassified(item.op.category)
            and _unclassified(item.op.name)
            and _unclassified(item.kernel.runtime_name)
        ):
            family = f"{item.op.category}:{item.op.name}:{item.kernel.runtime_name}"
            families[family] = families.get(family, 0.0) + item.volume.gpu_time_pct
    semantic_reasons = _semantic_reasons(evidence)
    return KernelPlanningCoverage(
        "kernel_family_gpu_time_coverage_v1", 90.0, 2.0,
        observed, selected, tuple(sorted(families.items())),
        not semantic_reasons, semantic_reasons,
    )


def _require_planning_coverage(
    coverage: KernelPlanningCoverage, *, required: bool
) -> None:
    if not required or coverage.satisfied:
        return
    largest = max((share for _, share in coverage.unclassified_families), default=0.0)
    reason = (
        "trace_semantic_coverage_unresolved"
        if not coverage.semantic_coverage_complete
        else (
            "unclassified_kernel_family_too_large"
            if largest > coverage.maximum_unclassified_family_gpu_pct
            else "insufficient_kernel_family_coverage"
        )
    )
    raise ContractError(
        f"Kernel planning coverage failed: selected={coverage.selected_gpu_pct:.6g}%, "
        f"largest_unclassified={largest:.6g}%",
        reason,
    )


def _unclassified(value: str) -> bool:
    return value.strip().lower() in {"", "unknown", "unclassified"}


def _semantic_reasons(evidence: tuple[TraceEvidence, ...]) -> tuple[str, ...]:
    reasons: set[str] = set()
    for item in evidence:
        reasons.update(
            warning.split(":", 1)[1]
            for warning in item.evidence.warnings
            if warning.startswith("semantic_coverage_unresolved:")
        )
        reasons.update(
            f"dropped:{reason}"
            for reason, count in item.evidence.coverage.dropped_by_reason
            if count
        )
    return tuple(sorted(reasons))


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
    "KernelPlanningCoverage",
    "build_kernel_opportunity_plan",
]
