"""Deterministic dual ranking for measured and modeled opportunity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from apex.core import ContractError

from .evidence import TraceEvidence


@dataclass(frozen=True, slots=True)
class RankedOpportunity:
    candidate_id: str
    measured_gpu_pct: float
    roi_prior: float
    perf_model_used: bool
    predicted_e2e_gain_pct: float | None = None


@dataclass(frozen=True, slots=True)
class OpportunityRankings:
    measured: tuple[RankedOpportunity, ...]
    recoverable: tuple[RankedOpportunity, ...]


def rank_evidence(
    records: tuple[TraceEvidence, ...],
    *,
    min_gpu_pct: float = 0.0,
    isolated_speedups: dict[str, float] | None = None,
) -> OpportunityRankings:
    """Return separate measured-share and headroom rankings."""

    if not math.isfinite(min_gpu_pct) or min_gpu_pct < 0:
        raise ContractError("min_gpu_pct must be finite and non-negative", "invalid_ranking_threshold")
    speedups = isolated_speedups or {}
    opportunities: list[RankedOpportunity] = []
    for record in records:
        share = record.volume.gpu_time_pct
        if share < min_gpu_pct:
            continue
        model = record.perf_model
        usable = (
            model.has_model
            and model.pct_roofline is not None
            and model.confidence in {"medium", "high"}
        )
        headroom = 1.0 - min(max(model.pct_roofline or 0.0, 0.0), 100.0) / 100.0 if usable else 1.0
        speedup = speedups.get(record.candidate_id)
        predicted = predicted_e2e_gain_pct(share, speedup) if speedup is not None else None
        opportunities.append(
            RankedOpportunity(record.candidate_id, share, share * headroom, usable, predicted)
        )
    measured = tuple(sorted(opportunities, key=lambda item: (-item.measured_gpu_pct, item.candidate_id)))
    recoverable = tuple(sorted(opportunities, key=lambda item: (-item.roi_prior, item.candidate_id)))
    return OpportunityRankings(measured, recoverable)


def predicted_e2e_gain_pct(gpu_time_pct: float, isolated_speedup: float) -> float:
    """Compute the Amdahl upper-bound gain for one isolated speedup."""

    if not (math.isfinite(gpu_time_pct) and 0 <= gpu_time_pct <= 100):
        raise ContractError("gpu_time_pct must be in [0, 100]", "invalid_amdahl_input")
    if not math.isfinite(isolated_speedup) or isolated_speedup <= 0:
        raise ContractError("isolated_speedup must be positive", "invalid_amdahl_input")
    fraction = gpu_time_pct / 100.0
    speedup = 1.0 / ((1.0 - fraction) + fraction / isolated_speedup)
    return (speedup - 1.0) * 100.0


__all__ = ["OpportunityRankings", "RankedOpportunity", "predicted_e2e_gain_pct", "rank_evidence"]
