"""Validated Magpie/TraceLens evidence consumed by Apex."""

from .adapter import MagpieTraceEvidenceAdapter
from .comparison import PinnedTraceLensComparisonAdapter
from .evidence import (
    AcquisitionCoverage,
    EvidenceArtifacts,
    KernelEvidence,
    KernelVolume,
    OperationEvidence,
    PerformanceModelEvidence,
    ShapeEvidence,
    TraceEvidence,
    TraceEvidenceNormalizer,
    derive_candidate_id,
)
from .ranking import OpportunityRankings, RankedOpportunity, predicted_e2e_gain_pct, rank_evidence
from .targeted_trace import (
    EvidenceArtifactReceipt,
    TargetedTraceValidator,
    ValidatedTargetedEvent,
    ValidatedTargetedTrace,
)

__all__ = [
    "AcquisitionCoverage",
    "EvidenceArtifactReceipt",
    "EvidenceArtifacts",
    "KernelEvidence",
    "KernelVolume",
    "MagpieTraceEvidenceAdapter",
    "PinnedTraceLensComparisonAdapter",
    "OperationEvidence",
    "OpportunityRankings",
    "PerformanceModelEvidence",
    "RankedOpportunity",
    "ShapeEvidence",
    "TraceEvidence",
    "TraceEvidenceNormalizer",
    "TargetedTraceValidator",
    "ValidatedTargetedEvent",
    "ValidatedTargetedTrace",
    "derive_candidate_id",
    "predicted_e2e_gain_pct",
    "rank_evidence",
]
