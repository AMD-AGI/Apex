"""Stable public facade for normalized diagnostic evidence.

Contracts live in :mod:`evidence_models`; the exact-symbol ingestion pipeline
lives in :mod:`evidence_normalizer`.  Keeping this facade preserves the public
import path while preventing either implementation unit from becoming a
monolith.
"""

from .evidence_models import (
    EvidenceArtifacts,
    KernelEvidence,
    KernelVolume,
    OperationEvidence,
    PerformanceModelEvidence,
    ShapeEvidence,
    TraceEvidence,
    derive_candidate_id,
)
from .evidence_normalizer import TraceEvidenceNormalizer
from .targeted_trace import AcquisitionCoverage, EvidenceArtifactReceipt

__all__ = [
    "AcquisitionCoverage",
    "EvidenceArtifactReceipt",
    "EvidenceArtifacts",
    "KernelEvidence",
    "KernelVolume",
    "OperationEvidence",
    "PerformanceModelEvidence",
    "ShapeEvidence",
    "TraceEvidence",
    "TraceEvidenceNormalizer",
    "derive_candidate_id",
]
