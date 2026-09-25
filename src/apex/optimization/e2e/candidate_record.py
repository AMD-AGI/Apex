"""Persist evaluator-owned candidate snapshots without reopening agent paths."""

from __future__ import annotations

from typing import Any

from apex.core import IntegrityError
from apex.storage import ArtifactReceipt, ArtifactStore

from .candidate import E2ECandidate, validate_frozen_sources


def store_candidate_sources(
    artifacts: ArtifactStore, candidate: E2ECandidate
) -> tuple[ArtifactReceipt, ...]:
    """Write the frozen bytes to CAS and verify every resulting identity."""

    if candidate.succeeded:
        validate_frozen_sources(candidate)
    receipts = tuple(
        artifacts.put_bytes(source.content, media_type="text/x-python")
        for source in candidate.frozen_sources
    )
    if any(
        receipt.digest != source.sha256
        for receipt, source in zip(receipts, candidate.frozen_sources, strict=True)
    ):
        raise IntegrityError(
            "Frozen candidate receipt differs from captured source",
            "candidate_source_receipt_mismatch",
        )
    return receipts


def candidate_manifest(
    candidate: E2ECandidate, source_receipts: tuple[ArtifactReceipt, ...]
) -> dict[str, Any]:
    """Project one candidate and its immutable-source bindings into JSON data."""

    return {
        "schema_version": 1,
        "attempt_id": candidate.attempt_id,
        "candidate_id": candidate.candidate_id,
        "succeeded": candidate.succeeded,
        "reason_code": candidate.reason_code,
        "workspace": str(candidate.workspace),
        "editable_files": list(candidate.editable_files),
        "changed_files": list(candidate.changed_files),
        "baseline_source_sha256": candidate.baseline_source_sha256,
        "candidate_source_sha256": candidate.candidate_source_sha256,
        "frozen_sources": [
            {
                "path": source.relative_path,
                "sha256": source.sha256,
                "mode": source.mode,
                "size": len(source.content),
            }
            for source in candidate.frozen_sources
        ],
        "source_receipts": [item.to_dict() for item in source_receipts],
    }


__all__ = ["candidate_manifest", "store_candidate_sources"]
