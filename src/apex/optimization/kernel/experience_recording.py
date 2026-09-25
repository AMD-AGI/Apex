"""Standalone experience events that are not measured outcomes."""

from __future__ import annotations

from typing import Protocol, Sequence

from apex.core import IntegrityError
from apex.knowledge import ExperienceIdentity
from apex.orchestration import RunController
from apex.storage import ArtifactReceipt


class DeferredExperienceRecord(Protocol):
    """Minimal run-record surface needed to append a deferred observation."""

    controller: RunController
    dataset_split: str
    data_visibility: str


def record_deferred_experience(
    record: DeferredExperienceRecord,
    attempt_id: str,
    *,
    identity: ExperienceIdentity,
    strategy_fingerprint: str,
    reason: str,
    evidence: Sequence[ArtifactReceipt],
) -> None:
    """Record an externally pending candidate without claiming an outcome."""

    unique = tuple({item.digest: item for item in evidence}.values())
    if not unique:
        raise IntegrityError(
            "Deferred experience has no evidence",
            "missing_experience_evidence",
        )
    record.controller.record_domain_event(
        "experience.deferred",
        {
            "attempt_id": attempt_id,
            "candidate_id": attempt_id,
            "anchor_generation": record.controller.state.anchor_generation,
            "split": record.dataset_split,
            "visibility": record.data_visibility,
            "evidence_class": "derived",
            "dry_run": False,
            "identity": identity.to_dict(),
            "status": "pending_external_evaluator",
            "strategy_fingerprint": strategy_fingerprint,
            "reason": reason,
            "external_verification_required": True,
            "evidence_receipts": [item.digest for item in unique],
            "artifacts": [
                {
                    "role": "deferred_experience_evidence",
                    "receipt": item.to_dict(),
                }
                for item in unique
            ],
        },
        idempotency_key=f"attempt.{attempt_id}.experience",
    )


__all__ = ["record_deferred_experience"]
