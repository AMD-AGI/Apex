"""Canonical CAS/event binding for a verified campaign software baseline."""

from __future__ import annotations

from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes
from apex.orchestration import RunController
from apex.runtime import ReleaseCandidateReceipt
from apex.storage import ArtifactReceipt, ArtifactStore


def record_campaign_baseline(
    artifacts: ArtifactStore,
    controller: RunController,
    document: Mapping[str, Any],
    *,
    idempotency_key: str = "campaign.baseline.verified",
) -> ArtifactReceipt:
    """Persist a path-free ready baseline before any workload command is recorded."""

    _validate(document)
    receipt = artifacts.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )
    controller.record_domain_event(
        "dependency_verified",
        {
            "kind": "campaign_baseline",
            "release_candidate_receipt_sha256": document["receipt_sha256"],
            "apex_tree": document["static"]["apex_checkout"]["tree"],
            "artifacts": [{"role": "campaign_baseline", "receipt": receipt.to_dict()}],
        },
        idempotency_key=idempotency_key,
    )
    return receipt


def _validate(value: Mapping[str, Any]) -> None:
    digest = value.get("receipt_sha256")
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    static = value.get("static")
    checkout = static.get("apex_checkout") if isinstance(static, Mapping) else None
    valid = (
        value.get("schema") == "apex.release-candidate-receipt/v2"
        and value.get("baseline_status") == "ready"
        and value.get("baseline_blockers") == []
        and isinstance(digest, str)
        and sha256_bytes(canonical_json_bytes(payload)) == digest
        and isinstance(checkout, Mapping)
        and isinstance(checkout.get("tree"), str)
    )
    if not valid:
        raise ContractError(
            "Campaign baseline receipt is invalid or blocked",
            "campaign_baseline_receipt_invalid",
        )


def validate_resume_campaign_baseline(
    stored: Mapping[str, Any] | None,
    current: ReleaseCandidateReceipt | None,
) -> None:
    """Require the current verified baseline to equal the interrupted run's bytes."""

    if stored is None:
        if current is not None:
            raise ContractError(
                "Interrupted run has no campaign baseline receipt",
                "resume_campaign_baseline_missing",
            )
        return
    if current is None:
        raise ContractError(
            "Resume requires the original campaign baseline receipt",
            "campaign_baseline_receipt_required",
        )
    if current.to_dict() != dict(stored):
        raise IntegrityError(
            "Resume campaign baseline differs from the original run",
            "resume_campaign_baseline_mismatch",
        )


__all__ = ["record_campaign_baseline", "validate_resume_campaign_baseline"]
