"""Canonical event/CAS binding for the Apex bytes executing a run."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, Mapping

from apex.core import ApexError, IntegrityError, canonical_json_bytes
from apex.orchestration import RunController
from apex.runtime import ApexExecutionIdentity
from apex.storage import ArtifactReceipt, ArtifactStore


def record_apex_execution_identity(
    artifacts: ArtifactStore,
    controller: RunController,
    identity: ApexExecutionIdentity,
) -> ArtifactReceipt:
    """Persist an observed execution identity without granting release authority."""

    document = identity.to_dict()
    receipt = artifacts.put_bytes(
        canonical_json_bytes(document), media_type="application/json"
    )
    controller.record_domain_event(
        "provenance_observed",
        {
            "kind": "apex_execution_identity",
            "execution_identity_sha256": identity.receipt_sha256,
            "apex_tree": identity.apex_tree,
            "source_manifest_sha256": identity.source_manifest_sha256,
            "dependency_lock_sha256": document["dependency_lock_sha256"],
            "artifacts": [
                {"role": "apex_execution_identity", "receipt": receipt.to_dict()}
            ],
        },
        idempotency_key="apex.execution_identity.recorded",
    )
    return receipt


def load_recorded_execution_identity(
    artifacts: ArtifactStore,
    events: Sequence[Any],
) -> ApexExecutionIdentity:
    """Load and independently validate the sole recorded execution identity."""

    matches = tuple(
        event
        for event in events
        if event.event_type == "provenance_observed"
        and event.payload.get("kind") == "apex_execution_identity"
    )
    if len(matches) != 1:
        raise IntegrityError(
            "Apex execution identity is missing or ambiguous",
            "execution_identity_ambiguous",
        )
    event = matches[0]
    receipt = _one_receipt(event.payload, "apex_execution_identity")
    try:
        document = json.loads(artifacts.read_bytes(receipt))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(
            "Apex execution identity artifact is invalid JSON",
            "execution_identity_tampered",
        ) from error
    if not isinstance(document, Mapping):
        raise IntegrityError(
            "Apex execution identity artifact root differs",
            "execution_identity_tampered",
        )
    identity = ApexExecutionIdentity.from_dict(document)
    if (
        event.payload.get("execution_identity_sha256") != identity.receipt_sha256
        or event.payload.get("apex_tree") != identity.apex_tree
        or event.payload.get("source_manifest_sha256")
        != identity.source_manifest_sha256
        or event.payload.get("dependency_lock_sha256")
        != identity.document["dependency_lock_sha256"]
    ):
        raise IntegrityError(
            "Apex execution identity event differs from its artifact",
            "execution_identity_tampered",
        )
    return identity


def require_same_execution_identity(
    expected_sha256: str,
    current: ApexExecutionIdentity,
) -> None:
    """Reject recovery under different Apex package bytes or lock identity."""

    if expected_sha256 != current.receipt_sha256:
        raise IntegrityError(
            "Current Apex execution identity differs from the interrupted run",
            "resume_execution_identity_mismatch",
        )


def recorded_execution_identity_reason(
    artifacts: ArtifactStore,
    events: Sequence[Any],
    current: ApexExecutionIdentity,
) -> str | None:
    """Return a stable reason when a formal continuation changed Apex bytes."""

    try:
        recorded = load_recorded_execution_identity(artifacts, events)
        require_same_execution_identity(recorded.receipt_sha256, current)
    except ApexError as error:
        return error.reason_code
    return None


def _one_receipt(payload: Mapping[str, Any], role: str) -> ArtifactReceipt:
    artifacts = payload.get("artifacts")
    matches = tuple(
        item
        for item in artifacts if isinstance(item, Mapping) and item.get("role") == role
    ) if isinstance(artifacts, list) else ()
    if len(matches) != 1 or not isinstance(matches[0].get("receipt"), Mapping):
        raise IntegrityError(
            "Apex execution identity artifact is missing or ambiguous",
            "execution_identity_ambiguous",
        )
    return ArtifactReceipt.from_dict(dict(matches[0]["receipt"]))


__all__ = [
    "load_recorded_execution_identity",
    "record_apex_execution_identity",
    "recorded_execution_identity_reason",
    "require_same_execution_identity",
]
