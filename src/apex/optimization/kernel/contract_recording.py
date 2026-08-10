"""Canonical CAS/event recording for a frozen kernel evaluation contract."""

from __future__ import annotations

from apex.core import canonical_json_bytes
from apex.evaluation import EvaluationContractReceipt
from apex.orchestration import RunController
from apex.storage import ArtifactReceipt, ArtifactStore


def record_evaluation_contract(
    *,
    artifacts: ArtifactStore,
    controller: RunController,
    contract: EvaluationContractReceipt,
) -> ArtifactReceipt:
    """Persist evaluator authority before the first candidate command."""

    receipt = artifacts.put_bytes(
        canonical_json_bytes(contract.to_dict()), media_type="application/json"
    )
    controller.record_domain_event(
        "dependency_verified",
        {
            "kind": "evaluation_contract",
            "status": contract.status,
            "contract_digest": contract.digest,
            "authority_receipt_digest": (
                contract.authority.digest if contract.authority else None
            ),
            "authority_id": (
                contract.authority.authority.authority_id
                if contract.authority
                else None
            ),
            "authority_kind": (
                contract.authority.authority.kind.value
                if contract.authority
                else None
            ),
            "artifacts": [
                {"role": "evaluation_contract", "receipt": receipt.to_dict()}
            ],
        },
        idempotency_key="evaluation_contract.frozen",
    )
    return receipt


def record_authorized_evaluation_contract(
    *,
    artifacts: ArtifactStore,
    controller: RunController,
    contract: EvaluationContractReceipt,
) -> ArtifactReceipt:
    """Append the exact user-confirmed authority without replacing the draft."""

    if not contract.verified or contract.authority is None:
        raise ValueError("authorized evaluation contract must be verified")
    receipt = artifacts.put_bytes(
        canonical_json_bytes(contract.to_dict()), media_type="application/json"
    )
    controller.record_domain_event(
        "dependency_verified",
        {
            "kind": "evaluation_contract_authorized",
            "status": contract.status,
            "contract_digest": contract.digest,
            "draft_digest": contract.draft.digest,
            "authority_receipt_digest": contract.authority.digest,
            "authority_id": contract.authority.authority.authority_id,
            "authority_kind": contract.authority.authority.kind.value,
            "artifacts": [
                {"role": "evaluation_contract", "receipt": receipt.to_dict()}
            ],
        },
        idempotency_key="evaluation_contract.authorized",
    )
    return receipt


__all__ = [
    "record_authorized_evaluation_contract",
    "record_evaluation_contract",
]
