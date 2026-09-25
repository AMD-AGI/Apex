"""Release binding for path-free official showcase-verifier receipts."""

from __future__ import annotations

import copy

import pytest

from apex.core import ContractError, sha256_json
from apex.runtime import ShowcaseEvidence, build_showcase_evidence


_SHA = "a" * 64


def _receipt() -> dict:
    value = {
        "schema": "apex.showcase-verification/v2",
        "showcase_id": "kernel-triton-paged-attention-2d",
        "status": "published",
        "file_count": 15,
        "checksums_sha256": _SHA,
        "event_count": 40,
        "artifact_count": 12,
        "reward_replayed": True,
        "bundle_verified": True,
        "reproduction_verified": True,
        "episode_sha256": _SHA,
        "artifact_manifest_sha256": _SHA,
        "reward_sha256": _SHA,
        "result_sha256": _SHA,
        "reproduction_sha256": _SHA,
    }
    value["verification_receipt_sha256"] = sha256_json(value)
    return value


def test_release_showcase_round_trips_official_verifier_receipt() -> None:
    evidence = build_showcase_evidence(
        apex_tree="b" * 40,
        verifier_receipt=_receipt(),
    )

    assert ShowcaseEvidence.from_dict(evidence.to_dict()) == evidence
    assert evidence.verification_receipt_sha256 == _receipt()[
        "verification_receipt_sha256"
    ]


def test_showcase_artifact_or_verifier_digest_tamper_fails() -> None:
    receipt = _receipt()
    receipt["reward_sha256"] = "f" * 64
    with pytest.raises(ContractError, match="verifier receipt digest differs"):
        build_showcase_evidence(
            apex_tree="b" * 40,
            verifier_receipt=receipt,
        )

    evidence = build_showcase_evidence(
        apex_tree="b" * 40,
        verifier_receipt=_receipt(),
    ).to_dict()
    changed = copy.deepcopy(evidence)
    changed["verification_receipt_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="verifier receipt digest differs"):
        ShowcaseEvidence.from_dict(changed)


def test_published_showcase_cannot_claim_incomplete_offline_verification() -> None:
    receipt = _receipt()
    receipt["reproduction_verified"] = False
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "verification_receipt_sha256"
    }
    receipt["verification_receipt_sha256"] = sha256_json(payload)

    with pytest.raises(ContractError, match="verification is incomplete"):
        build_showcase_evidence(
            apex_tree="b" * 40,
            verifier_receipt=receipt,
        )


def test_legacy_boolean_only_showcase_evidence_is_rejected() -> None:
    with pytest.raises(ContractError, match="fields differ"):
        ShowcaseEvidence.from_dict({
            "schema": "apex.release-showcase-verification/v1",
            "showcase_id": "kernel-triton-paged-attention-2d",
            "apex_tree": "b" * 40,
            "status": "published",
            "checksums_sha256": _SHA,
            "bundle_verified": True,
            "reward_replayed": True,
            "reproduction_verified": True,
        })
