"""Trusted verification boundary for release qualification artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from apex.core import ContractError, sha256_json


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class QualificationAuthorityReceipt:
    """Path-free result minted by an injected artifact verifier.

    The receipt is not a trust anchor by itself.  Release code accepts it only as
    the direct result of the composition-root supplied authority port.
    """

    qualification_id: str
    evidence_receipt_sha256: str
    artifact_manifest_sha256: str
    verifier_identity_sha256: str
    authority_id: str
    receipt_sha256: str

    SCHEMA = "apex.qualification-authority-receipt/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.qualification_id, str) or not self.qualification_id:
            _invalid("qualification authority id is invalid")
        if not isinstance(self.authority_id, str) or not self.authority_id:
            _invalid("qualification authority identity is invalid")
        for field in (
            "evidence_receipt_sha256",
            "artifact_manifest_sha256",
            "verifier_identity_sha256",
            "receipt_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, field))) is None:
                _invalid(f"{field} is invalid")
        if self.receipt_sha256 != sha256_json(self.payload()):
            _invalid("qualification authority receipt digest differs")

    def payload(self) -> dict[str, str]:
        return {
            "schema": self.SCHEMA,
            "qualification_id": self.qualification_id,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "verifier_identity_sha256": self.verifier_identity_sha256,
            "authority_id": self.authority_id,
        }

    def to_dict(self) -> dict[str, str]:
        return {**self.payload(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_dict(cls, value: object) -> QualificationAuthorityReceipt:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        if not isinstance(value, Mapping) or set(value) != fields:
            _invalid("qualification authority receipt fields differ")
        if value["schema"] != cls.SCHEMA:
            _invalid("qualification authority receipt schema differs")
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})


def build_qualification_authority_receipt(
    *,
    qualification_id: str,
    evidence_receipt_sha256: str,
    artifact_manifest_sha256: str,
    verifier_identity_sha256: str,
    authority_id: str,
) -> QualificationAuthorityReceipt:
    """Build the canonical result returned by a trusted verifier adapter."""

    payload = {
        "schema": QualificationAuthorityReceipt.SCHEMA,
        "qualification_id": qualification_id,
        "evidence_receipt_sha256": evidence_receipt_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "verifier_identity_sha256": verifier_identity_sha256,
        "authority_id": authority_id,
    }
    return QualificationAuthorityReceipt(
        qualification_id=qualification_id,
        evidence_receipt_sha256=evidence_receipt_sha256,
        artifact_manifest_sha256=artifact_manifest_sha256,
        verifier_identity_sha256=verifier_identity_sha256,
        authority_id=authority_id,
        receipt_sha256=sha256_json(payload),
    )


class QualificationAuthorityPort(Protocol):
    """Recompute one claim from evaluator-owned artifacts outside the claim."""

    def verify(
        self, evidence: Mapping[str, Any]
    ) -> QualificationAuthorityReceipt: ...


def _invalid(message: str) -> None:
    raise ContractError(message, "invalid_qualification_authority_receipt")


__all__ = [
    "QualificationAuthorityPort",
    "QualificationAuthorityReceipt",
    "build_qualification_authority_receipt",
]
