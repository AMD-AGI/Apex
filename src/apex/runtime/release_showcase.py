"""Path-free binding for an official offline showcase-verifier receipt."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import ContractError, sha256_json


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class ShowcaseEvidence:
    """Official verifier fields plus the Apex source tree that consumed them."""

    showcase_id: str
    apex_tree: str
    status: str
    file_count: int
    checksums_sha256: str
    event_count: int
    artifact_count: int
    reward_replayed: bool
    bundle_verified: bool
    reproduction_verified: bool
    episode_sha256: str
    artifact_manifest_sha256: str
    reward_sha256: str
    result_sha256: str
    reproduction_sha256: str
    verification_receipt_sha256: str

    SCHEMA = "apex.release-showcase-verification/v2"
    VERIFIER_SCHEMA = "apex.showcase-verification/v2"

    def __post_init__(self) -> None:
        if not isinstance(self.showcase_id, str) or not self.showcase_id:
            raise _invalid("showcase_id is invalid")
        _match(self.apex_tree, _GIT, "showcase Apex tree")
        if self.status not in {"pending", "published"}:
            raise _invalid("showcase status is invalid")
        for name in ("file_count", "event_count", "artifact_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise _invalid(f"{name} is invalid")
        for name in (
            "checksums_sha256",
            "episode_sha256",
            "artifact_manifest_sha256",
            "reward_sha256",
            "result_sha256",
            "reproduction_sha256",
            "verification_receipt_sha256",
        ):
            _match(getattr(self, name), _SHA256, name)
        for name in (
            "reward_replayed",
            "bundle_verified",
            "reproduction_verified",
        ):
            if type(getattr(self, name)) is not bool:
                raise _invalid(f"{name} is invalid")
        if self.status == "published" and not all((
            self.reward_replayed,
            self.bundle_verified,
            self.reproduction_verified,
        )):
            raise _invalid("published showcase verification is incomplete")
        if self.verification_receipt_sha256 != sha256_json(self.verifier_payload()):
            raise _invalid("showcase verifier receipt digest differs")

    def verifier_payload(self) -> dict[str, Any]:
        return {
            "schema": self.VERIFIER_SCHEMA,
            "showcase_id": self.showcase_id,
            "status": self.status,
            "file_count": self.file_count,
            "checksums_sha256": self.checksums_sha256,
            "event_count": self.event_count,
            "artifact_count": self.artifact_count,
            "reward_replayed": self.reward_replayed,
            "bundle_verified": self.bundle_verified,
            "reproduction_verified": self.reproduction_verified,
            "episode_sha256": self.episode_sha256,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "reward_sha256": self.reward_sha256,
            "result_sha256": self.result_sha256,
            "reproduction_sha256": self.reproduction_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "apex_tree": self.apex_tree,
            **{
                key: value
                for key, value in self.verifier_payload().items()
                if key != "schema"
            },
            "verification_receipt_sha256": self.verification_receipt_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> ShowcaseEvidence:
        fields = set(cls.__dataclass_fields__) | {"schema"}
        raw = _strict(value, fields)
        if raw["schema"] != cls.SCHEMA:
            raise _invalid("showcase verification schema differs")
        return cls(**{field: raw[field] for field in cls.__dataclass_fields__})


def build_showcase_evidence(
    *,
    apex_tree: str,
    verifier_receipt: Mapping[str, Any],
) -> ShowcaseEvidence:
    """Convert a path-free `showcase verify` v2 receipt into release evidence."""

    fields = set(ShowcaseEvidence.__dataclass_fields__) - {"apex_tree"}
    raw = _strict(verifier_receipt, fields | {"schema"})
    if raw["schema"] != ShowcaseEvidence.VERIFIER_SCHEMA:
        raise _invalid("showcase verifier schema differs")
    values = {field: raw[field] for field in fields}
    return ShowcaseEvidence(apex_tree=apex_tree, **values)


def _strict(value: object, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _invalid("showcase verification fields differ")
    return value


def _match(value: object, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _invalid(f"{label} is invalid")


def _invalid(message: str) -> ContractError:
    return ContractError(message, "invalid_release_evidence")


__all__ = ["ShowcaseEvidence", "build_showcase_evidence"]
