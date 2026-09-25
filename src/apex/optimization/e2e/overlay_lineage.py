"""Evidence-bound ancestry for cumulative Docker runtime overlays."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from apex.core import ContractError, IntegrityError, ValidationLevel, sha256_json
from apex.runtime import RunProvenance

from .overlay_runtime import BuiltOverlay, ContainerImage, LoadedFileReceipt
from .services import AcceptedCandidate


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPO_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_PARENT_KINDS = {"initial_provenance", "accepted_overlay"}


@dataclass(frozen=True, slots=True)
class OverlayBuildReceipt:
    """One overlay layer, including the complete accepted parent ancestry."""

    schema_version: int
    anchor_generation: int
    candidate_id: str
    candidate_source_sha256: str
    parent_kind: str
    parent_image_id: str
    parent_locator: str
    parent_candidate_id: str | None
    parent_decision_receipt: str | None
    parent_ancestry_image_ids: tuple[str, ...]
    parent_accepted_candidate_ids: tuple[str, ...]
    derived_image_id: str
    dockerfile_sha256: str
    candidate_file_sha256: str
    loaded_candidate_sha256: str

    def __post_init__(self) -> None:
        hashes = (
            self.candidate_source_sha256,
            self.dockerfile_sha256,
            self.candidate_file_sha256,
            self.loaded_candidate_sha256,
        )
        if (
            self.schema_version != 1
            or self.anchor_generation < 0
            or not self.candidate_id
            or self.parent_kind not in _PARENT_KINDS
            or not _IMAGE_ID.fullmatch(self.parent_image_id)
            or not _IMAGE_ID.fullmatch(self.derived_image_id)
            or any(
                not _IMAGE_ID.fullmatch(item)
                for item in self.parent_ancestry_image_ids
            )
            or any(not item for item in self.parent_accepted_candidate_ids)
            or any(not _SHA256.fullmatch(item) for item in hashes)
        ):
            raise ContractError(
                "Overlay build receipt is invalid", "invalid_overlay_build_receipt"
            )
        initial = self.parent_kind == "initial_provenance"
        if initial != (self.parent_candidate_id is None):
            raise ContractError(
                "Overlay parent candidate is invalid", "invalid_overlay_build_receipt"
            )
        if initial != (self.parent_decision_receipt is None):
            raise ContractError(
                "Overlay parent decision is invalid", "invalid_overlay_build_receipt"
            )
        if self.parent_decision_receipt is not None and not _SHA256.fullmatch(
            self.parent_decision_receipt
        ):
            raise ContractError(
                "Overlay parent decision is invalid", "invalid_overlay_build_receipt"
            )
        if len(self.parent_ancestry_image_ids) != self.anchor_generation + 1:
            raise ContractError(
                "Overlay image ancestry is incomplete", "invalid_overlay_build_receipt"
            )
        if len(self.parent_accepted_candidate_ids) != self.anchor_generation:
            raise ContractError(
                "Overlay candidate ancestry is incomplete",
                "invalid_overlay_build_receipt",
            )
        if (
            (initial and not _REPO_DIGEST.fullmatch(self.parent_locator))
            or (not initial and self.parent_locator != self.parent_image_id)
            or self.derived_image_id in self.parent_ancestry_image_ids
        ):
            raise ContractError(
                "Overlay parent locator is invalid", "invalid_overlay_build_receipt"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OverlayBuildReceipt":
        try:
            return cls(
                schema_version=int(value["schema_version"]),
                anchor_generation=int(value["anchor_generation"]),
                candidate_id=str(value["candidate_id"]),
                candidate_source_sha256=str(value["candidate_source_sha256"]),
                parent_kind=str(value["parent_kind"]),
                parent_image_id=str(value["parent_image_id"]),
                parent_locator=str(value["parent_locator"]),
                parent_candidate_id=_optional_string(value.get("parent_candidate_id")),
                parent_decision_receipt=_optional_string(
                    value.get("parent_decision_receipt")
                ),
                parent_ancestry_image_ids=_string_tuple(
                    value["parent_ancestry_image_ids"]
                ),
                parent_accepted_candidate_ids=_string_tuple(
                    value["parent_accepted_candidate_ids"]
                ),
                derived_image_id=str(value["derived_image_id"]),
                dockerfile_sha256=str(value["dockerfile_sha256"]),
                candidate_file_sha256=str(value["candidate_file_sha256"]),
                loaded_candidate_sha256=str(value["loaded_candidate_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError(
                "Overlay build receipt is malformed", "invalid_overlay_build_receipt"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["parent_ancestry_image_ids"] = list(self.parent_ancestry_image_ids)
        value["parent_accepted_candidate_ids"] = list(
            self.parent_accepted_candidate_ids
        )
        return value

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def validate_accepted_overlay_parent(
    *,
    reference: str,
    inspected: ContainerImage,
    provenance: RunProvenance,
    accepted: Sequence[AcceptedCandidate],
    anchor_generation: int,
) -> ContainerImage:
    """Authorize only the exact last image from a fully verified KEEP chain."""

    if not accepted or anchor_generation != len(accepted):
        raise IntegrityError(
            "Accepted overlay generation drifted", "overlay_ancestry_mismatch"
        )
    _validate_accepted_stack(accepted, provenance)
    expected = accepted[-1].deployment.deployed_image_id
    if (
        expected is None
        or reference != expected
        or inspected.reference != reference
        or inspected.image_id != expected
    ):
        raise IntegrityError(
            "Live anchor is not the committed derived image",
            "overlay_parent_identity_mismatch",
            {
                "expected_image_id": expected,
                "reference": reference,
                "observed": inspected.image_id,
            },
        )
    return ContainerImage(reference, inspected.image_id, inspected.repo_digests)


def capture_overlay_build_receipt(
    *,
    candidate_id: str,
    candidate_source_sha256: str,
    parent: ContainerImage,
    built: BuiltOverlay,
    dockerfile_sha256: str,
    candidate_file_sha256: str,
    loaded: LoadedFileReceipt,
    accepted: Sequence[AcceptedCandidate],
    anchor_generation: int,
    provenance: RunProvenance,
) -> OverlayBuildReceipt:
    if anchor_generation != len(accepted):
        raise IntegrityError(
            "Accepted overlay generation drifted", "overlay_ancestry_mismatch"
        )
    if accepted:
        _validate_accepted_stack(accepted, provenance)
        previous = accepted[-1]
        parent_kind = "accepted_overlay"
        parent_candidate_id = previous.candidate.candidate_id
        parent_decision_receipt = previous.decision_receipt
        parent_locator = parent.image_id
    else:
        parent_kind = "initial_provenance"
        parent_candidate_id = None
        parent_decision_receipt = None
        parent_locator = parent.verified_repo_digest or ""
    return OverlayBuildReceipt(
        1,
        anchor_generation,
        candidate_id,
        candidate_source_sha256,
        parent_kind,
        parent.image_id,
        parent_locator,
        parent_candidate_id,
        parent_decision_receipt,
        (
            provenance.container.image_id,
            *(item.deployment.deployed_image_id or "" for item in accepted),
        ),
        tuple(str(item.candidate.candidate_id) for item in accepted),
        built.image.image_id,
        dockerfile_sha256,
        candidate_file_sha256,
        loaded.sha256,
    )


def _validate_accepted_stack(
    accepted: Sequence[AcceptedCandidate], provenance: RunProvenance
) -> None:
    base_id = provenance.container.image_id
    if base_id is None or not _IMAGE_ID.fullmatch(base_id):
        raise IntegrityError(
            "Initial image identity is unresolved", "overlay_ancestry_mismatch"
        )
    prior_images: list[str] = [base_id]
    prior_candidates: list[str] = []
    seen_images: set[str] = set()
    for generation, item in enumerate(accepted):
        candidate_id = item.candidate.candidate_id
        deployment = item.deployment
        receipt = _deployment_build_receipt(item)
        expected_parent = prior_images[-1]
        expected_kind = "initial_provenance" if generation == 0 else "accepted_overlay"
        expected_parent_candidate = prior_candidates[-1] if prior_candidates else None
        expected_parent_decision = (
            accepted[generation - 1].decision_receipt if generation else None
        )
        if (
            not candidate_id
            or not _SHA256.fullmatch(item.decision_receipt)
            or deployment.candidate_id != candidate_id
            or not deployment.qualified
            or deployment.validation_level is not ValidationLevel.RUNTIME_OVERLAY_VERIFIED
            or receipt.anchor_generation != generation
            or receipt.candidate_id != candidate_id
            or receipt.candidate_source_sha256 != item.candidate.candidate_source_sha256
            or receipt.parent_kind != expected_kind
            or receipt.parent_image_id != expected_parent
            or receipt.parent_candidate_id != expected_parent_candidate
            or receipt.parent_decision_receipt != expected_parent_decision
            or receipt.parent_ancestry_image_ids != tuple(prior_images)
            or receipt.parent_accepted_candidate_ids != tuple(prior_candidates)
            or receipt.derived_image_id != deployment.deployed_image_id
            or receipt.derived_image_id in seen_images
        ):
            raise IntegrityError(
                "Accepted overlay ancestry is inconsistent", "overlay_ancestry_mismatch"
            )
        allowed_initial_locators = {
            *provenance.container.repo_digests,
            provenance.container.requested_image,
        }
        if generation == 0 and receipt.parent_locator not in allowed_initial_locators:
            raise IntegrityError(
                "Initial overlay parent lost provenance", "overlay_ancestry_mismatch"
            )
        if generation and receipt.parent_locator != expected_parent:
            raise IntegrityError(
                "Derived overlay parent locator drifted", "overlay_ancestry_mismatch"
            )
        prior_candidates.append(candidate_id)
        prior_images.append(receipt.derived_image_id)
        seen_images.add(receipt.derived_image_id)


def _deployment_build_receipt(item: AcceptedCandidate) -> OverlayBuildReceipt:
    evidence = item.deployment.evidence
    raw = evidence.get("overlay_build_receipt")
    digest = evidence.get("overlay_build_receipt_sha256")
    if not isinstance(raw, Mapping) or not isinstance(digest, str):
        raise IntegrityError(
            "Accepted overlay lacks build evidence", "overlay_build_receipt_missing"
        )
    receipt = OverlayBuildReceipt.from_mapping(raw)
    if receipt.digest != digest:
        raise IntegrityError(
            "Accepted overlay build evidence drifted", "overlay_build_receipt_mismatch"
        )
    parent = evidence.get("parent_image")
    derived = evidence.get("derived_image")
    loaded = evidence.get("loaded_candidate")
    if (
        not isinstance(parent, Mapping)
        or not isinstance(derived, Mapping)
        or not isinstance(loaded, Mapping)
        or parent.get("image_id") != receipt.parent_image_id
        or derived.get("image_id") != receipt.derived_image_id
        or evidence.get("dockerfile_sha256") != receipt.dockerfile_sha256
        or evidence.get("candidate_file_sha256") != receipt.candidate_file_sha256
        or loaded.get("sha256") != receipt.loaded_candidate_sha256
    ):
        raise IntegrityError(
            "Accepted overlay build evidence is inconsistent",
            "overlay_build_receipt_mismatch",
        )
    return receipt


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        raise TypeError("expected string sequence")
    return tuple(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected optional string")
    return value


__all__ = [
    "OverlayBuildReceipt",
    "capture_overlay_build_receipt",
    "validate_accepted_overlay_parent",
]
