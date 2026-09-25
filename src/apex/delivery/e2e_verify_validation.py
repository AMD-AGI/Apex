"""Pure validation for independently produced E2E delivery receipts."""

from __future__ import annotations

from pathlib import Path

from apex.core import IntegrityError, sha256_bytes, sha256_json

from .e2e_bundle import E2EPatchBundle
from .e2e_receipts import (
    CleanReplayReceipt,
    LoadedByteEngagementReceipt,
    ReplayConfigInvariantReceipt,
    SourceBuildReceipt,
)
from .git_patch import RepositoryApplyReceipt


def validate_build_receipt(
    bundle: E2EPatchBundle, receipt: SourceBuildReceipt
) -> None:
    if (
        receipt.bundle_digest != bundle.digest
        or receipt.recipe_sha256 != bundle.recipe.computed_sha256
        or receipt.expected_parent_digest != bundle.derived_image.parent_digest
        or receipt.observed_parent_digest != bundle.derived_image.parent_digest
        or receipt.expected_image_digest != bundle.derived_image.image_digest
        or receipt.observed_image_digest != bundle.derived_image.image_digest
        or receipt.expected_sbom_sha256 != bundle.derived_image.sbom_sha256
        or receipt.observed_sbom_sha256 != bundle.derived_image.sbom_sha256
        or receipt.source_stack_sha256
        != bundle.primary_evidence.source_stack_sha256
        or not receipt.verified
    ):
        raise IntegrityError(
            "Source build receipt does not match bundle",
            "source_build_receipt_mismatch",
        )
    changed = {item.runtime_component for item in bundle.repositories}
    if not changed.issubset({item.component for item in receipt.artifacts}):
        raise IntegrityError(
            "Build receipt does not identify every changed component",
            "source_build_receipt_mismatch",
        )
    if len(receipt.step_receipts) != len(bundle.recipe.steps):
        raise IntegrityError(
            "Build receipt is missing fixed recipe steps",
            "source_build_receipt_mismatch",
        )
    for index, (observed, expected) in enumerate(
        zip(receipt.step_receipts, bundle.recipe.steps, strict=True)
    ):
        if (
            observed.index != index
            or observed.repository_id != expected.repository_id
            or observed.cwd != expected.cwd
            or observed.argv_sha256 != sha256_json(list(expected.argv))
            or not observed.verified
        ):
            raise IntegrityError(
                "Build step receipt differs from recipe",
                "source_build_receipt_mismatch",
            )


def validate_engagement_receipt(
    bundle: E2EPatchBundle,
    build: SourceBuildReceipt,
    receipt: LoadedByteEngagementReceipt,
) -> None:
    if (
        receipt.bundle_digest != bundle.digest
        or receipt.image_digest != bundle.derived_image.image_digest
        or receipt.source_stack_sha256
        != bundle.primary_evidence.source_stack_sha256
        or not receipt.verified
    ):
        raise IntegrityError(
            "Runtime engagement does not match bundle",
            "loaded_byte_engagement_failed",
        )
    built = {
        (item.component, item.runtime_path, item.sha256, item.build_id)
        for item in build.artifacts
    }
    loaded = {
        (
            item.component,
            item.runtime_path,
            item.expected_sha256,
            item.expected_build_id,
        )
        for item in receipt.artifacts
        if item.verified
    }
    if built != loaded or len(loaded) != len(receipt.artifacts):
        raise IntegrityError(
            "Runtime loaded old or unexpected bytes",
            "loaded_byte_engagement_failed",
        )
    policies = _component_engagement_policies(bundle)
    for artifact in receipt.artifacts:
        policy = policies.get(artifact.component)
        if (
            policy is None
            or artifact.engagement_kind != policy[0]
            or policy[1]
            and not (artifact.expected_build_id and artifact.observed_build_id)
        ):
            raise IntegrityError(
                "Runtime engagement violates the component capability",
                "loaded_byte_engagement_failed",
            )


def _component_engagement_policies(
    bundle: E2EPatchBundle,
) -> dict[str, tuple[str, bool]]:
    policies: dict[str, tuple[str, bool]] = {}
    for repository in bundle.repositories:
        current = (repository.engagement_kind, repository.build_id_required)
        previous = policies.setdefault(repository.runtime_component, current)
        if previous != current:
            raise IntegrityError(
                "Repositories disagree on runtime engagement capability",
                "invalid_source_lock",
            )
    return policies


def validate_replay_receipt(
    bundle: E2EPatchBundle,
    config: ReplayConfigInvariantReceipt,
    receipt: CleanReplayReceipt,
    repositories: tuple[RepositoryApplyReceipt, ...],
    evidence_root: Path,
) -> None:
    if (
        receipt.bundle_digest != bundle.digest
        or receipt.primary_environment_id != bundle.primary_evidence.environment_id
        or receipt.image_digest != bundle.derived_image.image_digest
        or receipt.replay_config_sha256 != config.replay_config_sha256
        or receipt.source_stack_sha256
        != bundle.primary_evidence.source_stack_sha256
        or receipt.source_materialization_sha256
        != sha256_json([item.to_dict() for item in repositories])
        or receipt.primary_runtime_identity_sha256
        != bundle.primary_evidence.runtime_identity_sha256
        or not receipt.verified
    ):
        raise IntegrityError(
            "Second clean replay does not match bundle",
            "second_clean_replay_failed",
        )
    root = evidence_root.resolve(strict=True)
    for artifact in receipt.raw_artifacts:
        _validate_raw_artifact(root, artifact)


def _validate_raw_artifact(root: Path, artifact: object) -> None:
    candidate = root / artifact.relative_path  # type: ignore[attr-defined]
    if candidate.is_symlink():
        raise IntegrityError(
            "Raw replay artifact is a symlink", "second_clean_replay_failed"
        )
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise IntegrityError(
            "Raw replay artifact escapes its evidence root",
            "second_clean_replay_failed",
        ) from error
    if (
        not path.is_file()
        or path.stat().st_size != artifact.size_bytes  # type: ignore[attr-defined]
        or sha256_bytes(path.read_bytes()) != artifact.sha256  # type: ignore[attr-defined]
    ):
        raise IntegrityError(
            "Raw replay artifact differs from its receipt",
            "second_clean_replay_failed",
        )


__all__ = [
    "validate_build_receipt",
    "validate_engagement_receipt",
    "validate_replay_receipt",
]
