"""Fail-closed static loading and verification of E2E bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, sha256_file, validate_identifier

from .e2e_bundle_common import (
    CONFIG_PATHS,
    PRIMARY_RECEIPT_PATHS,
    SCHEMA,
    SECOND_RECEIPT_ROLE,
    E2EPatchBundle,
    compute_e2e_bundle_digest,
    read_json,
    role_paths,
    verify_replay_config_invariants,
)
from .e2e_bundle_validation import validate_final_verification_mapping
from .e2e_models import (
    BuildRecipeLock,
    BundleProvenanceLock,
    DerivedImageIdentity,
    SourceRepositoryLock,
    safe_bundle_path,
)
from .e2e_receipts import PrimaryVerificationEvidence, source_stack_digest
from .git_patch import validate_lock_order


def load_and_verify_e2e_bundle(
    bundle_dir: Path,
    *,
    expected_digest: str | None = None,
) -> E2EPatchBundle:
    """Fail closed on tree, hash, source, recipe, config, and verdict tampering."""

    root, manifest, digest, roles = _load_manifest_tree(bundle_dir, expected_digest)
    locks, recipe, image, primary, provenance = _load_contracts(roles)
    _validate_source_lineage(
        root,
        manifest,
        roles,
        locks,
        recipe,
        image,
        primary,
        provenance,
    )
    config_paths = _validate_configs(roles, image, provenance)
    verified, result = _load_terminal_verdict(
        root,
        manifest,
        roles,
        digest,
        locks,
        recipe,
        image,
        primary,
        config_paths,
    )
    return E2EPatchBundle(
        path=root,
        bundle_id=str(manifest["bundle_id"]),
        digest=digest,
        verified=verified,
        manifest=manifest,
        repositories=locks,
        recipe=recipe,
        derived_image=image,
        provenance=provenance,
        primary_evidence=primary,
        config_paths=config_paths,
        primary_receipt_paths={
            role: path for role, path in roles.items() if role in PRIMARY_RECEIPT_PATHS
        },
        sbom_path=roles["derived_image_sbom"],
        verification_result=result,
    )


def _load_manifest_tree(bundle_dir: Path, expected_digest: str | None):
    root = bundle_dir.resolve(strict=True)
    if not root.is_dir() or bundle_dir.is_symlink():
        raise IntegrityError("E2E bundle root is unsafe", "invalid_e2e_bundle")
    manifest_path = root / "bundle.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise IntegrityError("E2E bundle manifest is unsafe", "invalid_e2e_bundle")
    manifest = read_json(manifest_path, "invalid_e2e_bundle")
    if manifest.get("schema") != SCHEMA:
        raise IntegrityError("Unsupported E2E bundle schema", "invalid_e2e_bundle")
    try:
        validate_identifier(str(manifest.get("bundle_id", "")), field_name="bundle_id")
    except ContractError as error:
        raise IntegrityError(error.message, error.reason_code, error.details) from error
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise IntegrityError("Bundle files are missing", "invalid_e2e_bundle")
    declared = {"bundle.json"}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise IntegrityError("Bundle file entry is invalid", "invalid_e2e_bundle")
        relative = safe_bundle_path(str(entry.get("path", "")))
        if relative in declared:
            raise IntegrityError("Bundle declares a duplicate path", "invalid_e2e_bundle")
        declared.add(relative)
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise IntegrityError("Bundle file is missing or unsafe", "bundle_file_set_mismatch")
        if sha256_file(target) != str(entry.get("sha256", "")).removeprefix("sha256:"):
            raise IntegrityError(
                f"Bundle file digest mismatch: {relative}",
                "bundle_file_digest_mismatch",
            )
    actual = _actual_file_set(root)
    if actual != declared:
        raise IntegrityError(
            "Bundle contains missing or undeclared files",
            "bundle_file_set_mismatch",
        )
    digest = compute_e2e_bundle_digest(manifest, root)
    recorded = str(manifest.get("bundle_digest", "")).removeprefix("sha256:")
    expected = expected_digest.removeprefix("sha256:") if expected_digest else None
    if digest != recorded or expected is not None and digest != expected:
        raise IntegrityError("E2E bundle digest mismatch", "bundle_digest_mismatch")
    return root, manifest, digest, role_paths(manifest, root)


def _actual_file_set(root: Path) -> set[str]:
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise IntegrityError("Bundle may not contain symlinks", "bundle_symlink")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    return actual


def _load_contracts(roles: Mapping[str, Path]):
    required = {
        "source_locks",
        "build_recipe",
        "derived_image",
        "primary_verification",
        "delivery_provenance",
        "derived_image_sbom",
        "primary_build_receipt",
        "primary_engagement_receipt",
        "primary_benchmark_receipt",
        *CONFIG_PATHS.keys(),
    }
    if not required.issubset(roles):
        raise IntegrityError(
            "E2E bundle is missing required artifacts",
            "bundle_file_set_mismatch",
        )
    locks_value = read_json(roles["source_locks"], "invalid_source_lock")
    if locks_value.get("schema_version") != 1 or not isinstance(
        locks_value.get("repositories"), list
    ):
        raise IntegrityError("Source lock file is invalid", "invalid_source_lock")
    try:
        locks = validate_lock_order(
            tuple(
                SourceRepositoryLock.from_mapping(item)
                for item in locks_value["repositories"]
            )
        )
        recipe = BuildRecipeLock.from_mapping(
            read_json(roles["build_recipe"], "invalid_build_recipe")
        )
        image = DerivedImageIdentity.from_mapping(
            read_json(roles["derived_image"], "invalid_image_identity")
        )
        primary = PrimaryVerificationEvidence.from_mapping(
            read_json(roles["primary_verification"], "invalid_primary_evidence")
        )
        provenance = BundleProvenanceLock.from_mapping(
            read_json(roles["delivery_provenance"], "invalid_delivery_provenance")
        )
    except ContractError as error:
        raise IntegrityError(error.message, error.reason_code, error.details) from error
    return locks, recipe, image, primary, provenance


def _validate_source_lineage(
    root,
    manifest,
    roles,
    locks,
    recipe,
    image,
    primary,
    provenance,
) -> None:
    if list(manifest.get("repositories", ())) != [item.repository_id for item in locks]:
        raise IntegrityError(
            "Manifest repository order differs from source lock",
            "invalid_patch_order",
        )
    if any(item.build_recipe_sha256 != recipe.computed_sha256 for item in locks):
        raise IntegrityError("Source lock build recipe drift", "build_recipe_drift")
    if image.parent_digest != recipe.parent_image_digest:
        raise IntegrityError(
            "Derived image parent differs from recipe",
            "derived_image_parent_mismatch",
        )
    if sha256_file(roles["derived_image_sbom"]) != image.sbom_sha256:
        raise IntegrityError("Derived image SBOM digest mismatch", "image_sbom_mismatch")
    if provenance.baseline_image_digest != recipe.parent_image_digest:
        raise IntegrityError(
            "Delivery provenance parent differs from recipe",
            "derived_image_parent_mismatch",
        )
    stack = source_stack_digest(locks)
    if stack != manifest.get("source_stack_sha256") or stack != primary.source_stack_sha256:
        raise IntegrityError("Source stack digest mismatch", "candidate_lineage_mismatch")
    _validate_primary_receipts(roles, primary)
    for lock in locks:
        patch_role = f"source_patch:{lock.repository_id}"
        if (
            roles.get(patch_role) != root / lock.patch_path
            or sha256_file(root / lock.patch_path) != lock.patch_sha256
        ):
            raise IntegrityError(
                "Patch/source-lock mismatch",
                "bundle_patch_digest_mismatch",
            )


def _validate_primary_receipts(roles, primary) -> None:
    valid = all(
        (
            sha256_file(roles["primary_build_receipt"])
            == primary.build_receipt_sha256,
            sha256_file(roles["primary_engagement_receipt"])
            == primary.engagement_receipt_sha256,
            sha256_file(roles["primary_benchmark_receipt"])
            == primary.benchmark_receipt_sha256,
        )
    )
    if primary.safety_source_sha256 is not None:
        path = roles.get("primary_safety_receipt")
        valid = (
            valid
            and path is not None
            and sha256_file(path) == primary.safety_receipt_sha256
        )
    if not valid:
        raise IntegrityError(
            "Primary verification receipt mismatch",
            "primary_receipt_mismatch",
        )


def _validate_configs(roles, image, provenance) -> dict[str, Path]:
    paths = {key: roles[key] for key in CONFIG_PATHS}
    _, _, semantics_sha = verify_replay_config_invariants(
        paths["benchmark_measurement"],
        paths["benchmark_replay"],
        expected_image_locator=image.locator,
    )
    if (
        sha256_file(paths["benchmark_original"])
        != provenance.original_config_sha256
        or semantics_sha != provenance.workload_semantics_sha256
    ):
        raise IntegrityError(
            "Benchmark config provenance mismatch",
            "benchmark_provenance_mismatch",
        )
    return paths


def _load_terminal_verdict(
    root,
    manifest,
    roles,
    digest,
    locks,
    recipe,
    image,
    primary,
    configs,
) -> tuple[bool, Mapping[str, Any] | None]:
    verified = bool(manifest.get("verified", False))
    declaration = manifest.get("verification_receipt")
    if not verified:
        if declaration is not None or SECOND_RECEIPT_ROLE in roles:
            raise IntegrityError(
                "Unverified bundle contains a final verification claim",
                "invalid_bundle_verdict",
            )
        return False, None
    receipt_path = roles.get(SECOND_RECEIPT_ROLE)
    if not isinstance(declaration, Mapping) or receipt_path is None:
        raise IntegrityError(
            "Verified bundle lacks a second-clean-replay receipt",
            "missing_clean_replay_receipt",
        )
    if receipt_path != root / safe_bundle_path(str(declaration.get("path", ""))):
        raise IntegrityError(
            "Verification receipt declaration differs",
            "invalid_clean_replay_receipt",
        )
    if sha256_file(receipt_path) != str(declaration.get("sha256", "")):
        raise IntegrityError(
            "Verification receipt digest mismatch",
            "invalid_clean_replay_receipt",
        )
    result = read_json(receipt_path, "invalid_clean_replay_receipt")
    validate_final_verification_mapping(
        result, digest, locks, recipe, image, primary, configs
    )
    return True, result


__all__ = ["load_and_verify_e2e_bundle"]
