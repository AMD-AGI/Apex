"""Atomic assembly of candidate E2E source patch bundles."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_identifier,
)

from .e2e_bundle_common import (
    CONFIG_PATHS,
    PRIMARY_RECEIPT_PATHS,
    SCHEMA,
    E2EPatchBundle,
    compute_e2e_bundle_digest,
    file_entry,
    fsync_dir,
    verify_replay_config_invariants,
    write_new,
)
from .e2e_models import BuildRecipeLock, BundleProvenanceLock, DerivedImageIdentity
from .e2e_receipts import PrimaryVerificationEvidence, source_stack_digest
from .git_patch import CapturedRepositoryPatch, validate_lock_order


def build_e2e_patch_bundle(
    *,
    bundle_id: str,
    bundle_dir: Path,
    repositories: Sequence[CapturedRepositoryPatch],
    recipe: BuildRecipeLock,
    derived_image: DerivedImageIdentity,
    provenance: BundleProvenanceLock,
    configs: Mapping[str, Path],
    primary_evidence: PrimaryVerificationEvidence,
    primary_receipts: Mapping[str, Path],
    image_sbom: Path,
) -> E2EPatchBundle:
    """Atomically assemble an unverified source bundle from exact inputs."""

    ordered, patches, stack_digest = _validate_inputs(
        bundle_id=bundle_id,
        bundle_dir=bundle_dir,
        repositories=repositories,
        recipe=recipe,
        derived_image=derived_image,
        provenance=provenance,
        configs=configs,
        primary_evidence=primary_evidence,
        primary_receipts=primary_receipts,
    )
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".apex-e2e-bundle-", dir=bundle_dir.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        entries: list[dict[str, str]] = []
        _write_sources(staging, entries, ordered, patches)
        _write_configs(staging, entries, configs)
        _write_primary_receipts(staging, entries, primary_receipts, primary_evidence)
        _write_fixed_metadata(staging, entries, recipe, derived_image, primary_evidence)
        _write_sbom(staging, entries, image_sbom, derived_image)
        manifest = _write_provenance_manifest(
            staging,
            entries,
            bundle_id=bundle_id,
            ordered=ordered,
            stack_digest=stack_digest,
            provenance=provenance,
            derived_image=derived_image,
        )
        write_new(staging / "bundle.json", canonical_json_bytes(manifest) + b"\n")
        _sync_and_publish(staging, bundle_dir)
    from .e2e_bundle_load import load_and_verify_e2e_bundle

    return load_and_verify_e2e_bundle(bundle_dir)


def _validate_inputs(
    *,
    bundle_id: str,
    bundle_dir: Path,
    repositories: Sequence[CapturedRepositoryPatch],
    recipe: BuildRecipeLock,
    derived_image: DerivedImageIdentity,
    provenance: BundleProvenanceLock,
    configs: Mapping[str, Path],
    primary_evidence: PrimaryVerificationEvidence,
    primary_receipts: Mapping[str, Path],
):
    validate_identifier(bundle_id, field_name="bundle_id")
    if bundle_dir.exists():
        raise ContractError(f"Bundle directory already exists: {bundle_dir}", "bundle_exists")
    ordered = validate_lock_order(tuple(item.lock for item in repositories))
    by_id = {item.lock.repository_id: item for item in repositories}
    if tuple(by_id[item.repository_id].lock for item in ordered) != ordered:
        raise ContractError("Repository patch identities are ambiguous", "invalid_source_lock")
    if any(item.build_recipe_sha256 != recipe.computed_sha256 for item in ordered):
        raise IntegrityError("Source locks bind a different build recipe", "build_recipe_drift")
    if derived_image.parent_digest != recipe.parent_image_digest:
        raise IntegrityError(
            "Derived image parent differs from build recipe",
            "derived_image_parent_mismatch",
        )
    if provenance.baseline_image_digest != recipe.parent_image_digest:
        raise IntegrityError(
            "Provenance parent image differs from build recipe",
            "derived_image_parent_mismatch",
        )
    stack = source_stack_digest(ordered)
    if primary_evidence.source_stack_sha256 != stack:
        raise IntegrityError(
            "Primary evidence binds a different source stack",
            "candidate_lineage_mismatch",
        )
    _validate_primary_evidence(primary_evidence, configs, primary_receipts)
    return ordered, by_id, stack


def _validate_primary_evidence(
    evidence: PrimaryVerificationEvidence,
    configs: Mapping[str, Path],
    receipts: Mapping[str, Path],
) -> None:
    gates = (
        evidence.engagement_verified,
        evidence.normal_runtime_measurement,
        evidence.accuracy_passed,
        evidence.latency_gates_passed,
        evidence.objective_improved,
    )
    if not all(gates):
        raise ContractError(
            "Primary promotion evidence is not a verified winner",
            "primary_verification_failed",
        )
    if set(configs) != set(CONFIG_PATHS):
        raise ContractError(
            "Bundle requires all four benchmark config views",
            "missing_bundle_config",
        )
    required = {
        "primary_build_receipt",
        "primary_engagement_receipt",
        "primary_benchmark_receipt",
    }
    if not required.issubset(receipts) or not set(receipts).issubset(PRIMARY_RECEIPT_PATHS):
        raise ContractError(
            "Primary verification receipts are incomplete",
            "missing_primary_receipt",
        )
    if evidence.safety_source_sha256 is not None and "primary_safety_receipt" not in receipts:
        raise ContractError(
            "Safety lineage lacks its primary receipt",
            "missing_primary_receipt",
        )


def _write_sources(staging, entries, ordered, patches) -> None:
    locks_content = canonical_json_bytes(
        {"schema_version": 1, "repositories": [item.to_dict() for item in ordered]}
    ) + b"\n"
    write_new(staging / "sources.lock.json", locks_content)
    entries.append(file_entry("sources.lock.json", "source_locks", locks_content))
    for lock in ordered:
        captured = patches[lock.repository_id]
        if sha256_bytes(captured.content) != lock.patch_sha256:
            raise IntegrityError(
                "Captured patch bytes differ from source lock",
                "bundle_patch_digest_mismatch",
            )
        write_new(staging / lock.patch_path, captured.content)
        entries.append(
            file_entry(
                lock.patch_path,
                f"source_patch:{lock.repository_id}",
                captured.content,
            )
        )


def _write_configs(staging, entries, configs: Mapping[str, Path]) -> None:
    for role, destination in CONFIG_PATHS.items():
        source = configs[role]
        if source.is_symlink() or not source.is_file():
            raise IntegrityError(f"Config is missing or unsafe: {source}", "invalid_bundle_config")
        content = source.read_bytes()
        write_new(staging / destination, content)
        entries.append(file_entry(destination, role, content))


def _write_primary_receipts(
    staging,
    entries,
    receipts: Mapping[str, Path],
    evidence: PrimaryVerificationEvidence,
) -> None:
    hashes: dict[str, str] = {}
    for role, destination in PRIMARY_RECEIPT_PATHS.items():
        source = receipts.get(role)
        if source is None:
            continue
        if source.is_symlink() or not source.is_file():
            raise IntegrityError(
                f"Primary receipt is missing or unsafe: {source}",
                "missing_primary_receipt",
            )
        content = source.read_bytes()
        write_new(staging / destination, content)
        entries.append(file_entry(destination, role, content))
        hashes[role] = sha256_bytes(content)
    expected = {
        "primary_build_receipt": evidence.build_receipt_sha256,
        "primary_engagement_receipt": evidence.engagement_receipt_sha256,
        "primary_benchmark_receipt": evidence.benchmark_receipt_sha256,
    }
    if evidence.safety_receipt_sha256 is not None:
        expected["primary_safety_receipt"] = evidence.safety_receipt_sha256
    if any(hashes.get(role) != digest for role, digest in expected.items()):
        raise IntegrityError(
            "Primary receipt bytes differ from evidence summary",
            "primary_receipt_mismatch",
        )


def _write_fixed_metadata(staging, entries, recipe, image, evidence) -> None:
    documents = (
        ("build/recipe.lock.json", "build_recipe", recipe.to_dict()),
        ("build/derived_image.json", "derived_image", image.to_dict()),
        ("verification/primary.evidence.json", "primary_verification", evidence.to_dict()),
    )
    for path, role, document in documents:
        content = canonical_json_bytes(document) + b"\n"
        write_new(staging / path, content)
        entries.append(file_entry(path, role, content))


def _write_sbom(staging, entries, image_sbom: Path, image: DerivedImageIdentity) -> None:
    if image_sbom.is_symlink() or not image_sbom.is_file():
        raise IntegrityError("Derived image SBOM is missing or unsafe", "missing_image_sbom")
    content = image_sbom.read_bytes()
    if sha256_bytes(content) != image.sbom_sha256:
        raise IntegrityError("Derived image SBOM digest mismatch", "image_sbom_mismatch")
    write_new(staging / "build/sbom.json", content)
    entries.append(file_entry("build/sbom.json", "derived_image_sbom", content))


def _write_provenance_manifest(
    staging,
    entries,
    *,
    bundle_id,
    ordered,
    stack_digest,
    provenance,
    derived_image,
) -> dict[str, Any]:
    _, _, semantics_sha = verify_replay_config_invariants(
        staging / CONFIG_PATHS["benchmark_measurement"],
        staging / CONFIG_PATHS["benchmark_replay"],
        expected_image_locator=derived_image.locator,
    )
    original = staging / CONFIG_PATHS["benchmark_original"]
    if (
        sha256_file(original) != provenance.original_config_sha256
        or semantics_sha != provenance.workload_semantics_sha256
    ):
        raise IntegrityError(
            "Delivery provenance does not bind benchmark configs",
            "benchmark_provenance_mismatch",
        )
    content = canonical_json_bytes(provenance.to_dict()) + b"\n"
    write_new(staging / "provenance.lock.json", content)
    entries.append(file_entry("provenance.lock.json", "delivery_provenance", content))
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "bundle_id": bundle_id,
        "source_stack_sha256": stack_digest,
        "repositories": [item.repository_id for item in ordered],
        "files": entries,
        "verified": False,
        "verification_receipt": None,
    }
    manifest["bundle_digest"] = compute_e2e_bundle_digest(manifest, staging)
    return manifest


def _sync_and_publish(staging: Path, destination: Path) -> None:
    directories = sorted(
        (item for item in staging.rglob("*") if item.is_dir()), reverse=True
    )
    for directory in directories:
        fsync_dir(directory)
    fsync_dir(staging)
    os.replace(staging, destination)
    fsync_dir(destination.parent)


__all__ = ["build_e2e_patch_bundle"]
