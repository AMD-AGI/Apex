"""Immutable finalization of independently verified E2E bundles."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from apex.core import ContractError, IntegrityError, canonical_json_bytes

from .e2e_bundle_common import (
    SECOND_RECEIPT_ROLE,
    E2EPatchBundle,
    compute_e2e_bundle_digest,
    file_entry,
    fsync_dir,
    write_new,
)
from .e2e_bundle_load import load_and_verify_e2e_bundle
from .e2e_receipts import DeliveryVerificationResult


def finalize_verified_e2e_bundle(
    candidate: E2EPatchBundle,
    *,
    verification: DeliveryVerificationResult,
    destination: Path,
) -> E2EPatchBundle:
    """Create a new immutable final bundle carrying the independent receipt."""

    if candidate.verified:
        raise ContractError("Bundle is already finalized", "bundle_already_verified")
    if not verification.verified or verification.bundle_digest != candidate.digest:
        raise ContractError(
            "Verification result cannot finalize this bundle",
            "verification_failed",
        )
    if destination.exists():
        raise ContractError("Final bundle destination exists", "bundle_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".apex-final-bundle-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "bundle"
        shutil.copytree(candidate.path, staging, symlinks=False)
        manifest = dict(candidate.manifest)
        receipt_path = "verification/clean_replay.receipt.json"
        receipt = canonical_json_bytes(verification.to_dict()) + b"\n"
        write_new(staging / receipt_path, receipt)
        entry = file_entry(receipt_path, SECOND_RECEIPT_ROLE, receipt)
        manifest["files"] = [*manifest["files"], entry]
        manifest["verified"] = True
        manifest["verification_receipt"] = {
            "path": receipt_path,
            "sha256": entry["sha256"],
        }
        if compute_e2e_bundle_digest(manifest, staging) != candidate.digest:
            raise IntegrityError(
                "Final receipt changed the signed delivery scope",
                "bundle_digest_mismatch",
            )
        (staging / "bundle.json").unlink()
        write_new(staging / "bundle.json", canonical_json_bytes(manifest) + b"\n")
        os.replace(staging, destination)
        fsync_dir(destination.parent)
    return load_and_verify_e2e_bundle(destination, expected_digest=candidate.digest)


__all__ = ["finalize_verified_e2e_bundle"]
