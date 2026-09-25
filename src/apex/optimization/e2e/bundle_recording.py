"""Portable final-bundle evidence for formally verified E2E deliveries."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from apex.delivery import (
    PortableBundleEvidence,
    capture_portable_bundle,
    e2e_reproduction_declaration,
    load_and_verify_e2e_bundle,
)
from apex.storage import ArtifactStore

from .services import FinalDeliveryResult


@dataclass(frozen=True, slots=True)
class FinalDeliveryBundleCapture:
    portable: PortableBundleEvidence
    replication: dict[str, object]

    def artifact_bindings(self) -> tuple[dict[str, object], ...]:
        return self.portable.artifact_bindings()


def capture_final_delivery_bundle(
    result: FinalDeliveryResult, artifacts: ArtifactStore
) -> FinalDeliveryBundleCapture | None:
    """Capture only official directory bundles; absent evidence cannot qualify."""

    if not result.verified or result.bundle_path is None or result.bundle_digest is None:
        return None
    path = Path(result.bundle_path)
    if not path.is_dir() or path.is_symlink():
        return None
    portable = capture_portable_bundle(
        artifacts,
        path,
        bundle_kind="e2e",
        expected_digest=result.bundle_digest,
    )
    bundle = load_and_verify_e2e_bundle(path, expected_digest=result.bundle_digest)
    return FinalDeliveryBundleCapture(
        portable,
        e2e_reproduction_declaration(bundle, portable),
    )


__all__ = ["FinalDeliveryBundleCapture", "capture_final_delivery_bundle"]
