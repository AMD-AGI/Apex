"""Public facade for canonical E2E patch bundle contracts."""

from .e2e_bundle_build import build_e2e_patch_bundle
from .e2e_bundle_common import (
    E2EPatchBundle,
    compute_e2e_bundle_digest,
    detect_bundle_kind,
    verify_replay_config_invariants,
)
from .e2e_bundle_finalize import finalize_verified_e2e_bundle
from .e2e_bundle_load import load_and_verify_e2e_bundle


__all__ = [
    "E2EPatchBundle",
    "build_e2e_patch_bundle",
    "compute_e2e_bundle_digest",
    "detect_bundle_kind",
    "finalize_verified_e2e_bundle",
    "load_and_verify_e2e_bundle",
    "verify_replay_config_invariants",
]
