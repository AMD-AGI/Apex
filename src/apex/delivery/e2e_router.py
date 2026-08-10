"""Recipe-bound routing for independently reviewed E2E bundle verifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

from apex.core import ContractError, validate_identifier

from .e2e_bundle import load_and_verify_e2e_bundle
from .e2e_verify import BundleVerifyOutcome


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class E2EBundleVerificationPort(Protocol):
    """Common verifier interface used by the CLI and formal delivery profiles."""

    def verify(
        self,
        *,
        bundle_dir: Path,
        results_dir: Path,
        expected_digest: str | None = None,
        source_overrides: Mapping[str, Path] | None = None,
    ) -> BundleVerifyOutcome: ...


@dataclass(frozen=True, slots=True)
class E2EVerifierProfile:
    """Lazy verifier factory admitted for an exact frozen recipe set."""

    profile_id: str
    recipe_sha256s: frozenset[str]
    factory: Callable[[], E2EBundleVerificationPort]

    def __post_init__(self) -> None:
        validate_identifier(self.profile_id, field_name="E2E verifier profile ID")
        if not self.recipe_sha256s or any(
            not _DIGEST.fullmatch(value) for value in self.recipe_sha256s
        ):
            raise ContractError(
                "Verifier profile recipe digests are invalid",
                "invalid_e2e_verifier_profile",
            )
        if not callable(self.factory):
            raise ContractError(
                "Verifier profile factory is invalid",
                "invalid_e2e_verifier_profile",
            )


class E2EBundleVerifierRouter:
    """Select a reviewed verifier from the bundle's exact recipe digest.

    Static bundle verification happens before a concrete profile is composed.
    This keeps workload-specific Docker/source adapters lazy and prevents a
    reviewed vertical slice from becoming the default for unrelated bundles.
    """

    def __init__(self, profiles: tuple[E2EVerifierProfile, ...]) -> None:
        if not profiles:
            raise ContractError(
                "At least one E2E verifier profile is required",
                "invalid_e2e_verifier_profile",
            )
        routes: dict[str, E2EVerifierProfile] = {}
        ids: set[str] = set()
        for profile in profiles:
            if profile.profile_id in ids:
                raise ContractError(
                    "E2E verifier profile ID is duplicated",
                    "duplicate_e2e_verifier_profile",
                )
            ids.add(profile.profile_id)
            for digest in profile.recipe_sha256s:
                if digest in routes:
                    raise ContractError(
                        "E2E recipe is claimed by multiple verifier profiles",
                        "duplicate_e2e_verifier_recipe",
                    )
                routes[digest] = profile
        self._profiles = profiles
        self._routes = routes

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(profile.profile_id for profile in self._profiles)

    def verify(
        self,
        *,
        bundle_dir: Path,
        results_dir: Path,
        expected_digest: str | None = None,
        source_overrides: Mapping[str, Path] | None = None,
    ) -> BundleVerifyOutcome:
        if not bundle_dir.is_absolute() or not results_dir.is_absolute():
            raise ContractError(
                "Bundle verification paths must be absolute",
                "invalid_bundle_path",
            )
        candidate = load_and_verify_e2e_bundle(
            bundle_dir, expected_digest=expected_digest
        )
        recipe_sha256 = candidate.recipe.computed_sha256
        profile = self._routes.get(recipe_sha256)
        if profile is None:
            raise ContractError(
                "No reviewed verifier profile accepts this exact build recipe",
                "e2e_verifier_profile_unavailable",
                {"recipe_sha256": recipe_sha256},
            )
        verifier = profile.factory()
        return verifier.verify(
            bundle_dir=bundle_dir,
            results_dir=results_dir,
            expected_digest=expected_digest,
            source_overrides=source_overrides,
        )


__all__ = [
    "E2EBundleVerificationPort",
    "E2EBundleVerifierRouter",
    "E2EVerifierProfile",
]
