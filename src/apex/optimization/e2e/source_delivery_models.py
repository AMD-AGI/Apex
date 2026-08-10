"""Trusted composition contracts for formal E2E source delivery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import ContractError, validate_identifier
from apex.delivery import (
    BuildRecipeLock,
    BundleProvenanceLock,
    DerivedImageIdentity,
    SourceRepositoryLock,
)
from apex.delivery.e2e_models import SourceComponentCapability
from apex.evaluation import E2EAcceptancePolicy, E2EObservation

from .services import FinalDeliveryRequest


@dataclass(frozen=True, slots=True)
class FormalRepositoryProfile:
    """Controller-owned source/build policy for one changed repository."""

    repository_id: str
    runtime_component: str
    trusted_url: str
    editable_allowlist: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    license_id: str = "Apache-2.0"
    source_languages: tuple[str, ...] = ("python", "triton")
    engagement_kind: str = "python_import"
    build_id_required: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.repository_id, field_name="repository_id")
        validate_identifier(self.runtime_component, field_name="runtime_component")
        if (
            not self.runtime_component.strip()
            or not self.trusted_url.strip()
            or not self.editable_allowlist
            or not self.license_id.strip()
        ):
            raise ContractError(
                "Formal repository profile is incomplete",
                "invalid_source_delivery_profile",
            )
        if (
            not self.source_languages
            or len(set(self.source_languages)) != len(self.source_languages)
            or any(item not in {"python", "triton"} for item in self.source_languages)
            or self.engagement_kind
            not in {"python_import", "process_map", "linker_build_id"}
            or self.build_id_required
            and self.engagement_kind == "python_import"
        ):
            raise ContractError(
                "Formal component engagement capability is invalid",
                "invalid_source_delivery_profile",
            )

    @property
    def component_capability(self) -> SourceComponentCapability:
        return SourceComponentCapability(
            self.repository_id,
            self.runtime_component,
            self.engagement_kind,
            self.build_id_required,
        )


@dataclass(frozen=True, slots=True)
class FormalSourceDeliveryProfile:
    """One exact parent/repository-set/fixed-recipe trust decision."""

    profile_id: str
    repositories: tuple[FormalRepositoryProfile, ...]
    recipe: BuildRecipeLock

    def __post_init__(self) -> None:
        identities = tuple(item.repository_id for item in self.repositories)
        if not self.profile_id or not identities or len(set(identities)) != len(identities):
            raise ContractError(
                "Formal source delivery profile is ambiguous",
                "invalid_source_delivery_profile",
            )
        known = set(identities)
        stepped = {step.repository_id for step in self.recipe.steps}
        if stepped != known:
            raise ContractError(
                "Every changed repository needs a fixed build step",
                "invalid_source_delivery_profile",
            )
        for repository in self.repositories:
            if any(item not in known for item in repository.dependencies):
                raise ContractError(
                    "Source delivery profile has an unknown dependency",
                    "invalid_source_delivery_profile",
                )

    @property
    def repository_ids(self) -> frozenset[str]:
        return frozenset(item.repository_id for item in self.repositories)

    @property
    def component_capabilities(self) -> tuple[SourceComponentCapability, ...]:
        return tuple(item.component_capability for item in self.repositories)


@dataclass(frozen=True, slots=True)
class PrimarySourceBuildRequest:
    """Inputs to a trusted primary source build and unchanged E2E validation."""

    run_id: str
    source_stack_sha256: str
    recipe: BuildRecipeLock
    repository_roots: Mapping[str, Path]
    repository_locks: tuple[SourceRepositoryLock, ...]
    benchmark_original: Path
    benchmark_measurement: Path
    benchmark_diagnostic: Path
    benchmark_replay: Path
    baseline: E2EObservation
    overlay_final: E2EObservation
    acceptance_policy: E2EAcceptancePolicy
    artifact_root: Path


@dataclass(frozen=True, slots=True)
class PrimarySourceBuildOutput:
    """Evaluator-owned evidence from the first clean source-built environment."""

    environment_id: str
    runtime_identity_sha256: str
    source_stack_sha256: str
    derived_image: DerivedImageIdentity
    image_sbom: Path
    benchmark_measurement: Path
    benchmark_diagnostic: Path
    benchmark_replay: Path
    primary_receipts: Mapping[str, Path]
    engagement_verified: bool
    normal_runtime_measurement: bool
    accuracy_passed: bool
    latency_gates_passed: bool
    objective_improved: bool
    overlay_rebuild_parity_passed: bool
    safety_certified: bool = False


class PrimarySourceBuildPort(Protocol):
    """Build, engage and benchmark a cumulative stack from clean source roots."""

    def build_and_validate(
        self, request: PrimarySourceBuildRequest
    ) -> PrimarySourceBuildOutput: ...


class DeliveryProvenancePort(Protocol):
    """Create the exact policy/model provenance lock for this formal delivery."""

    def lock(self, request: FinalDeliveryRequest) -> BundleProvenanceLock: ...


__all__ = [
    "DeliveryProvenancePort",
    "FormalRepositoryProfile",
    "FormalSourceDeliveryProfile",
    "PrimarySourceBuildOutput",
    "PrimarySourceBuildPort",
    "PrimarySourceBuildRequest",
]
