"""Trusted composition contracts for formal E2E source delivery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import ContractError
from apex.delivery import (
    BuildRecipeLock,
    BundleProvenanceLock,
    DerivedImageIdentity,
    SourceRepositoryLock,
)
from apex.evaluation import E2EMeasurement

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

    def __post_init__(self) -> None:
        if self.repository_id not in {"vllm", "aiter"}:
            raise ContractError(
                "Formal Python/Triton delivery supports vllm or aiter",
                "unsupported_delivery",
            )
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
    baseline: E2EMeasurement
    overlay_final: E2EMeasurement
    artifact_root: Path


@dataclass(frozen=True, slots=True)
class PrimarySourceBuildOutput:
    """Evaluator-owned evidence from the first clean source-built environment."""

    environment_id: str
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
