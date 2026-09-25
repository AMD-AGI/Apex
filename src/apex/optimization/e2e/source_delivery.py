"""Formal cumulative source delivery through an independent clean verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from apex.core import (
    ApexError,
    ContractError,
    IntegrityError,
    TaskStatus,
    ValidationLevel,
    sha256_file,
)
from apex.delivery import (
    E2EBundleVerifier,
    PrimaryVerificationEvidence,
    build_e2e_patch_bundle,
    source_stack_digest,
    verify_replay_config_invariants,
)
from apex.execution import SubprocessSupervisor
from apex.evaluation import load_e2e_paired_measurement

from .candidate import validate_frozen_sources
from .services import FinalDeliveryRequest, FinalDeliveryResult
from .source_delivery_models import (
    DeliveryProvenancePort,
    FormalSourceDeliveryProfile,
    PrimarySourceBuildPort,
    PrimarySourceBuildRequest,
)
from .source_delivery_provenance import ExactRequestProvenance
from .source_delivery_sources import CumulativeSourceMaterializer


_PRIMARY_RECEIPTS = frozenset(
    {
        "primary_build_receipt",
        "primary_engagement_receipt",
        "primary_benchmark_receipt",
        "primary_safety_receipt",
    }
)
_PROVENANCE_REASONS = frozenset(
    {
        "source_provenance_unresolved",
        "invalid_delivery_provenance",
        "invalid_repository_lock",
        "source_lock_unresolved",
    }
)


@dataclass(frozen=True, slots=True)
class FormalDeliveryBinding:
    """Explicitly trusted build/attestation/replay composition for one profile."""

    profile: FormalSourceDeliveryProfile
    primary_builder: PrimarySourceBuildPort
    verifier: E2EBundleVerifier
    verification_source_overrides: Mapping[str, Path] = field(default_factory=dict)


class SourceRebuildFinalDelivery:
    """Capture the accepted stack, source-build it, then verify it independently."""

    def __init__(
        self,
        bindings: Sequence[FormalDeliveryBinding],
        *,
        provenance: DeliveryProvenancePort | None = None,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self._bindings = tuple(bindings)
        keys = [
            (item.profile.recipe.parent_image_digest, item.profile.repository_ids)
            for item in self._bindings
        ]
        if not self._bindings or len(set(keys)) != len(keys):
            raise ContractError(
                "Formal delivery bindings must be non-empty and unambiguous",
                "invalid_source_delivery_profile",
            )
        self._provenance = provenance or ExactRequestProvenance()
        self._supervisor = supervisor or SubprocessSupervisor(
            max_output_bytes=16 * 1024 * 1024
        )
        self._sources = CumulativeSourceMaterializer(self._supervisor)

    def finalize(self, request: FinalDeliveryRequest) -> FinalDeliveryResult:
        try:
            return self._finalize(request)
        except ApexError as error:
            return _failed_result(request, error.reason_code, error.details or {})
        except (OSError, ValueError, yaml.YAMLError) as error:
            return _failed_result(
                request,
                "source_delivery_verification_failed",
                {"error_type": type(error).__name__},
            )
        except Exception as error:  # trusted adapter boundary; never promote on crash
            return _failed_result(
                request,
                "source_delivery_backend_error",
                {"error_type": type(error).__name__},
            )

    def _finalize(self, request: FinalDeliveryRequest) -> FinalDeliveryResult:
        changed_ids = _accepted_components(request)
        binding = self._select_binding(request, changed_ids)
        _validate_accepted_stack(request, binding.profile)
        provenance = self._provenance.lock(request)
        _validate_provenance_lock(request, provenance, binding.profile)
        root = _new_artifact_root(request.artifact_root)
        worktrees = self._sources.materialize(
            request, binding.profile, root / "primary-source-worktrees"
        )
        patches = self._sources.capture(request, binding.profile, worktrees)
        stack_sha = source_stack_digest(tuple(item.lock for item in patches))
        source_fingerprints = self._sources.fingerprints(worktrees)
        input_fingerprints = _input_fingerprints(request)
        primary = binding.primary_builder.build_and_validate(
            _primary_request(request, binding.profile, worktrees, patches, stack_sha, root)
        )
        if (
            source_fingerprints
            != self._sources.fingerprints(worktrees)
            or input_fingerprints != _input_fingerprints(request)
        ):
            raise IntegrityError(
                "Primary builder mutated frozen source or benchmark inputs",
                "primary_build_input_mutation",
            )
        evidence, configs = _validate_primary_output(
            request, binding.profile, primary, stack_sha
        )
        candidate = build_e2e_patch_bundle(
            bundle_id=f"{request.run_id}-source",
            bundle_dir=root / "candidate-bundle",
            repositories=patches,
            recipe=binding.profile.recipe,
            derived_image=primary.derived_image,
            provenance=provenance,
            configs=configs,
            primary_evidence=evidence,
            primary_receipts=primary.primary_receipts,
            image_sbom=primary.image_sbom,
        )
        outcome = binding.verifier.verify(
            bundle_dir=candidate.path,
            results_dir=root / "independent-verification",
            expected_digest=candidate.digest,
            source_overrides=binding.verification_source_overrides or None,
        )
        return _delivery_result(candidate.path, outcome)

    def _select_binding(
        self, request: FinalDeliveryRequest, changed_ids: frozenset[str]
    ) -> FormalDeliveryBinding:
        image_id = request.provenance.container.image_id
        if image_id is None or request.provenance.model_revision is None:
            raise ContractError(
                "Exact image/model provenance is required",
                "source_provenance_unresolved",
            )
        matches = tuple(
            item
            for item in self._bindings
            if item.profile.recipe.parent_image_digest == image_id
            and item.profile.repository_ids == changed_ids
        )
        if len(matches) != 1:
            raise ContractError(
                "No trusted fixed recipe matches the accepted source stack",
                "untrusted_build_recipe",
            )
        return matches[0]

def _accepted_components(request: FinalDeliveryRequest) -> frozenset[str]:
    if not request.accepted:
        raise ContractError("Formal delivery rejects config-only output", "config_only_candidate")
    identities = {
        (item.candidate.agent_result.backend.value, item.candidate.agent_result.model)
        for item in request.accepted
    }
    if (
        len(identities) != 1
        or request.agent_backend is None
        or request.agent_model is None
        or next(iter(identities)) != (request.agent_backend, request.agent_model)
    ):
        raise ContractError(
            "Accepted candidates lack one exact agent identity",
            "source_provenance_unresolved",
        )
    return frozenset(item.opportunity.origin_library for item in request.accepted)


def _validate_accepted_stack(
    request: FinalDeliveryRequest, profile: FormalSourceDeliveryProfile
) -> None:
    repositories = {item.repository_id: item for item in profile.repositories}
    for item in request.accepted:
        candidate = item.candidate
        repository = repositories.get(item.opportunity.origin_library)
        changed_paths = candidate.changed_files
        if (
            request.safety_policy_sha256 is not None
            and item.safety.evidence.get("policy_fingerprint")
            != request.safety_policy_sha256
        ):
            raise IntegrityError(
                "Accepted safety evidence binds another policy",
                "safety_policy_mismatch",
            )
        if (
            repository is None
            or item.opportunity.language not in repository.source_languages
            or not candidate.succeeded
            or not candidate.candidate_id
            or not candidate.candidate_source_sha256
            or not item.deployment.qualified
            or item.deployment.deployed_source_sha256
            != candidate.candidate_source_sha256
            or changed_paths != candidate.editable_files
            or not changed_paths
            or any(
                not _matches_allowlist(path, repository.editable_allowlist)
                for path in changed_paths
            )
        ):
            raise ContractError("Accepted source stack is not deliverable", "unsupported_delivery")
        validate_frozen_sources(candidate)


def _matches_allowlist(path: str, allowlist: Sequence[str]) -> bool:
    return any(
        path.startswith(item) if item.endswith("/") else path == item
        for item in allowlist
    )


def _primary_request(request, profile, worktrees, patches, stack_sha, root):
    return PrimarySourceBuildRequest(
        request.run_id,
        stack_sha,
        profile.recipe,
        worktrees,
        tuple(item.lock for item in patches),
        request.benchmark_original,
        request.benchmark_measurement,
        request.benchmark_diagnostic,
        request.benchmark_replay,
        request.baseline,
        request.final,
        request.acceptance_policy,
        root / "primary-source-build",
    )


def _validate_primary_output(request, profile, primary, stack_sha):
    gates = (
        primary.engagement_verified,
        primary.normal_runtime_measurement,
        primary.accuracy_passed,
        primary.latency_gates_passed,
        primary.objective_improved,
        primary.overlay_rebuild_parity_passed,
    )
    if (
        not primary.environment_id
        or not primary.runtime_identity_sha256
        or primary.source_stack_sha256 != stack_sha
        or primary.derived_image.parent_digest != profile.recipe.parent_image_digest
        or not all(gates)
        or set(primary.primary_receipts) != _PRIMARY_RECEIPTS
    ):
        raise IntegrityError("Primary source rebuild evidence failed", "primary_verification_failed")
    output_root = request.artifact_root / "primary-source-build"
    if primary.image_sbom.is_symlink() or not primary.image_sbom.is_file():
        raise IntegrityError("Primary SBOM is missing", "missing_image_sbom")
    if sha256_file(primary.image_sbom) != primary.derived_image.sbom_sha256:
        raise IntegrityError("Primary SBOM digest differs", "image_sbom_mismatch")
    for path in (
        primary.image_sbom,
        primary.benchmark_measurement,
        primary.benchmark_diagnostic,
        primary.benchmark_replay,
        *primary.primary_receipts.values(),
    ):
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or not _is_within(path, output_root)
        ):
            raise IntegrityError("Primary evidence path is unsafe", "missing_primary_receipt")
    _, _, semantics = verify_replay_config_invariants(
        primary.benchmark_measurement,
        primary.benchmark_replay,
        expected_image_locator=primary.derived_image.locator,
    )
    if semantics != request.baseline.protocol_hash:
        raise IntegrityError("Primary rebuild changed workload semantics", "benchmark_semantics_changed")
    receipt_hashes = {
        role: sha256_file(path) for role, path in primary.primary_receipts.items()
    }
    evidence = PrimaryVerificationEvidence(
        environment_id=primary.environment_id,
        runtime_identity_sha256=primary.runtime_identity_sha256,
        source_stack_sha256=stack_sha,
        build_receipt_sha256=receipt_hashes["primary_build_receipt"],
        engagement_receipt_sha256=receipt_hashes["primary_engagement_receipt"],
        benchmark_receipt_sha256=receipt_hashes["primary_benchmark_receipt"],
        safety_source_sha256=stack_sha,
        performance_source_sha256=stack_sha,
        deployed_source_sha256=stack_sha,
        engagement_verified=True,
        normal_runtime_measurement=True,
        accuracy_passed=True,
        latency_gates_passed=True,
        objective_improved=True,
        overlay_verified=True,
        overlay_source_sha256=stack_sha,
        overlay_rebuild_parity_passed=True,
        safety_certified=primary.safety_certified,
        safety_receipt_sha256=receipt_hashes["primary_safety_receipt"],
    )
    configs = {
        "benchmark_original": request.benchmark_original,
        "benchmark_measurement": primary.benchmark_measurement,
        "benchmark_diagnostic": primary.benchmark_diagnostic,
        "benchmark_replay": primary.benchmark_replay,
    }
    return evidence, configs


def _validate_provenance_lock(request, provenance, profile) -> None:
    actual = request.provenance
    if (
        provenance.primary_run_id != request.run_id
        or provenance.framework != actual.framework
        or provenance.model_id != actual.model_id
        or provenance.model_revision != actual.model_revision
        or provenance.baseline_image_digest != actual.container.image_id
        or provenance.workload_semantics_sha256 != request.baseline.protocol_hash
        or provenance.agent_backend != request.agent_backend
        or provenance.agent_model != request.agent_model
        or provenance.baseline_image_digest != profile.recipe.parent_image_digest
    ):
        raise IntegrityError("Formal provenance lock differs from run", "delivery_provenance_mismatch")


def _delivery_result(candidate_path: Path, outcome: Any) -> FinalDeliveryResult:
    result = outcome.result
    verified = outcome.verified_bundle
    if result.verified and verified is not None:
        if result.replay_receipt is None:
            raise IntegrityError(
                "Verified delivery lacks paired replay", "second_clean_replay_failed"
            )
        terminal = load_e2e_paired_measurement(
            result.replay_receipt.paired_measurement
        )
        return FinalDeliveryResult(
            True,
            TaskStatus.SUCCEEDED,
            result.reason_code,
            ValidationLevel.SOURCE_REBUILD_VERIFIED,
            bool(result.replay_receipt and result.replay_receipt.verified),
            str(verified.path),
            verified.digest,
            {
                "schema_version": 1,
                "candidate_bundle": str(candidate_path),
                "verification": result.to_dict(),
            },
            terminal,
            tuple(
                {
                    **item.to_dict(),
                    "path": str(
                        outcome.result_path.parent
                        / "clean-replay"
                        / item.relative_path
                    ),
                }
                for item in result.replay_receipt.raw_artifacts
            ),
        )
    return FinalDeliveryResult(
        False,
        result.status,
        result.reason_code,
        result.validation_level,
        False,
        None,
        None,
        {
            "schema_version": 1,
            "candidate_bundle": str(candidate_path),
            "verification": result.to_dict(),
        },
    )


def _failed_result(
    request: FinalDeliveryRequest, reason: str, details: Mapping[str, Any]
) -> FinalDeliveryResult:
    status = (
        TaskStatus.PROVENANCE_UNRESOLVED
        if reason in _PROVENANCE_REASONS
        else TaskStatus.VERIFICATION_FAILED
    )
    level = (
        ValidationLevel.RUNTIME_OVERLAY_VERIFIED
        if any(
            item.deployment.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
            for item in request.accepted
        )
        else ValidationLevel.NONE
    )
    return FinalDeliveryResult(
        False,
        status,
        reason,
        level,
        False,
        None,
        None,
        {"schema_version": 1, "failure": reason, "details": dict(details)},
    )


def _new_artifact_root(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or path.exists():
        raise IntegrityError("Formal delivery root must be a new absolute path", "immutable_delivery_artifact")
    path.mkdir(parents=True)
    return path.resolve()


def _input_fingerprints(request: FinalDeliveryRequest) -> dict[str, str]:
    return {
        role: sha256_file(path)
        for role, path in {
            "original": request.benchmark_original,
            "measurement": request.benchmark_measurement,
            "diagnostic": request.benchmark_diagnostic,
            "replay": request.benchmark_replay,
        }.items()
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


__all__ = ["FormalDeliveryBinding", "SourceRebuildFinalDelivery"]
