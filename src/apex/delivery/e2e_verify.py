"""Independent source-rebuild and second-clean-replay verification use case."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import (
    ContractError,
    IntegrityError,
    TaskStatus,
    ValidationLevel,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)
from apex.execution import SubprocessSupervisor
from apex.runtime import canonical_repository

from .e2e_bundle import E2EPatchBundle, finalize_verified_e2e_bundle, load_and_verify_e2e_bundle, verify_replay_config_invariants
from .e2e_models import BuildRecipeLock, DerivedImageIdentity
from .e2e_receipts import (
    CleanReplayReceipt,
    BuildStepReceipt,
    DeliveryVerificationResult,
    LoadedByteEngagementReceipt,
    ReplayConfigInvariantReceipt,
    SourceBuildReceipt,
)
from .git_patch import CleanPatchMaterializer, RepositoryApplyReceipt


@dataclass(frozen=True, slots=True)
class SourceBuildRequest:
    bundle_digest: str
    source_stack_sha256: str
    recipe: BuildRecipeLock
    expected_image: DerivedImageIdentity
    repository_roots: Mapping[str, Path]
    repository_receipts: tuple[RepositoryApplyReceipt, ...]
    output_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class EngagementRequest:
    bundle_digest: str
    source_stack_sha256: str
    expected_image: DerivedImageIdentity
    build_receipt: SourceBuildReceipt


@dataclass(frozen=True, slots=True)
class ReplayRequest:
    bundle_digest: str
    source_stack_sha256: str
    primary_environment_id: str
    expected_image: DerivedImageIdentity
    replay_config: Path
    config_receipt: ReplayConfigInvariantReceipt
    engagement_receipt: LoadedByteEngagementReceipt
    repository_receipts: tuple[RepositoryApplyReceipt, ...] = ()
    primary_receipts: Mapping[str, Path] | None = None
    output_dir: Path | None = None


class SourceBuildBackend(Protocol):
    def build(self, request: SourceBuildRequest) -> SourceBuildReceipt: ...


class EngagementBackend(Protocol):
    def verify_loaded_bytes(self, request: EngagementRequest) -> LoadedByteEngagementReceipt: ...


class CleanReplayBackend(Protocol):
    def replay(self, request: ReplayRequest) -> CleanReplayReceipt: ...


class BuildAttestor(Protocol):
    def attest(self, request: SourceBuildRequest) -> SourceBuildReceipt: ...


class SupervisedRecipeBuildBackend:
    """Execute a trusted fixed recipe as argv arrays, then invoke an attestor.

    This adapter deliberately has no shell-string fallback.  Image identity and
    installed-artifact evidence remain the responsibility of an independent
    attestor (normally the deployment adapter), not the build command itself.
    """

    def __init__(
        self,
        attestor: BuildAttestor,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self._attestor = attestor
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=16 * 1024 * 1024)

    def build(self, request: SourceBuildRequest) -> SourceBuildReceipt:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        receipts: list[BuildStepReceipt] = []
        for index, step in enumerate(request.recipe.steps):
            root = request.repository_roots.get(step.repository_id)
            if root is None:
                raise IntegrityError("Build step references an unmaterialized repository", "invalid_build_recipe")
            cwd = root if step.cwd == "." else root.joinpath(*step.cwd.split("/"))
            resolved = cwd.resolve(strict=True)
            if resolved != root.resolve() and root.resolve() not in resolved.parents or not resolved.is_dir():
                raise IntegrityError("Build step cwd escapes source root", "unsafe_build_cwd")
            step_environment = dict(environment)
            step_environment.update(dict(step.environment))
            result = self._supervisor.run(
                step.argv,
                cwd=resolved,
                environment=step_environment,
                timeout_seconds=step.timeout_seconds,
            )
            receipts.append(
                BuildStepReceipt(
                    index=index,
                    repository_id=step.repository_id,
                    cwd=step.cwd,
                    argv_sha256=sha256_json(list(step.argv)),
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                    stdout_sha256=sha256_bytes(result.stdout.encode("utf-8")),
                    stderr_sha256=sha256_bytes(result.stderr.encode("utf-8")),
                )
            )
            if result.exit_code != 0 or result.timed_out or result.stdout_truncated or result.stderr_truncated:
                raise IntegrityError("Fixed source build recipe failed", "source_build_failed")
        return replace(self._attestor.attest(request), step_receipts=tuple(receipts))


@dataclass(frozen=True, slots=True)
class TerminalDeliveryVerdict:
    verified: bool
    status: TaskStatus
    validation_level: ValidationLevel
    reason_code: str


def delivery_terminal_policy(
    *,
    source_locks_resolved: bool,
    overlay_verified: bool,
    repositories_verified: bool,
    build_verified: bool,
    engagement_verified: bool,
    config_verified: bool,
    clean_replay_verified: bool,
) -> TerminalDeliveryVerdict:
    """The sole fail-closed mapping from delivery evidence to user verdict."""

    overlay_level = (
        ValidationLevel.RUNTIME_OVERLAY_VERIFIED if overlay_verified else ValidationLevel.NONE
    )
    if not source_locks_resolved:
        return TerminalDeliveryVerdict(
            False,
            TaskStatus.PROVENANCE_UNRESOLVED,
            overlay_level,
            "source_provenance_unresolved",
        )
    if all(
        (
            repositories_verified,
            build_verified,
            engagement_verified,
            config_verified,
            clean_replay_verified,
        )
    ):
        return TerminalDeliveryVerdict(
            True,
            TaskStatus.SUCCEEDED,
            ValidationLevel.SOURCE_REBUILD_VERIFIED,
            "source_rebuild_and_second_clean_replay_verified",
        )
    return TerminalDeliveryVerdict(
        False,
        TaskStatus.VERIFICATION_FAILED,
        overlay_level,
        "source_delivery_verification_failed",
    )


@dataclass(frozen=True, slots=True)
class BundleVerifyOutcome:
    result: DeliveryVerificationResult
    result_path: Path
    verified_bundle: E2EPatchBundle | None


@dataclass(slots=True)
class _VerificationEvidence:
    repositories: tuple[RepositoryApplyReceipt, ...] = ()
    build: SourceBuildReceipt | None = None
    engagement: LoadedByteEngagementReceipt | None = None
    config: ReplayConfigInvariantReceipt | None = None
    replay: CleanReplayReceipt | None = None


class E2EBundleVerifier:
    """Independently rebuild, engage, and replay one trusted E2E bundle."""

    def __init__(
        self,
        *,
        trusted_recipes: Mapping[str, BuildRecipeLock],
        trusted_source_urls: Mapping[str, str],
        build_backend: SourceBuildBackend,
        engagement_backend: EngagementBackend,
        replay_backend: CleanReplayBackend,
        materializer: CleanPatchMaterializer | None = None,
        default_source_overrides: Mapping[str, Path] | None = None,
        trusted_recipe_repositories: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self._trusted_recipes = dict(trusted_recipes)
        self._trusted_source_urls = dict(trusted_source_urls)
        self._build = build_backend
        self._engagement = engagement_backend
        self._replay = replay_backend
        self._materializer = materializer or CleanPatchMaterializer()
        self._default_source_overrides = dict(default_source_overrides or {})
        self._trusted_recipe_repositories = dict(trusted_recipe_repositories or {})

    def verify(
        self,
        *,
        bundle_dir: Path,
        results_dir: Path,
        expected_digest: str | None = None,
        source_overrides: Mapping[str, Path] | None = None,
    ) -> BundleVerifyOutcome:
        """Rebuild and replay from a second disposable clean environment."""

        if not bundle_dir.is_absolute() or not results_dir.is_absolute():
            raise ContractError("Bundle verification paths must be absolute", "invalid_bundle_path")
        candidate = load_and_verify_e2e_bundle(bundle_dir, expected_digest=expected_digest)
        if results_dir.exists():
            raise ContractError("Bundle verification results directory already exists", "results_exist")
        results_dir.mkdir(parents=True)
        result_path = results_dir / "verification.result.json"
        trust_failure = self._trust_failure(candidate)
        evidence = _VerificationEvidence()
        if trust_failure is not None:
            result = _verification_result(candidate, evidence, trust_failure)
            return _publish_outcome(candidate, result, result_path, results_dir)
        try:
            selected_sources = (
                source_overrides
                if source_overrides is not None
                else self._default_source_overrides or None
            )
            self._collect_evidence(
                candidate,
                results_dir,
                evidence,
                source_overrides=selected_sources,
            )
            result = _verification_result(candidate, evidence)
        except (ContractError, IntegrityError) as error:
            result = _verification_result(candidate, evidence, error.reason_code)
        return _publish_outcome(candidate, result, result_path, results_dir)

    def _trust_failure(self, candidate: E2EPatchBundle) -> str | None:
        recipe_sha = candidate.recipe.computed_sha256
        trusted = self._trusted_recipes.get(recipe_sha)
        if trusted is None or trusted.to_dict() != candidate.recipe.to_dict():
            return "untrusted_build_recipe"
        trusted_repositories = self._trusted_recipe_repositories.get(recipe_sha)
        candidate_repositories = frozenset(
            item.repository_id for item in candidate.repositories
        )
        if (
            trusted_repositories is not None
            and trusted_repositories != candidate_repositories
        ):
            return "untrusted_build_recipe"
        urls_match = all(
            repository.repository_id in self._trusted_source_urls
            and canonical_repository(self._trusted_source_urls[repository.repository_id])
            == canonical_repository(repository.url)
            for repository in candidate.repositories
        )
        return None if urls_match else "untrusted_source_repository"

    def _collect_evidence(
        self,
        candidate: E2EPatchBundle,
        results_dir: Path,
        evidence: _VerificationEvidence,
        *,
        source_overrides: Mapping[str, Path] | None,
    ) -> None:
        roots, evidence.repositories = self._materializer.materialize(
            bundle_root=candidate.path,
            locks=candidate.repositories,
            destination=results_dir / "worktrees",
            source_overrides=source_overrides,
        )
        request = SourceBuildRequest(
            candidate.digest,
            candidate.primary_evidence.source_stack_sha256,
            candidate.recipe,
            candidate.derived_image,
            roots,
            evidence.repositories,
            results_dir / "source-build",
        )
        evidence.build = self._build.build(request)
        _validate_build_receipt(candidate, evidence.build)
        evidence.engagement = self._engagement.verify_loaded_bytes(
            EngagementRequest(
                candidate.digest,
                candidate.primary_evidence.source_stack_sha256,
                candidate.derived_image,
                evidence.build,
            )
        )
        _validate_engagement_receipt(candidate, evidence.build, evidence.engagement)
        measurement_sha, replay_sha, semantics_sha = verify_replay_config_invariants(
            candidate.config_paths["benchmark_measurement"],
            candidate.config_paths["benchmark_replay"],
            expected_image_locator=candidate.derived_image.locator,
        )
        evidence.config = ReplayConfigInvariantReceipt(
            measurement_sha,
            replay_sha,
            semantics_sha,
            candidate.derived_image.locator,
            True,
        )
        evidence.replay = self._replay.replay(
            ReplayRequest(
                candidate.digest,
                candidate.primary_evidence.source_stack_sha256,
                candidate.primary_evidence.environment_id,
                candidate.derived_image,
                candidate.config_paths["benchmark_replay"],
                evidence.config,
                evidence.engagement,
                evidence.repositories,
                candidate.primary_receipt_paths,
                results_dir / "clean-replay",
            )
        )
        _validate_replay_receipt(candidate, evidence.config, evidence.replay)


def _verification_result(
    candidate: E2EPatchBundle,
    evidence: _VerificationEvidence,
    reason_code: str | None = None,
) -> DeliveryVerificationResult:
    verdict = delivery_terminal_policy(
        source_locks_resolved=bool(candidate.repositories),
        overlay_verified=candidate.primary_evidence.overlay_verified,
        repositories_verified=bool(evidence.repositories)
        and all(item.verified for item in evidence.repositories),
        build_verified=bool(evidence.build and evidence.build.verified),
        engagement_verified=bool(evidence.engagement and evidence.engagement.verified),
        config_verified=bool(evidence.config and evidence.config.verified),
        clean_replay_verified=bool(evidence.replay and evidence.replay.verified),
    )
    return DeliveryVerificationResult(
        candidate.digest,
        verdict.verified,
        verdict.status,
        verdict.validation_level,
        reason_code or verdict.reason_code,
        evidence.repositories,
        evidence.build,
        evidence.engagement,
        evidence.config,
        evidence.replay,
    )


def _publish_outcome(
    candidate: E2EPatchBundle,
    result: DeliveryVerificationResult,
    result_path: Path,
    results_dir: Path,
) -> BundleVerifyOutcome:
    _atomic_write(result_path, canonical_json_bytes(result.to_dict()) + b"\n")
    final = (
        finalize_verified_e2e_bundle(
            candidate,
            verification=result,
            destination=results_dir / "bundle",
        )
        if result.verified and not candidate.verified
        else None
    )
    return BundleVerifyOutcome(result, result_path, final)


def _validate_build_receipt(bundle: E2EPatchBundle, receipt: SourceBuildReceipt) -> None:
    if (
        receipt.bundle_digest != bundle.digest
        or receipt.recipe_sha256 != bundle.recipe.computed_sha256
        or receipt.expected_parent_digest != bundle.derived_image.parent_digest
        or receipt.observed_parent_digest != bundle.derived_image.parent_digest
        or receipt.expected_image_digest != bundle.derived_image.image_digest
        or receipt.observed_image_digest != bundle.derived_image.image_digest
        or receipt.expected_sbom_sha256 != bundle.derived_image.sbom_sha256
        or receipt.observed_sbom_sha256 != bundle.derived_image.sbom_sha256
        or receipt.source_stack_sha256 != bundle.primary_evidence.source_stack_sha256
        or not receipt.verified
    ):
        raise IntegrityError("Source build receipt does not match bundle", "source_build_receipt_mismatch")
    changed_components = {item.runtime_component for item in bundle.repositories}
    engaged_components = {item.component for item in receipt.artifacts}
    if not changed_components.issubset(engaged_components):
        raise IntegrityError(
            "Build receipt does not identify an artifact for every changed repository",
            "source_build_receipt_mismatch",
        )
    if len(receipt.step_receipts) != len(bundle.recipe.steps):
        raise IntegrityError("Build receipt is missing fixed recipe steps", "source_build_receipt_mismatch")
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
            raise IntegrityError("Build step receipt differs from recipe", "source_build_receipt_mismatch")


def _validate_engagement_receipt(
    bundle: E2EPatchBundle,
    build: SourceBuildReceipt,
    receipt: LoadedByteEngagementReceipt,
) -> None:
    if (
        receipt.bundle_digest != bundle.digest
        or receipt.image_digest != bundle.derived_image.image_digest
        or receipt.source_stack_sha256 != bundle.primary_evidence.source_stack_sha256
        or not receipt.verified
    ):
        raise IntegrityError("Runtime engagement does not match bundle", "loaded_byte_engagement_failed")
    built = {
        (item.component, item.runtime_path, item.sha256, item.build_id)
        for item in build.artifacts
    }
    loaded = {
        (item.component, item.runtime_path, item.expected_sha256, item.expected_build_id)
        for item in receipt.artifacts
        if item.verified
    }
    if built != loaded or len(loaded) != len(receipt.artifacts):
        raise IntegrityError("Runtime loaded old or unexpected bytes", "loaded_byte_engagement_failed")


def _validate_replay_receipt(
    bundle: E2EPatchBundle,
    config: ReplayConfigInvariantReceipt,
    receipt: CleanReplayReceipt,
) -> None:
    if (
        receipt.bundle_digest != bundle.digest
        or receipt.primary_environment_id != bundle.primary_evidence.environment_id
        or receipt.image_digest != bundle.derived_image.image_digest
        or receipt.replay_config_sha256 != config.replay_config_sha256
        or receipt.source_stack_sha256 != bundle.primary_evidence.source_stack_sha256
        or not receipt.verified
    ):
        raise IntegrityError("Second clean replay does not match bundle", "second_clean_replay_failed")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "BuildAttestor",
    "BundleVerifyOutcome",
    "CleanReplayBackend",
    "E2EBundleVerifier",
    "EngagementBackend",
    "EngagementRequest",
    "ReplayRequest",
    "SourceBuildBackend",
    "SourceBuildRequest",
    "SupervisedRecipeBuildBackend",
    "TerminalDeliveryVerdict",
    "delivery_terminal_policy",
]
