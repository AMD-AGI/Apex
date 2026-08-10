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
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    SubprocessSupervisor,
    build_subprocess_environment,
)
from apex.runtime import canonical_repository

from .e2e_bundle import E2EPatchBundle, finalize_verified_e2e_bundle, load_and_verify_e2e_bundle, verify_replay_config_invariants
from .e2e_models import BuildRecipeLock, DerivedImageIdentity, SourceComponentCapability
from .e2e_receipts import (
    CleanReplayReceipt,
    BuildStepReceipt,
    DeliveryVerificationResult,
    LoadedByteEngagementReceipt,
    ReplayConfigInvariantReceipt,
    SourceBuildReceipt,
)
from .git_patch import CleanPatchMaterializer, RepositoryApplyReceipt
from .e2e_verify_validation import (
    validate_build_receipt,
    validate_engagement_receipt,
    validate_replay_receipt,
)


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
    baseline_config: Path
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
        receipts: list[BuildStepReceipt] = []
        for index, step in enumerate(request.recipe.steps):
            root = request.repository_roots.get(step.repository_id)
            if root is None:
                raise IntegrityError("Build step references an unmaterialized repository", "invalid_build_recipe")
            cwd = root if step.cwd == "." else root.joinpath(*step.cwd.split("/"))
            resolved = cwd.resolve(strict=True)
            if resolved != root.resolve() and root.resolve() not in resolved.parents or not resolved.is_dir():
                raise IntegrityError("Build step cwd escapes source root", "unsafe_build_cwd")
            step_environment = build_subprocess_environment(
                dict(step.environment),
                inherit=(
                    *DOCKER_RUNTIME_ENVIRONMENT_KEYS,
                    *GPU_RUNTIME_ENVIRONMENT_KEYS,
                    *HF_RUNTIME_ENVIRONMENT_KEYS,
                ),
                fixed={
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_OPTIONAL_LOCKS": "0",
                },
            )
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
        trusted_recipe_capabilities: Mapping[
            str, tuple[SourceComponentCapability, ...]
        ],
        build_backend: SourceBuildBackend,
        engagement_backend: EngagementBackend,
        replay_backend: CleanReplayBackend,
        materializer: CleanPatchMaterializer | None = None,
        default_source_overrides: Mapping[str, Path] | None = None,
    ) -> None:
        self._trusted_recipes = dict(trusted_recipes)
        self._trusted_source_urls = dict(trusted_source_urls)
        self._trusted_recipe_capabilities = {
            key: tuple(value) for key, value in trusted_recipe_capabilities.items()
        }
        if set(self._trusted_recipe_capabilities) != set(self._trusted_recipes):
            raise ContractError(
                "Every trusted recipe requires component capabilities",
                "invalid_source_delivery_profile",
            )
        for digest, recipe in self._trusted_recipes.items():
            repository_ids = {
                item.repository_id
                for item in self._trusted_recipe_capabilities[digest]
            }
            if repository_ids != {item.repository_id for item in recipe.steps}:
                raise ContractError(
                    "Recipe capabilities do not match its source steps",
                    "invalid_source_delivery_profile",
                )
        self._build = build_backend
        self._engagement = engagement_backend
        self._replay = replay_backend
        self._materializer = materializer or CleanPatchMaterializer()
        self._default_source_overrides = dict(default_source_overrides or {})

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
        trusted_capabilities = {
            item.repository_id: item.to_dict()
            for item in self._trusted_recipe_capabilities[recipe_sha]
        }
        observed_capabilities = {
            item.repository_id: item.component_capability.to_dict()
            for item in candidate.repositories
        }
        if trusted_capabilities != observed_capabilities:
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
        validate_build_receipt(candidate, evidence.build)
        evidence.engagement = self._engagement.verify_loaded_bytes(
            EngagementRequest(
                candidate.digest,
                candidate.primary_evidence.source_stack_sha256,
                candidate.derived_image,
                evidence.build,
            )
        )
        validate_engagement_receipt(candidate, evidence.build, evidence.engagement)
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
                candidate.config_paths["benchmark_original"],
                candidate.config_paths["benchmark_replay"],
                evidence.config,
                evidence.engagement,
                evidence.repositories,
                candidate.primary_receipt_paths,
                results_dir / "clean-replay",
            )
        )
        validate_replay_receipt(
            candidate,
            evidence.config,
            evidence.replay,
            evidence.repositories,
            results_dir / "clean-replay",
        )


def _verification_result(
    candidate: E2EPatchBundle,
    evidence: _VerificationEvidence,
    reason_code: str | None = None,
) -> DeliveryVerificationResult:
    evidence_valid = reason_code is None
    verdict = delivery_terminal_policy(
        source_locks_resolved=bool(candidate.repositories),
        overlay_verified=candidate.primary_evidence.overlay_verified,
        repositories_verified=bool(evidence.repositories)
        and all(item.verified for item in evidence.repositories),
        build_verified=bool(evidence.build and evidence.build.verified),
        engagement_verified=bool(evidence.engagement and evidence.engagement.verified),
        config_verified=bool(evidence.config and evidence.config.verified),
        clean_replay_verified=bool(
            evidence_valid and evidence.replay and evidence.replay.verified
        ),
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
