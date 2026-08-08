"""Runtime-verified Docker overlay deployment for exact vLLM/AITER source locks."""

from __future__ import annotations

import os
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

import yaml

from apex.core import ApexError, ContractError, IntegrityError, TaskStatus, ValidationLevel
from apex.core import sha256_file
from apex.execution import SubprocessSupervisor
from apex.runtime import RepositoryLock, RunProvenance

from .candidate import (
    E2ECandidate,
    frozen_candidate_source,
    materialize_frozen_sources,
    validate_frozen_sources,
)
from .kernel_lane import KernelOpportunity
from .overlay_lineage import (
    capture_overlay_build_receipt,
    validate_accepted_overlay_parent,
)
from .overlay_config import OverlayConfigSet, derive_overlay_configs
from .overlay_runtime import (
    BuiltOverlay,
    ContainerEngine,
    ContainerImage,
    DockerEngine,
    InstalledPythonTarget,
    LoadedFileReceipt,
)
from .services import (
    AcceptedCandidate,
    CandidateDeployment,
    CandidateDeploymentRequest,
    DeploymentConfigDigests,
    FinalDeliveryRequest,
    FinalDeliveryResult,
)


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourceLockVerifier(Protocol):
    def verify(self, lock: RepositoryLock, *, expected_root: Path) -> Mapping[str, Any]: ...


class GitSourceLockVerifier:
    """Recheck commit, tree, and cleanliness immediately before deployment."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=2 * 1024 * 1024)

    def verify(self, lock: RepositoryLock, *, expected_root: Path) -> Mapping[str, Any]:
        root = Path(lock.path).resolve(strict=True)
        if root != expected_root.resolve(strict=True) or not lock.clean:
            raise IntegrityError("Source lock does not match kernel root", "source_lock_mismatch")
        commit = self._git(root, ("git", "rev-parse", "HEAD"))
        tree = self._git(root, ("git", "rev-parse", "HEAD^{tree}"))
        status = self._git(
            root, ("git", "status", "--porcelain=v1", "--untracked-files=all")
        )
        if commit != lock.commit or tree != lock.tree or status:
            raise IntegrityError("Source repository moved or is dirty", "source_lock_drift")
        return {
            "name": lock.name,
            "path": str(root),
            "url": lock.url,
            "commit": commit,
            "tree": tree,
            "clean": True,
        }

    def _git(self, cwd: Path, argv: tuple[str, ...]) -> str:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        result = self._supervisor.run(
            argv, cwd=cwd, environment=environment, timeout_seconds=60
        )
        if result.timed_out or result.exit_code != 0 or result.stdout_truncated:
            raise IntegrityError("Cannot verify source lock", "source_lock_inspection_failed")
        return result.stdout.strip()


class DockerOverlayDeployment:
    """Install one frozen Python file into an immutable derived container image."""

    def __init__(
        self,
        engine: ContainerEngine | None = None,
        source_locks: SourceLockVerifier | None = None,
    ) -> None:
        self._engine = engine or DockerEngine()
        self._source_locks = source_locks or GitSourceLockVerifier()

    def supports(self, opportunity: KernelOpportunity, provenance: RunProvenance) -> bool:
        return bool(
            opportunity.eligible
            and opportunity.language in {"python", "triton"}
            and opportunity.origin_library in {"vllm", "aiter"}
            and opportunity.source_root
            and opportunity.source_path
            and opportunity.source_path.suffix == ".py"
            and opportunity.origin_library in provenance.active_components
            and provenance.container.resolved
            and _matching_lock(opportunity, provenance) is not None
            and _safe_package_relative(opportunity) is not None
        )

    def deploy(self, request: CandidateDeploymentRequest) -> CandidateDeployment:
        try:
            return self._deploy(request)
        except ApexError as error:
            return _failed_deployment(
                request,
                error.reason_code,
                error.details or {},
                infrastructure_failure=True,
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            return _failed_deployment(
                request,
                "overlay_deployment_failed",
                {"error_type": type(error).__name__},
                infrastructure_failure=True,
            )

    def _deploy(self, request: CandidateDeploymentRequest) -> CandidateDeployment:
        candidate = request.candidate
        opportunity = request.opportunity
        _validate_request(request)
        relative, baseline_path = _candidate_paths(candidate, opportunity)
        lock = _matching_lock(opportunity, request.provenance)
        if lock is None:
            raise ContractError("No exact source lock for candidate", "source_lock_unresolved")
        lock_receipt = self._source_locks.verify(lock, expected_root=opportunity.source_root)
        artifact_root = _prepare_artifact_root(request.artifact_root)
        snapshot_root = materialize_frozen_sources(
            candidate, artifact_root / "frozen-candidate"
        )
        candidate_path = snapshot_root.joinpath(*PurePosixPath(relative).parts)
        parent, target = self._resolve_target(
            request, relative=relative, artifact_root=artifact_root
        )
        baseline_sha, candidate_sha = _verify_source_bytes(
            target, baseline_path=baseline_path, candidate_path=candidate_path
        )
        built, loaded = self._build_and_attest(
            parent,
            target,
            candidate_path=candidate_path,
            candidate_sha=candidate_sha,
            artifact_root=artifact_root,
        )
        configs = _derive_configs(request, artifact_root, built.image.image_id)
        return _successful_deployment(
            request,
            lock_receipt=lock_receipt,
            parent=parent,
            target=target,
            baseline_sha=baseline_sha,
            candidate_sha=candidate_sha,
            built=built,
            loaded=loaded,
            configs=configs,
        )

    def _resolve_target(
        self,
        request: CandidateDeploymentRequest,
        *,
        relative: str,
        artifact_root: Path,
    ) -> tuple[ContainerImage, InstalledPythonTarget]:
        parent_reference = _config_image(request.benchmark_measurement)
        inspected = self._engine.inspect_image(parent_reference, cwd=artifact_root)
        parent = _validate_parent(
            parent_reference,
            inspected,
            request.provenance,
            accepted=request.accepted_stack,
            anchor_generation=request.anchor_generation,
        )
        target = self._engine.resolve_python_target(
            parent.image_id,
            library=request.opportunity.origin_library,
            repo_relative_path=relative,
            cwd=artifact_root,
        )
        _validate_target_mapping(target, request.opportunity, relative)
        return parent, target

    def _build_and_attest(
        self,
        parent: ContainerImage,
        target: InstalledPythonTarget,
        *,
        candidate_path: Path,
        candidate_sha: str,
        artifact_root: Path,
    ) -> tuple[BuiltOverlay, LoadedFileReceipt]:
        built = self._engine.build_overlay(
            parent=parent,
            candidate_source=candidate_path,
            target=target,
            build_root=artifact_root,
            cwd=artifact_root,
        )
        if built.context_source_sha256 != candidate_sha:
            raise IntegrityError("Build context changed candidate bytes", "candidate_lineage_mismatch")
        loaded = self._engine.read_file(
            built.image.image_id,
            container_path=target.container_path,
            cwd=artifact_root,
        )
        if loaded.container_path != target.container_path or loaded.sha256 != candidate_sha:
            raise IntegrityError(
                "Derived image did not load frozen candidate bytes",
                "loaded_candidate_bytes_mismatch",
                {"loaded_sha256": loaded.sha256, "candidate_sha256": candidate_sha},
            )
        return built, loaded

    def rollback(self, _deployment: CandidateDeployment) -> None:
        """Select the prior immutable config; never edit source or host packages."""

        return None


class OverlayOnlyFinalDelivery:
    """Report a verified runtime overlay without claiming source-build delivery."""

    def finalize(self, request: FinalDeliveryRequest) -> FinalDeliveryResult:
        accepted = bool(request.accepted)
        level = ValidationLevel.RUNTIME_OVERLAY_VERIFIED if accepted else ValidationLevel.NONE
        return FinalDeliveryResult(
            False,
            TaskStatus.PROVENANCE_UNRESOLVED,
            "runtime_overlay_only_source_rebuild_unverified",
            level,
            False,
            None,
            None,
            {
                "schema_version": 1,
                "overlay_count": len(request.accepted),
                "primary_no_regression_verified": accepted,
                "source_rebuild_verified": False,
                "second_clean_replay_verified": False,
                "formal_success": False,
            },
        )


def _verify_source_bytes(
    target: InstalledPythonTarget,
    *,
    baseline_path: Path,
    candidate_path: Path,
) -> tuple[str, str]:
    baseline_sha = sha256_file(baseline_path)
    if target.sha256 != baseline_sha:
        raise IntegrityError(
            "Installed parent bytes differ from the exact source lock",
            "installed_baseline_source_mismatch",
            {"installed_sha256": target.sha256, "source_sha256": baseline_sha},
        )
    candidate_sha = sha256_file(candidate_path)
    if candidate_sha == baseline_sha:
        raise IntegrityError("Candidate file did not change", "agent_made_no_source_change")
    return baseline_sha, candidate_sha


def _derive_configs(
    request: CandidateDeploymentRequest, artifact_root: Path, image_id: str
) -> OverlayConfigSet:
    replay_source = request.benchmark_replay or request.benchmark_measurement
    return derive_overlay_configs(
        measurement=request.benchmark_measurement,
        diagnostic=request.benchmark_diagnostic,
        replay=replay_source,
        output_dir=artifact_root / "configs",
        image_id=image_id,
        workload_semantics_sha256=request.workload_semantics_sha256,
    )


def _successful_deployment(
    request: CandidateDeploymentRequest,
    *,
    lock_receipt: Mapping[str, Any],
    parent: ContainerImage,
    target: InstalledPythonTarget,
    baseline_sha: str,
    candidate_sha: str,
    built: BuiltOverlay,
    loaded: LoadedFileReceipt,
    configs: OverlayConfigSet,
) -> CandidateDeployment:
    candidate = request.candidate
    assert candidate.candidate_id and candidate.candidate_source_sha256
    config_digests = DeploymentConfigDigests.capture(
        configs.measurement,
        configs.diagnostic,
        configs.replay,
    )
    build_receipt = capture_overlay_build_receipt(
        candidate_id=candidate.candidate_id,
        candidate_source_sha256=candidate.candidate_source_sha256,
        parent=parent,
        built=built,
        dockerfile_sha256=built.dockerfile_sha256,
        candidate_file_sha256=candidate_sha,
        loaded=loaded,
        accepted=request.accepted_stack,
        anchor_generation=request.anchor_generation,
        provenance=request.provenance,
    )
    evidence = {
        "schema_version": 1,
        "deployment_kind": "docker_python_runtime_overlay",
        "formal_source_rebuild": False,
        "host_site_packages_mutated": False,
        "source_repository_mutated": False,
        "source_lock": dict(lock_receipt),
        "mapping": target.to_dict(),
        "parent_image": asdict(parent),
        "derived_image": asdict(built.image),
        "dockerfile_sha256": built.dockerfile_sha256,
        "baseline_source_sha256": baseline_sha,
        "candidate_file_sha256": candidate_sha,
        "loaded_candidate": loaded.to_dict(),
        "overlay_build_receipt": build_receipt.to_dict(),
        "overlay_build_receipt_sha256": build_receipt.digest,
        "config_paths": {
            "measurement": str(configs.measurement),
            "diagnostic": str(configs.diagnostic),
            "replay": str(configs.replay),
        },
        "config_sha256": config_digests.to_dict(),
    }
    return CandidateDeployment(
        candidate.candidate_id,
        True,
        "runtime_overlay_loaded_bytes_verified",
        configs.measurement,
        configs.diagnostic,
        configs.replay,
        request.workload_semantics_sha256,
        candidate.candidate_source_sha256,
        built.image.image_id,
        ValidationLevel.RUNTIME_OVERLAY_VERIFIED,
        True,
        evidence,
        config_sha256=config_digests,
    )


def _validate_request(request: CandidateDeploymentRequest) -> None:
    candidate = request.candidate
    if not request.safety.qualified:
        raise ContractError("Safety gate did not permit deployment", "safety_gate_failed")
    if (
        request.anchor_generation != len(request.accepted_stack)
        or len(
            {
                item.candidate.candidate_id
                for item in request.accepted_stack
                if item.candidate.candidate_id is not None
            }
        )
        != len(request.accepted_stack)
        or not candidate.succeeded
        or not candidate.candidate_id
        or not candidate.candidate_source_sha256
        or len(candidate.changed_files) != 1
        or candidate.changed_files != candidate.editable_files
    ):
        raise ContractError("Overlay requires one frozen source edit", "invalid_frozen_candidate")
    validate_frozen_sources(candidate)


def _validate_target_mapping(
    target: InstalledPythonTarget,
    opportunity: KernelOpportunity,
    relative: str,
) -> None:
    expected_module = PurePosixPath(*PurePosixPath(relative).parts[1:]).as_posix()
    path = PurePosixPath(target.container_path)
    if (
        target.package != opportunity.origin_library
        or target.repo_relative_path != relative
        or target.module_relative_path != expected_module
        or not path.is_absolute()
        or path.suffix != ".py"
    ):
        raise IntegrityError(
            "Container package probe returned a different source mapping",
            "container_source_mapping_mismatch",
        )


def _candidate_paths(
    candidate: E2ECandidate, opportunity: KernelOpportunity
) -> tuple[str, Path]:
    if opportunity.source_root is None or opportunity.source_path is None:
        raise ContractError("Kernel source is unresolved", "source_unresolved")
    relative = opportunity.source_path.resolve(strict=True).relative_to(
        opportunity.source_root.resolve(strict=True)
    ).as_posix()
    if candidate.changed_files != (relative,):
        raise ContractError(
            "Candidate changed files do not match the selected source",
            "candidate_source_mapping_mismatch",
        )
    if _safe_package_relative(opportunity) is None:
        raise ContractError(
            "Selected source does not map to an overlay package",
            "source_mapping_mismatch",
        )
    frozen_candidate_source(candidate, relative)
    return relative, opportunity.source_path


def _matching_lock(
    opportunity: KernelOpportunity, provenance: RunProvenance
) -> RepositoryLock | None:
    if opportunity.source_root is None:
        return None
    try:
        root = opportunity.source_root.resolve(strict=True)
    except OSError:
        return None
    matches = [
        lock
        for lock in provenance.source_locks
        if lock.name.lower() == opportunity.origin_library
        and lock.exact
        and Path(lock.path).resolve() == root
    ]
    return matches[0] if len(matches) == 1 else None


def _safe_package_relative(opportunity: KernelOpportunity) -> str | None:
    if opportunity.source_root is None or opportunity.source_path is None:
        return None
    try:
        relative = opportunity.source_path.resolve(strict=True).relative_to(
            opportunity.source_root.resolve(strict=True)
        )
    except (OSError, ValueError):
        return None
    parts = relative.parts
    if (
        len(parts) < 2
        or parts[0] != opportunity.origin_library
        or relative.suffix != ".py"
        or ".." in parts
    ):
        return None
    return relative.as_posix()


def _prepare_artifact_root(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise IntegrityError("Overlay artifact root is unsafe", "unsafe_path")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise IntegrityError("Overlay artifact root is not a directory", "unsafe_path")
    return path.resolve()


def _config_image(path: Path) -> str:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ContractError("Measurement config is unsafe", "invalid_replay_config")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    benchmark = value.get("benchmark") if isinstance(value, Mapping) else None
    image = benchmark.get("docker_image") if isinstance(benchmark, Mapping) else None
    if not isinstance(image, str) or not image.strip():
        raise ContractError("Measurement config lacks docker image", "invalid_replay_config")
    return image.strip()


def _validate_parent(
    reference: str,
    parent: ContainerImage,
    provenance: RunProvenance,
    *,
    accepted: tuple[AcceptedCandidate, ...],
    anchor_generation: int,
) -> ContainerImage:
    if not _IMAGE_ID.fullmatch(parent.image_id):
        raise IntegrityError("Parent image is not immutable", "invalid_image_id")
    if accepted:
        return validate_accepted_overlay_parent(
            reference=reference,
            inspected=parent,
            provenance=provenance,
            accepted=accepted,
            anchor_generation=anchor_generation,
        )
    if anchor_generation != 0:
        raise IntegrityError(
            "Initial overlay generation is not zero", "overlay_ancestry_mismatch"
        )
    if parent.image_id != provenance.container.image_id:
        raise IntegrityError("Parent differs from provenance", "image_identity_mismatch")
    if _IMAGE_ID.fullmatch(reference):
        if reference != parent.image_id:
            raise IntegrityError("Exact parent image changed identity", "image_identity_mismatch")
    elif reference != provenance.container.requested_image:
        raise IntegrityError("Mutable parent differs from provenance", "image_identity_mismatch")
    allowed = set(provenance.container.repo_digests)
    if parent.verified_repo_digest is not None:
        observed = {parent.verified_repo_digest}
        matching = (
            (parent.verified_repo_digest,)
            if not allowed or parent.verified_repo_digest in allowed
            else ()
        )
    else:
        observed = set(parent.repo_digests)
        if not _IMAGE_ID.fullmatch(reference):
            repository = _repository_name(reference)
            observed = {
                item for item in observed if _repository_name(item) == repository
            }
        matching = tuple(sorted(observed & allowed))
    if len(matching) != 1:
        raise IntegrityError(
            "Parent image has no unique provenance-approved repo digest",
            "immutable_parent_locator_unresolved",
            {
                "parent_image_id": parent.image_id,
                "observed_repo_digests": sorted(observed),
                "allowed_repo_digests": sorted(allowed),
            },
        )
    return ContainerImage(
        parent.reference,
        parent.image_id,
        parent.repo_digests,
        matching[0],
    )


def _repository_name(reference: str) -> str:
    name = reference.split("@", 1)[0]
    slash = name.rfind("/")
    colon = name.rfind(":")
    return name[:colon] if colon > slash else name


def _failed_deployment(
    request: CandidateDeploymentRequest,
    reason: str,
    details: Mapping[str, Any],
    *,
    infrastructure_failure: bool,
) -> CandidateDeployment:
    candidate = request.candidate
    return CandidateDeployment(
        candidate.candidate_id or candidate.attempt_id,
        False,
        reason,
        request.benchmark_measurement,
        request.benchmark_diagnostic,
        request.benchmark_replay or request.benchmark_measurement,
        request.workload_semantics_sha256,
        candidate.candidate_source_sha256 or "",
        None,
        ValidationLevel.NONE,
        False,
        {"schema_version": 1, "failure": reason, "details": dict(details)},
        infrastructure_failure,
    )


__all__ = [
    "DockerOverlayDeployment",
    "GitSourceLockVerifier",
    "OverlayOnlyFinalDelivery",
    "SourceLockVerifier",
]
