"""Disposable cumulative Git source materialization for formal E2E delivery."""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from apex.core import ContractError, IntegrityError, sha256_file, sha256_json
from apex.delivery import CapturedRepositoryPatch, capture_repository_patch
from apex.execution import SubprocessSupervisor
from apex.runtime import RepositoryLock, canonical_repository

from .services import AcceptedCandidate, FinalDeliveryRequest
from .source_delivery_models import FormalRepositoryProfile, FormalSourceDeliveryProfile


class CumulativeSourceMaterializer:
    """Clone exact locks and reproduce the ordered accepted source stack."""

    def __init__(self, supervisor: SubprocessSupervisor) -> None:
        self._supervisor = supervisor

    def materialize(
        self,
        request: FinalDeliveryRequest,
        profile: FormalSourceDeliveryProfile,
        destination: Path,
    ) -> dict[str, Path]:
        destination.mkdir(parents=True)
        locks = {item.name: item for item in request.provenance.source_locks}
        roots: dict[str, Path] = {}
        for repository in profile.repositories:
            lock = locks.get(repository.repository_id)
            if lock is None or not lock.exact:
                raise ContractError("Exact source lock is missing", "source_lock_unresolved")
            self._verify_runtime_lock(lock, repository)
            target = destination / repository.repository_id
            self._clone_exact_source(lock, target)
            roots[repository.repository_id] = target
        for accepted in request.accepted:
            repository = accepted.opportunity.origin_library
            self._apply_accepted_bytes(accepted, locks[repository], roots[repository])
        return roots

    def capture(
        self,
        request: FinalDeliveryRequest,
        profile: FormalSourceDeliveryProfile,
        worktrees: Mapping[str, Path],
    ) -> tuple[CapturedRepositoryPatch, ...]:
        candidate_ids = _candidate_ids(request, profile)
        patches = []
        for order, repository in enumerate(profile.repositories):
            combined = (
                f"accepted-{sha256_json(candidate_ids[repository.repository_id])[:24]}"
            )
            patches.append(
                capture_repository_patch(
                    repository_id=repository.repository_id,
                    base_root=Path(_source_lock(request, repository.repository_id).path),
                    candidate_root=worktrees[repository.repository_id],
                    patch_path=f"patches/{order:02d}-{repository.repository_id}.patch",
                    order=order,
                    dependencies=repository.dependencies,
                    editable_allowlist=repository.editable_allowlist,
                    build_recipe_sha256=profile.recipe.computed_sha256,
                    accepted_candidate_id=combined,
                    anchor_generation=len(request.accepted),
                    license_id=repository.license_id,
                    runtime_component=repository.runtime_component,
                    supervisor=self._supervisor,
                )
            )
        return tuple(patches)

    def fingerprints(self, roots: Mapping[str, Path]) -> dict[str, str]:
        return {
            repository_id: sha256_json(
                {
                    "diff": self._git(
                        root,
                        "diff",
                        "--binary",
                        "--full-index",
                        "--no-ext-diff",
                        "--no-textconv",
                        "HEAD",
                        "--",
                    )
                }
            )
            for repository_id, root in roots.items()
        }

    def _verify_runtime_lock(
        self, lock: RepositoryLock, profile: FormalRepositoryProfile
    ) -> None:
        root = Path(lock.path).resolve(strict=True)
        if (
            lock.name != profile.repository_id
            or canonical_repository(lock.url)
            != canonical_repository(profile.trusted_url)
            or self._git(root, "rev-parse", "--show-toplevel") != str(root)
            or self._git(root, "rev-parse", "HEAD") != lock.commit
            or self._git(root, "rev-parse", "HEAD^{tree}") != lock.tree
            or self._git(
                root, "status", "--porcelain=v1", "--untracked-files=all"
            )
        ):
            raise IntegrityError("Exact source lock drifted", "source_lock_drift")

    def _clone_exact_source(self, lock: RepositoryLock, target: Path) -> None:
        source = Path(lock.path).resolve(strict=True)
        self._git(
            target.parent,
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            str(source),
            str(target),
            timeout=600,
        )
        self._git(target, "checkout", "--detach", lock.commit)
        self._git(target, "remote", "set-url", "origin", lock.url)
        if self._git(target, "rev-parse", "HEAD^{tree}") != lock.tree:
            raise IntegrityError(
                "Cloned source tree differs from lock", "repository_tree_mismatch"
            )

    def _apply_accepted_bytes(
        self, accepted: AcceptedCandidate, lock: RepositoryLock, target_root: Path
    ) -> None:
        opportunity = accepted.opportunity
        candidate = accepted.candidate
        if opportunity.source_root is None or opportunity.source_path is None:
            raise ContractError(
                "Accepted source path is unresolved", "source_lock_unresolved"
            )
        if opportunity.source_root.resolve(strict=True) != Path(lock.path).resolve(
            strict=True
        ):
            raise IntegrityError(
                "Accepted source uses another repository", "source_lock_mismatch"
            )
        _verify_candidate_digests(candidate, Path(lock.path))
        for relative in candidate.changed_files:
            _copy_candidate_file(candidate.workspace, relative, target_root)

    def _git(self, cwd: Path, *args: str, timeout: int = 60) -> str:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        result = self._supervisor.run(
            ("git", *args),
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout,
        )
        if result.exit_code != 0 or result.timed_out or result.stdout_truncated:
            raise IntegrityError(
                "Git source delivery operation failed",
                "source_materialization_failed",
            )
        return result.stdout.strip()


def _copy_candidate_file(workspace: Path, relative: str, target_root: Path) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ContractError(
            "Only safe Python/Triton source is deliverable", "unsupported_delivery"
        )
    source = workspace.joinpath(*path.parts)
    destination = target_root.joinpath(*path.parts)
    if (
        source.is_symlink()
        or not source.is_file()
        or destination.is_symlink()
        or not destination.is_file()
    ):
        raise IntegrityError(
            "Accepted source file is unsafe", "invalid_frozen_candidate"
        )
    destination_mode = destination.stat().st_mode & 0o777
    shutil.copyfile(source, destination)
    destination.chmod(destination_mode)


def _verify_candidate_digests(candidate: object, base_root: Path) -> None:
    editable = tuple(getattr(candidate, "editable_files"))
    workspace = Path(getattr(candidate, "workspace"))
    baseline = _source_set_digest(base_root, editable)
    optimized = _source_set_digest(workspace, editable, mode_root=base_root)
    if (
        baseline != getattr(candidate, "baseline_source_sha256")
        or optimized != getattr(candidate, "candidate_source_sha256")
    ):
        raise IntegrityError(
            "Candidate source lineage changed", "candidate_lineage_mismatch"
        )


def _source_set_digest(
    root: Path, paths: Sequence[str], *, mode_root: Path | None = None
) -> str:
    values: list[dict[str, object]] = []
    for relative in paths:
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise IntegrityError(
                "Candidate source is not regular", "invalid_frozen_candidate"
            )
        mode_path = mode_root.joinpath(*PurePosixPath(relative).parts) if mode_root else path
        values.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "mode": mode_path.stat().st_mode & 0o777,
            }
        )
    return sha256_json({"schema_version": 1, "files": values})


def _candidate_ids(
    request: FinalDeliveryRequest, profile: FormalSourceDeliveryProfile
) -> dict[str, tuple[str | None, ...]]:
    return {
        repository.repository_id: tuple(
            item.candidate.candidate_id
            for item in request.accepted
            if item.opportunity.origin_library == repository.repository_id
        )
        for repository in profile.repositories
    }


def _source_lock(request: FinalDeliveryRequest, repository_id: str) -> RepositoryLock:
    matches = tuple(
        item for item in request.provenance.source_locks if item.name == repository_id
    )
    if len(matches) != 1 or not matches[0].exact:
        raise ContractError("Exact source lock is missing", "source_lock_unresolved")
    return matches[0]


__all__ = ["CumulativeSourceMaterializer"]
