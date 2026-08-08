"""Stateless E2E kernel candidate generation in an isolated source checkout."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from apex.core import (
    AgentBackendName,
    ContractError,
    IntegrityError,
    sha256_json,
)
from apex.execution import AgentRegistry, SubprocessSupervisor
from apex.ports import AgentRequest, AgentResult

from .candidate_fingerprint import (
    IGNORED_DIRECTORIES as _IGNORED_DIRECTORIES,
    IGNORED_SUFFIXES as _IGNORED_SUFFIXES,
    SourceFingerprint as _Fingerprint,
    fingerprint_git_tree as _fingerprint_git_tree,
    fingerprint_materialized_git_tree as _fingerprint_materialized_git_tree,
    fingerprint_tree as _fingerprint_tree,
    git_environment as _git_environment,
    git_output as _git_output,
    iter_bounded_tree as _iter_bounded_tree,
    preflight_materialized_tree as _preflight_materialized_tree,
)
from .candidate_snapshot import (
    FrozenCandidateSource,
    capture_frozen_sources as _capture_frozen_sources,
    frozen_candidate_source,
    materialize_frozen_sources,
    source_set_digest as _source_set_digest,
    validate_frozen_sources,
)
from .kernel_lane import KernelOpportunity


@dataclass(frozen=True, slots=True)
class E2ECandidateRequest:
    """Frozen input for one fresh backend invocation."""

    run_id: str
    attempt_id: str
    opportunity: KernelOpportunity
    prompt: str
    destination: Path
    backend: AgentBackendName
    model: str | None
    effort: str | None
    max_turns: int
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class E2ECandidate:
    """Frozen source candidate; it contains no evaluator verdict."""

    attempt_id: str
    candidate_id: str | None
    succeeded: bool
    reason_code: str
    workspace: Path
    editable_files: tuple[str, ...]
    changed_files: tuple[str, ...]
    baseline_source_sha256: str
    candidate_source_sha256: str | None
    agent_result: AgentResult
    frozen_sources: tuple[FrozenCandidateSource, ...] = ()


class CandidateWorker(Protocol):
    def generate(self, request: E2ECandidateRequest) -> E2ECandidate: ...


class SourceCandidateWorkspace:
    """Materialize one exact source anchor and freeze only the declared edit."""

    def __init__(
        self,
        *,
        root: Path,
        baseline: Mapping[str, _Fingerprint],
        editable_files: tuple[str, ...],
    ) -> None:
        self.root = root
        self._baseline = dict(baseline)
        self.editable_files = editable_files

    @classmethod
    def create(cls, opportunity: KernelOpportunity, *, destination: Path) -> "SourceCandidateWorkspace":
        if opportunity.source_root is None or opportunity.source_path is None:
            raise ContractError("Kernel source is unresolved", "source_unresolved")
        source_root = opportunity.source_root.resolve(strict=True)
        source_path = opportunity.source_path.resolve(strict=True)
        try:
            relative = source_path.relative_to(source_root).as_posix()
        except ValueError as error:
            raise IntegrityError("Kernel source escapes its repository", "source_outside_root") from error
        if destination.exists():
            raise ContractError("Candidate checkout already exists", "candidate_workspace_exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        git_checkout = _is_clean_git_checkout(source_root)
        if git_checkout:
            _clone_git_checkout(source_root, destination)
        else:
            shutil.copytree(source_root, destination, symlinks=True, ignore=_ignore)
        baseline = _fingerprint_git_tree(source_root) if git_checkout else _fingerprint_tree(source_root)
        copied = _fingerprint_git_tree(destination) if git_checkout else _fingerprint_tree(destination)
        if copied != baseline:
            raise IntegrityError("Candidate checkout differs from source anchor", "candidate_copy_mismatch")
        workspace = cls(root=destination.resolve(), baseline=baseline, editable_files=(relative,))
        workspace._git_checkout = git_checkout
        return workspace

    def freeze(self) -> tuple[tuple[str, ...], str, str | None]:
        git_checkout = bool(getattr(self, "_git_checkout", False))
        _preflight_materialized_tree(self.root, self._baseline)
        if git_checkout:
            _validate_git_control_directory(self.root)
        candidate = (
            _fingerprint_materialized_git_tree(self.root, self._baseline)
            if git_checkout
            else _fingerprint_tree(self.root)
        )
        all_paths = set(self._baseline) | set(candidate)
        changed = tuple(
            sorted(path for path in all_paths if self._baseline.get(path) != candidate.get(path))
        )
        forbidden = tuple(sorted(set(changed).difference(self.editable_files)))
        if git_checkout:
            forbidden = tuple(
                sorted(
                    set(forbidden)
                    .union(_materialized_gitlinks(self.root, self._baseline))
                    .union(_unexpected_worktree_files(self.root, self._baseline))
                )
            )
        if forbidden:
            raise IntegrityError(
                f"Agent changed undeclared source: {', '.join(forbidden)}",
                "undeclared_agent_edit",
                {"paths": list(forbidden)},
            )
        if any(path not in candidate for path in changed):
            raise IntegrityError("Agent deleted the kernel source", "editable_source_deleted")
        if any(
            self._baseline[path].mode != candidate[path].mode
            for path in changed
            if path in self._baseline and path in candidate
        ):
            raise IntegrityError("Agent changed source file mode", "source_mode_change_forbidden")
        baseline_digest = _source_set_digest(self._baseline, self.editable_files)
        candidate_digest = _source_set_digest(candidate, self.editable_files) if changed else None
        return changed, baseline_digest, candidate_digest

    @property
    def baseline_source_sha256(self) -> str:
        """Return the immutable editable-source digest captured before the agent ran."""

        return _source_set_digest(self._baseline, self.editable_files)


class AgentCandidateWorker:
    """Run Codex, Claude, or Cursor once; freeze bytes after the process exits."""

    def __init__(self, agents: AgentRegistry) -> None:
        self._agents = agents

    def generate(self, request: E2ECandidateRequest) -> E2ECandidate:
        workspace = SourceCandidateWorkspace.create(
            request.opportunity,
            destination=request.destination,
        )
        baseline_digest = workspace.baseline_source_sha256
        result = self._agents.get(request.backend).run(_agent_request(request, workspace))
        failure = _agent_failure_reason(result)
        if failure is not None:
            return _agent_rejection(request, workspace, result, baseline_digest, failure)
        return _candidate_after_agent(
            request,
            workspace,
            result,
            baseline_digest,
        )


def _agent_request(
    request: E2ECandidateRequest, workspace: SourceCandidateWorkspace
) -> AgentRequest:
    return AgentRequest(
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        backend=request.backend,
        prompt=request.prompt,
        workspace=workspace.root,
        allowed_files=workspace.editable_files,
        model=request.model,
        effort=request.effort,
        max_turns=request.max_turns,
        timeout_seconds=request.timeout_seconds,
    )


def _candidate_after_agent(
    request: E2ECandidateRequest,
    workspace: SourceCandidateWorkspace,
    result: AgentResult,
    baseline_digest: str,
) -> E2ECandidate:
    try:
        changed, observed_baseline_digest, candidate_digest = workspace.freeze()
    except IntegrityError as error:
        # The checkout and agent result now exist, so freeze integrity is a
        # candidate outcome. Pre-agent infrastructure failures still raise.
        return _freeze_rejection(request, workspace, result, baseline_digest, error)
    if observed_baseline_digest != baseline_digest:
        raise IntegrityError(
            "Candidate baseline identity changed during execution",
            "candidate_baseline_drift",
        )
    if not changed or candidate_digest is None:
        return E2ECandidate(
            request.attempt_id,
            None,
            False,
            "agent_made_no_source_change",
            workspace.root,
            workspace.editable_files,
            (),
            baseline_digest,
            None,
            result,
        )
    return _capture_successful_candidate(
        request, workspace, result, changed, baseline_digest, candidate_digest
    )


def _capture_successful_candidate(
    request: E2ECandidateRequest,
    workspace: SourceCandidateWorkspace,
    result: AgentResult,
    changed: tuple[str, ...],
    baseline_digest: str,
    candidate_digest: str,
) -> E2ECandidate:
    try:
        frozen_sources = _capture_frozen_sources(
            workspace,
            expected_source_sha256=candidate_digest,
        )
    except IntegrityError as error:
        if error.reason_code != "candidate_source_too_large":
            raise
        return _freeze_rejection(request, workspace, result, baseline_digest, error)
    candidate_id = f"candidate-{sha256_json({'attempt': request.attempt_id, 'source': candidate_digest})[:24]}"
    return E2ECandidate(
        request.attempt_id,
        candidate_id,
        True,
        "candidate_frozen",
        workspace.root,
        workspace.editable_files,
        changed,
        baseline_digest,
        candidate_digest,
        result,
        frozen_sources,
    )


def _agent_failure_reason(result: AgentResult) -> str | None:
    if result.candidate_capture_allowed:
        return None
    return result.candidate_rejection_reason or "agent_failed"


def _agent_rejection(
    request: E2ECandidateRequest,
    workspace: SourceCandidateWorkspace,
    result: AgentResult,
    baseline_digest: str,
    reason: str,
) -> E2ECandidate:
    return E2ECandidate(
        request.attempt_id,
        None,
        False,
        reason,
        workspace.root,
        workspace.editable_files,
        (),
        baseline_digest,
        None,
        result,
    )


def _freeze_rejection(
    request: E2ECandidateRequest,
    workspace: SourceCandidateWorkspace,
    result: AgentResult,
    baseline_digest: str,
    error: IntegrityError,
) -> E2ECandidate:
    return E2ECandidate(
        request.attempt_id,
        None,
        False,
        error.reason_code,
        workspace.root,
        workspace.editable_files,
        _rejected_changed_files(error, workspace.editable_files),
        baseline_digest,
        None,
        result,
    )


def _rejected_changed_files(
    error: IntegrityError, editable_files: tuple[str, ...]
) -> tuple[str, ...]:
    details = dict(error.details or {})
    paths = details.get("paths")
    if isinstance(paths, list) and all(isinstance(path, str) and path for path in paths):
        return tuple(sorted(set(paths)))
    if error.reason_code in {"editable_source_deleted", "source_mode_change_forbidden"}:
        return editable_files
    return ()


def _materialized_gitlinks(
    root: Path, baseline: Mapping[str, _Fingerprint]
) -> set[str]:
    """Reject any agent-created content below an intentionally uninitialized gitlink."""

    changed: set[str] = set()
    for relative, fingerprint in baseline.items():
        if fingerprint.kind != "gitlink":
            continue
        path = root.joinpath(*relative.split("/"))
        if path.is_symlink() or path.is_file():
            changed.add(relative)
        elif path.is_dir() and any(path.iterdir()):
            changed.add(relative)
    return changed


def _unexpected_worktree_files(
    root: Path, baseline: Mapping[str, _Fingerprint]
) -> set[str]:
    """Find files even when mutable Git excludes try to hide them."""

    expected = set(baseline)
    gitlinks = tuple(
        path for path, fingerprint in baseline.items() if fingerprint.kind == "gitlink"
    )
    unexpected: set[str] = set()
    for value, path, _metadata in _iter_bounded_tree(root):
        if path.is_symlink() or path.is_file():
            if value not in expected:
                unexpected.add(_gitlink_owner(value, gitlinks) or value)
    return unexpected


def _gitlink_owner(path: str, gitlinks: tuple[str, ...]) -> str | None:
    return next(
        (gitlink for gitlink in gitlinks if path.startswith(f"{gitlink}/")),
        None,
    )


def _validate_git_control_directory(root: Path) -> None:
    control = root / ".git"
    if control.is_symlink() or not control.is_dir():
        raise IntegrityError(
            "Candidate Git control directory is unsafe",
            "workspace_git_metadata_unsafe",
        )


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in _IGNORED_DIRECTORIES or Path(name).suffix in _IGNORED_SUFFIXES
    }


def _is_clean_git_checkout(root: Path) -> bool:
    supervisor = SubprocessSupervisor(max_output_bytes=1024 * 1024)
    result = supervisor.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=root,
        environment=_git_environment(),
        timeout_seconds=30,
    )
    if result.exit_code != 0 or result.timed_out:
        return False
    try:
        if Path(result.stdout.strip()).resolve(strict=True) != root.resolve(strict=True):
            return False
        _git_output(root, ("git", "remote", "get-url", "origin"), timeout=30)
        status = _git_output(
            root,
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            timeout=30,
        )
        return not status.strip()
    except (ContractError, OSError):
        return False


def _clone_git_checkout(source: Path, destination: Path) -> None:
    supervisor = SubprocessSupervisor(max_output_bytes=4 * 1024 * 1024)

    def run(argv: tuple[str, ...], cwd: Path) -> str:
        result = supervisor.run(
            argv,
            cwd=cwd,
            environment=_git_environment(),
            timeout_seconds=600,
        )
        if result.exit_code != 0 or result.timed_out:
            raise IntegrityError("Cannot materialize candidate Git checkout", "candidate_copy_failed")
        return result.stdout.strip()

    commit = run(("git", "rev-parse", "HEAD"), source)
    origin = run(("git", "remote", "get-url", "origin"), source)
    run(
        ("git", "clone", "--no-hardlinks", "--no-checkout", str(source), str(destination)),
        source.parent,
    )
    run(("git", "checkout", "--detach", commit), destination)
    run(("git", "remote", "set-url", "origin", origin), destination)


__all__ = [
    "AgentCandidateWorker",
    "CandidateWorker",
    "E2ECandidate",
    "E2ECandidateRequest",
    "FrozenCandidateSource",
    "SourceCandidateWorkspace",
    "frozen_candidate_source",
    "materialize_frozen_sources",
    "validate_frozen_sources",
]
