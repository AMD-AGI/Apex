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
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from apex.execution import AgentRegistry, SubprocessSupervisor
from apex.ports import AgentRequest, AgentResult

from .kernel_lane import KernelOpportunity


_IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


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


class CandidateWorker(Protocol):
    def generate(self, request: E2ECandidateRequest) -> E2ECandidate: ...


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    sha256: str
    mode: int
    kind: str = "regular"


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
        _purge_generated_artifacts(self.root)
        if git_checkout:
            _validate_git_control_directory(self.root)
        candidate = _fingerprint_git_tree(self.root) if git_checkout else _fingerprint_tree(self.root)
        all_paths = set(self._baseline) | set(candidate)
        changed = tuple(
            sorted(path for path in all_paths if self._baseline.get(path) != candidate.get(path))
        )
        forbidden = tuple(sorted(set(changed).difference(self.editable_files)))
        if git_checkout:
            observed_changes = _git_changed_paths(self.root)
            forbidden = tuple(
                sorted(
                    set(forbidden)
                    .union(observed_changes.difference(self.editable_files))
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


class AgentCandidateWorker:
    """Run Codex, Claude, or Cursor once; freeze bytes after the process exits."""

    def __init__(self, agents: AgentRegistry) -> None:
        self._agents = agents

    def generate(self, request: E2ECandidateRequest) -> E2ECandidate:
        workspace = SourceCandidateWorkspace.create(
            request.opportunity,
            destination=request.destination,
        )
        result = self._agents.get(request.backend).run(
            AgentRequest(
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
        )
        changed, baseline_digest, candidate_digest = workspace.freeze()
        failure = _agent_failure_reason(result)
        if failure is not None:
            return E2ECandidate(
                request.attempt_id,
                None,
                False,
                failure,
                workspace.root,
                workspace.editable_files,
                changed,
                baseline_digest,
                candidate_digest,
                result,
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
        )


def _agent_failure_reason(result: AgentResult) -> str | None:
    if result.timed_out:
        return "agent_timeout"
    if result.budget_enforcement_failed:
        return "agent_turn_budget_unverifiable"
    if result.budget_exceeded:
        return "agent_turn_budget_exceeded"
    return None if result.succeeded else "agent_failed"


def candidate_file_paths(candidate: E2ECandidate) -> tuple[Path, ...]:
    return tuple(
        candidate.workspace.joinpath(*relative.split("/"))
        for relative in candidate.editable_files
    )


def make_candidate_read_only(candidate: E2ECandidate) -> None:
    """Seal declared source bytes before evaluator-only phases."""

    for path in candidate_file_paths(candidate):
        mode = path.stat().st_mode & 0o777
        path.chmod(mode & ~0o222)


def _source_set_digest(
    values: Mapping[str, _Fingerprint], editable_files: tuple[str, ...]
) -> str:
    try:
        payload = [
            {"path": path, "sha256": values[path].sha256, "mode": values[path].mode}
            for path in editable_files
        ]
    except KeyError as error:
        raise IntegrityError("Declared kernel source is missing", "missing_candidate_source") from error
    return sha256_json({"schema_version": 1, "files": payload})


def _fingerprint_tree(root: Path) -> dict[str, _Fingerprint]:
    result: dict[str, _Fingerprint] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            target = path.resolve(strict=True)
            try:
                target.relative_to(root.resolve(strict=True))
            except ValueError as error:
                raise IntegrityError(
                    f"Source symlink escapes checkout: {relative}", "workspace_symlink_escape"
                ) from error
            metadata = path.lstat()
            result[relative.as_posix()] = _Fingerprint(
                sha256_bytes(os.readlink(path).encode()),
                metadata.st_mode & 0o777,
                "symlink",
            )
            continue
        if not path.is_file() or path.suffix in _IGNORED_SUFFIXES:
            continue
        metadata = os.lstat(path)
        if metadata.st_nlink != 1:
            raise IntegrityError(
                f"Source checkout contains hard link: {relative}",
                "workspace_hardlink",
            )
        result[relative.as_posix()] = _Fingerprint(sha256_file(path), metadata.st_mode & 0o777)
    return result


def _fingerprint_git_tree(root: Path) -> dict[str, _Fingerprint]:
    """Fingerprint tracked blobs, safe symlinks, and gitlinks without traversing submodules."""

    output = _git_output(root, ("git", "ls-files", "--stage", "-z"), timeout=60)
    result: dict[str, _Fingerprint] = {}
    for raw in output.split("\0"):
        if not raw:
            continue
        metadata, relative = raw.split("\t", 1)
        mode, object_id, stage = metadata.split(" ", 2)
        if stage != "0":
            raise IntegrityError("Source checkout contains an unmerged path", "dirty_source_base")
        path = root.joinpath(*relative.split("/"))
        if mode == "160000":
            result[relative] = _Fingerprint(
                sha256_json({"gitlink": object_id}), int(mode, 8), "gitlink"
            )
            continue
        if mode == "120000":
            if not path.is_symlink():
                raise IntegrityError("Tracked symlink is not materialized", "candidate_copy_mismatch")
            target = path.resolve(strict=True)
            try:
                target.relative_to(root.resolve(strict=True))
            except ValueError as error:
                raise IntegrityError(
                    f"Tracked symlink escapes checkout: {relative}",
                    "workspace_symlink_escape",
                ) from error
            result[relative] = _Fingerprint(
                sha256_bytes(os.readlink(path).encode()), int(mode, 8), "symlink"
            )
            continue
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("Tracked source is not a regular file", "candidate_copy_mismatch")
        if path.stat().st_nlink != 1:
            raise IntegrityError("Tracked source is hard linked", "workspace_hardlink")
        result[relative] = _Fingerprint(sha256_file(path), int(mode, 8), "regular")
    return result


def _git_changed_paths(root: Path) -> set[str]:
    changed: set[str] = set()
    for argv in (
        ("git", "diff", "--name-only", "-z"),
        ("git", "diff", "--cached", "--name-only", "-z"),
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
    ):
        changed.update(item for item in _git_output(root, argv, timeout=60).split("\0") if item)
    return changed


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
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        value = relative.as_posix()
        if path.is_symlink() or path.is_file():
            if value not in expected:
                unexpected.add(_gitlink_owner(value, gitlinks) or value)
    return unexpected


def _gitlink_owner(path: str, gitlinks: tuple[str, ...]) -> str | None:
    return next(
        (gitlink for gitlink in gitlinks if path.startswith(f"{gitlink}/")),
        None,
    )


def _purge_generated_artifacts(root: Path) -> None:
    """Remove interpreter/test caches from the isolated checkout before freezing."""

    directories = sorted(
        (
            path
            for path in root.rglob("*")
            if path.name in _IGNORED_DIRECTORIES and path.name != ".git"
            and ".git" not in path.relative_to(root).parts
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    for path in root.rglob("*"):
        if (
            ".git" not in path.relative_to(root).parts
            and path.is_file()
            and path.suffix in _IGNORED_SUFFIXES
        ):
            path.unlink()


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


def _git_output(root: Path, argv: tuple[str, ...], *, timeout: int) -> str:
    result = SubprocessSupervisor(max_output_bytes=16 * 1024 * 1024).run(
        argv,
        cwd=root,
        environment=_git_environment(),
        timeout_seconds=timeout,
    )
    if result.exit_code != 0 or result.timed_out or result.stdout_truncated:
        raise IntegrityError("Git source inspection failed", "repository_inspection_failed")
    return result.stdout


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    return environment


__all__ = [
    "AgentCandidateWorker",
    "CandidateWorker",
    "E2ECandidate",
    "E2ECandidateRequest",
    "SourceCandidateWorkspace",
    "candidate_file_paths",
    "make_candidate_read_only",
]
