"""Isolated candidate workspace creation and post-agent source freeze."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_file
from apex.intake import ResolvedTaskSpec


_IGNORED_DIRECTORIES = {".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    sha256: str
    mode: int


@dataclass(frozen=True, slots=True)
class CandidateFreeze:
    root: Path
    changed_files: tuple[str, ...]
    baseline: Mapping[str, FileFingerprint]
    candidate: Mapping[str, FileFingerprint]


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _IGNORED_DIRECTORIES or Path(name).suffix in _IGNORED_SUFFIXES}


def _fingerprint_tree(root: Path) -> dict[str, FileFingerprint]:
    result: dict[str, FileFingerprint] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            raise IntegrityError(f"Workspace contains symlink: {relative}", "workspace_symlink")
        if not path.is_file() or path.suffix in _IGNORED_SUFFIXES:
            continue
        stat = os.lstat(path)
        if stat.st_nlink != 1:
            raise IntegrityError(f"Workspace contains hard-linked file: {relative}", "workspace_hardlink")
        result[relative.as_posix()] = FileFingerprint(sha256_file(path), stat.st_mode & 0o777)
    return result


class CandidateWorkspace:
    """Copy a resolved task and reject any non-allowlisted agent changes."""

    def __init__(
        self,
        *,
        root: Path,
        anchor: Path,
        baseline: Mapping[str, FileFingerprint],
        allowed_files: tuple[str, ...],
    ) -> None:
        self.root = root
        self._anchor = anchor
        self._baseline = dict(baseline)
        self._allowed = frozenset(allowed_files)

    @classmethod
    def create(
        cls,
        resolved: ResolvedTaskSpec,
        *,
        destination: Path,
        anchor: Path | None = None,
    ) -> "CandidateWorkspace":
        if destination.exists():
            raise ContractError("candidate destination already exists", "candidate_workspace_exists")
        source = (anchor or resolved.workspace).resolve(strict=True)
        baseline = _fingerprint_tree(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, symlinks=False, ignore=_ignore)
        copied = _fingerprint_tree(destination)
        if copied != baseline:
            raise IntegrityError("candidate copy does not match anchor", "candidate_copy_mismatch")
        return cls(
            root=destination,
            anchor=source,
            baseline=baseline,
            allowed_files=resolved.task.editable_files,
        )

    def freeze(self, *, destination: Path | None = None) -> CandidateFreeze:
        agent_root = self.root
        candidate = _fingerprint_tree(agent_root)
        paths = set(self._baseline) | set(candidate)
        changed = tuple(sorted(path for path in paths if self._baseline.get(path) != candidate.get(path)))
        forbidden = sorted(set(changed).difference(self._allowed))
        if forbidden:
            raise IntegrityError(
                f"Agent changed non-editable files: {', '.join(forbidden)}",
                "undeclared_agent_edit",
                {"paths": forbidden},
            )
        mode_changed = sorted(
            path
            for path in changed
            if path in self._baseline
            and path in candidate
            and self._baseline[path].mode != candidate[path].mode
        )
        if mode_changed:
            raise IntegrityError(
                f"Agent changed source file mode: {', '.join(mode_changed)}",
                "source_mode_change_forbidden",
            )
        deleted = sorted(path for path in changed if path not in candidate)
        if deleted:
            raise IntegrityError(
                f"Agent deleted editable files: {', '.join(deleted)}",
                "editable_source_deleted",
            )
        projection = destination or agent_root.with_name(f"{agent_root.name}.frozen")
        projected = self._materialize_projection(
            agent_root,
            projection,
            changed,
            candidate,
        )
        self.root = projection
        return CandidateFreeze(
            root=projection,
            changed_files=changed,
            baseline=self._baseline,
            candidate=projected,
        )

    def _materialize_projection(
        self,
        agent_root: Path,
        destination: Path,
        changed: tuple[str, ...],
        candidate: Mapping[str, FileFingerprint],
    ) -> dict[str, FileFingerprint]:
        if destination.exists():
            raise ContractError("candidate projection already exists", "candidate_projection_exists")
        if _fingerprint_tree(self._anchor) != self._baseline:
            raise IntegrityError("candidate anchor changed before freeze", "candidate_anchor_drift")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self._anchor, destination, symlinks=False, ignore=_ignore)
        if _fingerprint_tree(destination) != self._baseline:
            raise IntegrityError("candidate projection baseline mismatch", "candidate_projection_mismatch")
        expected = dict(self._baseline)
        for relative in changed:
            fingerprint = candidate[relative]
            content = (agent_root / relative).read_bytes()
            if sha256_bytes(content) != fingerprint.sha256:
                raise IntegrityError("candidate source changed during freeze", "candidate_freeze_race")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            os.chmod(target, fingerprint.mode)
            expected[relative] = fingerprint
        projected = _fingerprint_tree(destination)
        if projected != expected:
            raise IntegrityError("candidate projection differs from frozen source", "candidate_projection_mismatch")
        return projected
