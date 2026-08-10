"""Bounded filesystem and Git fingerprints for isolated E2E source trees."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from apex.core import IntegrityError, sha256_bytes, sha256_file, sha256_json
from apex.execution import SubprocessSupervisor, build_subprocess_environment


IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
MAX_FROZEN_SOURCE_BYTES = 16 * 1024 * 1024
MAX_WORKSPACE_CHANGED_BYTES = 32 * 1024 * 1024
MAX_WORKSPACE_DEPTH = 64
MAX_WORKSPACE_ENTRIES = 20_000


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """One materialized source entry used to compute a source-set identity."""

    sha256: str
    mode: int
    kind: str = "regular"
    size: int = 0


def fingerprint_tree(root: Path) -> dict[str, SourceFingerprint]:
    """Fingerprint a non-Git source tree without following escaping links."""

    result: dict[str, SourceFingerprint] = {}
    for relative, path, metadata in iter_bounded_tree(root):
        relative_path = Path(relative)
        if stat.S_ISLNK(metadata.st_mode):
            result[relative] = _fingerprint_safe_symlink(root, path, relative_path)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            continue
        if metadata.st_nlink != 1:
            raise IntegrityError(
                f"Source checkout contains hard link: {relative}",
                "workspace_hardlink",
            )
        result[relative] = SourceFingerprint(
            sha256_file(path), metadata.st_mode & 0o777, "regular", metadata.st_size
        )
    return result


def _fingerprint_safe_symlink(
    root: Path, path: Path, relative: Path
) -> SourceFingerprint:
    target = path.resolve(strict=True)
    try:
        target.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise IntegrityError(
            f"Source symlink escapes checkout: {relative}", "workspace_symlink_escape"
        ) from error
    metadata = path.lstat()
    return SourceFingerprint(
        sha256_bytes(os.readlink(path).encode()),
        metadata.st_mode & 0o777,
        "symlink",
        metadata.st_size,
    )


def fingerprint_git_tree(root: Path) -> dict[str, SourceFingerprint]:
    """Fingerprint tracked blobs, safe symlinks, and untraversed gitlinks."""

    output = git_output(root, ("git", "ls-files", "--stage", "-z"), timeout=60)
    result: dict[str, SourceFingerprint] = {}
    for raw in output.split("\0"):
        if raw:
            _metadata, relative = raw.split("\t", 1)
            if _ignored_source_path(relative):
                continue
            if len(result) >= MAX_WORKSPACE_ENTRIES:
                _workspace_limit(
                    "entries", MAX_WORKSPACE_ENTRIES, len(result) + 1
                )
            _add_git_entry(root, result, raw)
    return result


def _ignored_source_path(relative: str) -> bool:
    path = Path(relative)
    return (
        any(part in IGNORED_DIRECTORIES for part in path.parts)
        or path.suffix in IGNORED_SUFFIXES
    )


def _add_git_entry(
    root: Path, result: dict[str, SourceFingerprint], raw: str
) -> None:
    metadata, relative = raw.split("\t", 1)
    mode, object_id, stage = metadata.split(" ", 2)
    if stage != "0":
        raise IntegrityError("Source checkout contains an unmerged path", "dirty_source_base")
    path = root.joinpath(*relative.split("/"))
    reject_symlink_parents(root, relative)
    if mode == "160000":
        result[relative] = SourceFingerprint(
            sha256_json({"gitlink": object_id}), int(mode, 8) & 0o777, "gitlink"
        )
    elif mode == "120000":
        result[relative] = _fingerprint_tracked_symlink(root, path, relative)
    else:
        result[relative] = _fingerprint_tracked_file(path)


def _fingerprint_tracked_symlink(
    root: Path, path: Path, relative: str
) -> SourceFingerprint:
    if not path.is_symlink():
        raise IntegrityError("Tracked symlink is not materialized", "candidate_copy_mismatch")
    target = path.resolve(strict=True)
    try:
        target.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise IntegrityError(
            f"Tracked symlink escapes checkout: {relative}", "workspace_symlink_escape"
        ) from error
    metadata = path.lstat()
    return SourceFingerprint(
        sha256_bytes(os.readlink(path).encode()),
        metadata.st_mode & 0o777,
        "symlink",
        metadata.st_size,
    )


def _fingerprint_tracked_file(path: Path) -> SourceFingerprint:
    if path.is_symlink() or not path.is_file():
        raise IntegrityError("Tracked source is not a regular file", "candidate_copy_mismatch")
    metadata = path.stat()
    if metadata.st_nlink != 1:
        raise IntegrityError("Tracked source is hard linked", "workspace_hardlink")
    return SourceFingerprint(
        sha256_file(path), metadata.st_mode & 0o777, "regular", metadata.st_size
    )


def fingerprint_materialized_git_tree(
    root: Path, baseline: Mapping[str, SourceFingerprint]
) -> dict[str, SourceFingerprint]:
    """Inspect post-agent bytes without executing Git or mutable checkout config."""

    result: dict[str, SourceFingerprint] = {}
    for relative, expected in baseline.items():
        observed = _fingerprint_materialized_git_entry(root, relative, expected)
        if observed is not None:
            result[relative] = observed
    return result


def _fingerprint_materialized_git_entry(
    root: Path, relative: str, expected: SourceFingerprint
) -> SourceFingerprint | None:
    path = root.joinpath(*relative.split("/"))
    if expected.kind == "gitlink":
        if path.is_symlink() or path.is_file() or (path.is_dir() and any(path.iterdir())):
            return SourceFingerprint("materialized", 0, "gitlink")
        return expected
    reject_symlink_parents(root, relative)
    if expected.kind == "symlink":
        if not path.is_symlink():
            return None
        return _fingerprint_tracked_symlink(root, path, relative)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        return SourceFingerprint("unsafe", 0, "unsafe")
    return SourceFingerprint(
        sha256_file(path), metadata.st_mode & 0o777, "regular", metadata.st_size
    )


def preflight_materialized_tree(
    root: Path, baseline: Mapping[str, SourceFingerprint]
) -> None:
    """Bound post-agent entry, depth, and changed-byte work before hashing."""

    changed_bytes = 0
    for relative, _path, metadata in iter_bounded_tree(root):
        if not stat.S_ISREG(metadata.st_mode):
            continue
        expected = baseline.get(relative)
        changed = (
            expected is None
            or expected.kind != "regular"
            or metadata.st_size != expected.size
        )
        if not changed:
            continue
        if metadata.st_size > MAX_FROZEN_SOURCE_BYTES:
            raise IntegrityError(
                "Agent-created source exceeds the workspace byte limit",
                "candidate_source_too_large",
                {
                    "path": relative,
                    "limit_bytes": MAX_FROZEN_SOURCE_BYTES,
                    "observed_bytes": metadata.st_size,
                },
            )
        changed_bytes += metadata.st_size
        if changed_bytes > MAX_WORKSPACE_CHANGED_BYTES:
            raise IntegrityError(
                "Candidate workspace changed-byte budget was exceeded",
                "candidate_workspace_too_large",
                {
                    "dimension": "changed_bytes",
                    "limit": MAX_WORKSPACE_CHANGED_BYTES,
                    "observed": changed_bytes,
                },
            )


def iter_bounded_tree(
    root: Path,
) -> Iterator[tuple[str, Path, os.stat_result]]:
    """Yield non-ignored entries without materializing an unbounded directory."""

    stack: list[tuple[Path, int]] = [(root, 0)]
    observed = 0
    try:
        while stack:
            directory, parent_depth = stack.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    observed += 1
                    if observed > MAX_WORKSPACE_ENTRIES:
                        _workspace_limit("entries", MAX_WORKSPACE_ENTRIES, observed)
                    depth = parent_depth + 1
                    if depth > MAX_WORKSPACE_DEPTH:
                        _workspace_limit("depth", MAX_WORKSPACE_DEPTH, depth)
                    path = Path(entry.path)
                    metadata = entry.stat(follow_symlinks=False)
                    ignored_directory = (
                        stat.S_ISDIR(metadata.st_mode)
                        and entry.name in IGNORED_DIRECTORIES
                    )
                    if ignored_directory:
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        stack.append((path, depth))
                    if path.suffix in IGNORED_SUFFIXES:
                        continue
                    yield path.relative_to(root).as_posix(), path, metadata
    except OSError as error:
        raise IntegrityError(
            "Candidate workspace could not be inspected safely",
            "candidate_workspace_inspection_failed",
        ) from error


def _workspace_limit(dimension: str, limit: int, observed: int) -> None:
    raise IntegrityError(
        "Candidate workspace traversal budget was exceeded",
        "candidate_workspace_too_large",
        {"dimension": dimension, "limit": limit, "observed": observed},
    )


def reject_symlink_parents(root: Path, relative: str) -> None:
    """Reject a relative path whose materialized parent traverses a symlink."""

    current = root
    for part in Path(relative).parts[:-1]:
        current /= part
        try:
            if current.is_symlink():
                raise IntegrityError(
                    "Candidate source parent is a symlink",
                    "workspace_symlink_escape",
                    {"path": relative},
                )
        except OSError as error:
            raise IntegrityError(
                "Candidate source parent could not be inspected",
                "candidate_source_capture_unsafe",
                {"path": relative},
            ) from error


def git_output(root: Path, argv: tuple[str, ...], *, timeout: int) -> str:
    """Run one fixed Git inspection command under the trusted Git environment."""

    result = SubprocessSupervisor(max_output_bytes=16 * 1024 * 1024).run(
        argv,
        cwd=root,
        environment=git_environment(),
        timeout_seconds=timeout,
    )
    if result.exit_code != 0 or result.timed_out or result.stdout_truncated:
        raise IntegrityError("Git source inspection failed", "repository_inspection_failed")
    return result.stdout


def git_environment() -> dict[str, str]:
    """Return an environment that excludes ambient Git and Python injection."""

    return build_subprocess_environment(
        fixed={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )


__all__ = [
    "IGNORED_DIRECTORIES",
    "IGNORED_SUFFIXES",
    "MAX_FROZEN_SOURCE_BYTES",
    "MAX_WORKSPACE_CHANGED_BYTES",
    "MAX_WORKSPACE_DEPTH",
    "MAX_WORKSPACE_ENTRIES",
    "SourceFingerprint",
    "fingerprint_git_tree",
    "fingerprint_materialized_git_tree",
    "fingerprint_tree",
    "git_environment",
    "git_output",
    "iter_bounded_tree",
    "preflight_materialized_tree",
    "reject_symlink_parents",
]
