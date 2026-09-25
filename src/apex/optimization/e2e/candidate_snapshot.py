"""Trusted source identity and immutable-byte snapshots for E2E candidates."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol

from apex.core import IntegrityError, sha256_bytes, sha256_json

from .candidate_fingerprint import (
    MAX_FROZEN_SOURCE_BYTES,
    SourceFingerprint,
    reject_symlink_parents,
)


@dataclass(frozen=True, slots=True)
class FrozenCandidateSource:
    """Evaluator-owned source bytes captured after containment teardown."""

    relative_path: str
    sha256: str
    mode: int
    content: bytes = field(repr=False)


class CandidateWorkspaceShape(Protocol):
    """Narrow workspace view needed to capture trusted source bytes."""

    root: Path
    editable_files: tuple[str, ...]


class FrozenCandidateShape(Protocol):
    """Narrow candidate view consumed by evaluator snapshot operations."""

    succeeded: bool
    editable_files: tuple[str, ...]
    candidate_source_sha256: str | None
    frozen_sources: tuple[FrozenCandidateSource, ...]


def source_set_digest(
    values: Mapping[str, SourceFingerprint], editable_files: tuple[str, ...]
) -> str:
    """Return the canonical identity of the declared editable source set."""

    try:
        payload = [
            {"path": path, "sha256": values[path].sha256, "mode": values[path].mode}
            for path in editable_files
        ]
    except KeyError as error:
        raise IntegrityError("Declared kernel source is missing", "missing_candidate_source") from error
    return sha256_json({"schema_version": 1, "files": payload})


def capture_frozen_sources(
    workspace: CandidateWorkspaceShape,
    *,
    expected_source_sha256: str,
) -> tuple[FrozenCandidateSource, ...]:
    """Capture editable source bytes without following a workspace link."""

    captured: list[FrozenCandidateSource] = []
    fingerprints: dict[str, SourceFingerprint] = {}
    root_fd = os.open(
        workspace.root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        for relative in workspace.editable_files:
            reject_symlink_parents(workspace.root, relative)
            source, fingerprint = _capture_source(root_fd, relative)
            captured.append(source)
            fingerprints[relative] = fingerprint
    except OSError as error:
        raise IntegrityError(
            "Candidate source could not be captured without following links",
            "candidate_source_capture_unsafe",
        ) from error
    finally:
        os.close(root_fd)
    if source_set_digest(fingerprints, workspace.editable_files) != expected_source_sha256:
        raise IntegrityError(
            "Candidate source changed after freeze",
            "candidate_source_capture_drift",
        )
    return tuple(captured)


def _capture_source(
    root_fd: int, relative: str
) -> tuple[FrozenCandidateSource, SourceFingerprint]:
    descriptor = os.open(
        relative,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        _validate_capture_metadata(before)
        content = _read_all(descriptor, MAX_FROZEN_SOURCE_BYTES)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _capture_identity(before) != _capture_identity(after):
        raise IntegrityError(
            "Candidate source changed while it was captured",
            "candidate_source_capture_drift",
        )
    digest = sha256_bytes(content)
    mode = after.st_mode & 0o777
    return (
        FrozenCandidateSource(relative, digest, mode, content),
        SourceFingerprint(digest, mode),
    )


def _validate_capture_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise IntegrityError(
            "Frozen candidate source is not a unique regular file",
            "candidate_source_capture_unsafe",
        )
    if metadata.st_size > MAX_FROZEN_SOURCE_BYTES:
        raise IntegrityError(
            "Candidate source exceeds the frozen-source byte limit",
            "candidate_source_too_large",
            {
                "limit_bytes": MAX_FROZEN_SOURCE_BYTES,
                "observed_bytes": metadata.st_size,
            },
        )


def _read_all(descriptor: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise IntegrityError(
                "Candidate source exceeds the frozen-source byte limit",
                "candidate_source_too_large",
                {"limit_bytes": limit, "observed_bytes": total},
            )
        chunks.append(chunk)


def _capture_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def frozen_candidate_source(
    candidate: FrozenCandidateShape, relative: str
) -> FrozenCandidateSource:
    """Return one source from the evaluator-owned immutable byte snapshot."""

    validate_frozen_sources(candidate)
    matches = tuple(
        source for source in candidate.frozen_sources if source.relative_path == relative
    )
    if len(matches) != 1:
        raise IntegrityError(
            "Frozen candidate source mapping is incomplete",
            "candidate_source_mapping_mismatch",
        )
    return matches[0]


def validate_frozen_sources(candidate: FrozenCandidateShape) -> None:
    """Bind captured source bytes to the candidate's declared source-set digest."""

    paths = tuple(source.relative_path for source in candidate.frozen_sources)
    if (
        not candidate.succeeded
        or candidate.candidate_source_sha256 is None
        or paths != candidate.editable_files
        or len(paths) != len(set(paths))
    ):
        raise IntegrityError(
            "Candidate lacks one complete frozen source snapshot",
            "invalid_frozen_candidate",
        )
    fingerprints = _validated_frozen_fingerprints(candidate.frozen_sources)
    if source_set_digest(fingerprints, candidate.editable_files) != (
        candidate.candidate_source_sha256
    ):
        raise IntegrityError(
            "Frozen candidate source-set digest drifted",
            "candidate_source_capture_drift",
        )


def _validated_frozen_fingerprints(
    sources: tuple[FrozenCandidateSource, ...],
) -> dict[str, SourceFingerprint]:
    fingerprints: dict[str, SourceFingerprint] = {}
    for source in sources:
        if not isinstance(source.relative_path, str) or type(source.mode) is not int:
            raise IntegrityError(
                "Frozen candidate source metadata is unsafe",
                "candidate_source_mapping_mismatch",
            )
        path = PurePosixPath(source.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.as_posix() != source.relative_path
            or not 0 <= source.mode <= 0o777
            or type(source.content) is not bytes
        ):
            raise IntegrityError(
                "Frozen candidate source path is unsafe",
                "candidate_source_mapping_mismatch",
            )
        if len(source.content) > MAX_FROZEN_SOURCE_BYTES:
            raise IntegrityError(
                "Candidate source exceeds the frozen-source byte limit",
                "candidate_source_too_large",
                {
                    "path": source.relative_path,
                    "limit_bytes": MAX_FROZEN_SOURCE_BYTES,
                    "observed_bytes": len(source.content),
                },
            )
        digest = sha256_bytes(source.content)
        if digest != source.sha256:
            raise IntegrityError(
                "Frozen candidate source bytes drifted",
                "candidate_source_capture_drift",
            )
        fingerprints[source.relative_path] = SourceFingerprint(digest, source.mode)
    return fingerprints


def materialize_frozen_sources(
    candidate: FrozenCandidateShape, destination: Path
) -> Path:
    """Create a read-only evaluator snapshot without reopening agent paths."""

    validate_frozen_sources(candidate)
    _create_snapshot_destination(destination)
    directories = {destination}
    try:
        for source in candidate.frozen_sources:
            path = _write_frozen_source(destination, source)
            directories.update((path.parent, *path.parent.parents))
        _seal_snapshot_directories(destination, directories)
    except IntegrityError:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    except OSError as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise IntegrityError(
            "Frozen candidate snapshot could not be materialized",
            "candidate_snapshot_materialization_failed",
            {"error_type": type(error).__name__},
        ) from error
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination


def _create_snapshot_destination(destination: Path) -> None:
    if not destination.is_absolute() or destination.exists():
        _unsafe_snapshot_destination()
    parent = destination.parent
    try:
        metadata = parent.lstat()
        canonical = parent.resolve(strict=True)
    except OSError as error:
        raise IntegrityError(
            "Frozen candidate destination is unsafe",
            "candidate_snapshot_destination_unsafe",
        ) from error
    if (
        parent != canonical
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o022
    ):
        _unsafe_snapshot_destination()
    descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    created = False
    try:
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) != (metadata.st_dev, metadata.st_ino):
            _unsafe_snapshot_destination()
        os.mkdir(destination.name, mode=0o700, dir_fd=descriptor)
        created = True
        child = os.open(
            destination.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            child_metadata = os.fstat(child)
            if (
                not stat.S_ISDIR(child_metadata.st_mode)
                or child_metadata.st_uid != os.geteuid()
                or child_metadata.st_mode & 0o077
            ):
                _unsafe_snapshot_destination()
        finally:
            os.close(child)
    except Exception as error:
        if created:
            try:
                os.rmdir(destination.name, dir_fd=descriptor)
            except OSError:
                pass
        if isinstance(error, IntegrityError):
            raise
        raise IntegrityError(
            "Frozen candidate destination is unsafe",
            "candidate_snapshot_destination_unsafe",
        ) from error
    finally:
        os.close(descriptor)


def _unsafe_snapshot_destination() -> None:
    raise IntegrityError(
        "Frozen candidate destination is unsafe",
        "candidate_snapshot_destination_unsafe",
    )


def _write_frozen_source(destination: Path, source: FrozenCandidateSource) -> Path:
    relative = PurePosixPath(source.relative_path)
    path = destination.joinpath(*relative.parts)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        _write_all(descriptor, source.content)
        os.fsync(descriptor)
        os.fchmod(descriptor, source.mode & ~0o222)
    finally:
        os.close(descriptor)
    return path


def _seal_snapshot_directories(destination: Path, directories: set[Path]) -> None:
    selected = (
        item
        for item in directories
        if item == destination or item.is_relative_to(destination)
    )
    for directory in sorted(selected, key=lambda item: len(item.parts), reverse=True):
        directory.chmod(0o555)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short candidate snapshot write")
        view = view[written:]


__all__ = [
    "FrozenCandidateSource",
    "capture_frozen_sources",
    "frozen_candidate_source",
    "materialize_frozen_sources",
    "source_set_digest",
    "validate_frozen_sources",
]
