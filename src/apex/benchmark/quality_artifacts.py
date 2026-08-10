"""Race-resistant loading of exact evaluator-declared quality artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import IntegrityError


MAX_ARTIFACT_FILES = 256
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_FIELDS = frozenset({"path", "size_bytes", "sha256"})


@dataclass(frozen=True, slots=True)
class LoadedQualityArtifact:
    """Bytes read from the same file descriptor used for receipt validation."""

    path: Path
    relative_path: str
    content: bytes
    sha256: str


def load_declared_quality_artifacts(
    root: Path,
    receipts: tuple[Mapping[str, Any], ...],
    *,
    max_files: int = MAX_ARTIFACT_FILES,
    max_total_bytes: int = MAX_ARTIFACT_BYTES,
    require_read_only: bool = False,
) -> tuple[LoadedQualityArtifact, ...]:
    """Load only declared files beneath one nonlinked authority root."""

    if (
        not 0 < max_files <= MAX_ARTIFACT_FILES
        or not 0 < max_total_bytes <= MAX_ARTIFACT_BYTES
        or not receipts
        or len(receipts) > max_files
    ):
        raise _unsafe("Quality artifact bounds are invalid")
    parsed = tuple(_receipt(value) for value in receipts)
    if len({value["path"] for value in parsed}) != len(parsed):
        raise _unsafe("Quality artifact locators are duplicated")
    root_fd = _open_root(root, require_read_only)
    loaded: list[LoadedQualityArtifact] = []
    total = 0
    try:
        for receipt in parsed:
            remaining = max_total_bytes - total
            item = _load_one(
                root, root_fd, receipt, remaining, require_read_only
            )
            total += len(item.content)
            loaded.append(item)
    finally:
        os.close(root_fd)
    return tuple(loaded)


def _receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise _unsafe("Quality artifact receipt fields are invalid")
    relative = value.get("path")
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    size = value.get("size_bytes")
    digest = value.get("sha256")
    if (
        pure is None
        or pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or set(digest) - set("0123456789abcdef")
    ):
        raise _unsafe("Quality artifact receipt values are invalid")
    return value


def _open_root(root: Path, require_read_only: bool) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(root, flags)
        observed = os.fstat(descriptor)
    except OSError as error:
        raise _unsafe("Quality artifact root is unsafe") from error
    if not stat.S_ISDIR(observed.st_mode) or (
        require_read_only and stat.S_IMODE(observed.st_mode) & 0o222
    ):
        os.close(descriptor)
        raise _unsafe("Quality artifact root is not a directory")
    return descriptor


def _load_one(
    root: Path,
    root_fd: int,
    receipt: Mapping[str, Any],
    remaining: int,
    require_read_only: bool,
) -> LoadedQualityArtifact:
    parts = PurePosixPath(str(receipt["path"])).parts
    parent_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = _open_directory(parent_fd, part, require_read_only)
            os.close(parent_fd)
            parent_fd = child
        descriptor = _open_file(parent_fd, parts[-1])
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(descriptor)
        expected_size = int(receipt["size_bytes"])
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
            or expected_size > remaining
            or require_read_only and stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise _unsafe("Quality artifact file identity is invalid")
        content = _read_bounded(descriptor, expected_size)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise _unsafe("Quality artifact changed while being read")
    digest = hashlib.sha256(content).hexdigest()
    if digest != receipt["sha256"]:
        raise IntegrityError(
            "Quality artifact differs from its authority receipt",
            "quality_artifact_receipt_mismatch",
        )
    return LoadedQualityArtifact(
        root.resolve() / Path(*parts), str(receipt["path"]), content, digest
    )


def _open_directory(parent_fd: int, name: str, require_read_only: bool) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise _unsafe("Quality artifact parent directory is unsafe") from error
    if require_read_only and stat.S_IMODE(os.fstat(descriptor).st_mode) & 0o222:
        os.close(descriptor)
        raise _unsafe("Quality artifact parent directory is writable")
    return descriptor


def _open_file(parent_fd: int, name: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise _unsafe("Quality artifact file is unsafe") from error


def _read_bounded(descriptor: int, expected_size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise _unsafe("Quality artifact ended before its declared size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise _unsafe("Quality artifact exceeds its declared size")
    return b"".join(chunks)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _unsafe(message: str) -> IntegrityError:
    return IntegrityError(message, "unsafe_quality_artifact")


__all__ = [
    "LoadedQualityArtifact",
    "MAX_ARTIFACT_BYTES",
    "MAX_ARTIFACT_FILES",
    "load_declared_quality_artifacts",
]
