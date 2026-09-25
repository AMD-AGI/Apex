"""Exclusive publication of sidecar outputs into the official Magpie workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from apex.core import ConfigurationError

from .evaluator_artifact_receipt import EvaluatorArtifactReceipt


MAX_FILES = 256
MAX_TOTAL_BYTES = 128 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PublishedEvaluatorOutputs:
    """Exact official locators produced from one sealed private output tree."""

    root: Path
    result_artifacts: tuple[EvaluatorArtifactReceipt, ...]
    sample_artifacts: tuple[EvaluatorArtifactReceipt, ...]


@dataclass(frozen=True, slots=True)
class _LoadedOutput:
    relative_path: str
    payload: bytes
    sha256: str
    role: str


def publish_evaluator_outputs(
    source_root: Path,
    workspace_root: Path,
    *,
    contract_sha256: str,
) -> PublishedEvaluatorOutputs:
    """Rehash a bounded exact set, copy exclusively, then seal both trees."""

    if _SHA256_PATTERN.fullmatch(contract_sha256) is None:
        raise _invalid("Evaluator contract digest is invalid")
    source = _safe_directory(source_root, writable=True)
    workspace = _safe_directory(workspace_root, writable=True)
    loaded = _load_outputs(source)
    results = tuple(item for item in loaded if item.role == "result")
    samples = tuple(item for item in loaded if item.role == "sample")
    if len(results) != 1 or not samples:
        raise _invalid("Evaluator output set is incomplete")
    publication_parent: Path | None = None
    parent_created = False
    destination_created = False
    try:
        publication_parent, parent_created = _publication_parent(workspace)
        destination = publication_parent / contract_sha256
        try:
            destination.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError as error:
            raise _invalid("Evaluator output publication already exists") from error
        except OSError as error:
            raise _invalid("Cannot create evaluator output publication") from error
        destination_created = True
        receipts = {
            item.relative_path: _publish_one(item, destination, workspace)
            for item in loaded
        }
        _seal(destination)
        _seal(source)
    except Exception:
        if destination_created:
            _remove_publication(destination)
        if parent_created and publication_parent is not None:
            _remove_empty_directory(publication_parent)
        raise
    return PublishedEvaluatorOutputs(
        destination.resolve(strict=True),
        tuple(receipts[item.relative_path] for item in results),
        tuple(receipts[item.relative_path] for item in samples),
    )


def _load_outputs(root: Path) -> tuple[_LoadedOutput, ...]:
    files = _discover_output_files(root)
    if not files or len(files) > MAX_FILES:
        raise _invalid("Evaluator output file count is invalid")
    loaded = tuple(_load_one(root, path) for path in files)
    if sum(len(item.payload) for item in loaded) > MAX_TOTAL_BYTES:
        raise _invalid("Evaluator output bytes exceed the authority bound")
    return loaded


def _load_one(root: Path, path: Path) -> _LoadedOutput:
    _validate_relative_directory_chain(root, path.parent)
    relative = path.relative_to(root).as_posix()
    role = _role(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_TOTAL_BYTES
        ):
            raise _invalid("Evaluator output file is unsafe")
        payload = _read_exact(descriptor, before.st_size)
        after = os.fstat(descriptor)
    except OSError as error:
        raise _invalid("Cannot read evaluator output file") from error
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    if _identity(before) != _identity(after):
        raise _invalid("Evaluator output changed while being read")
    _validate_relative_directory_chain(root, path.parent)
    if role == "result":
        _validate_result(payload)
    return _LoadedOutput(relative, payload, hashlib.sha256(payload).hexdigest(), role)


def _role(path: Path) -> str:
    if path.name.startswith("results") and path.suffix == ".json":
        return "result"
    if path.name.startswith("samples") and path.suffix == ".jsonl":
        return "sample"
    raise _invalid(f"Unexpected evaluator output: {path.name}")


def _validate_result(payload: bytes) -> None:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _invalid("Evaluator result is invalid JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("results"), dict):
        raise _invalid("Evaluator result lacks a results mapping")


def _publish_one(
    item: _LoadedOutput, destination: Path, workspace: Path
) -> EvaluatorArtifactReceipt:
    target = _publication_target(destination, item.relative_path)
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o400,
        )
    except OSError as error:
        raise _invalid("Cannot create evaluator output file") from error
    try:
        view = memoryview(item.payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise _invalid("Cannot publish evaluator output")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return EvaluatorArtifactReceipt(
        target.relative_to(workspace).as_posix(), len(item.payload), item.sha256
    )


def _safe_directory(path: Path, *, writable: bool) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        _validate_absolute_directory_chain(absolute)
        observed = absolute.lstat()
        selected = absolute.resolve(strict=True)
    except OSError as error:
        raise _invalid("Evaluator output directory is unavailable") from error
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (writable and stat.S_IMODE(observed.st_mode) & 0o300 != 0o300)
    ):
        raise _invalid("Evaluator output directory is unsafe")
    return selected


def _validate_absolute_directory_chain(path: Path) -> None:
    for candidate in reversed((path, *path.parents)):
        observed = candidate.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise _invalid("Evaluator output directory chain is unsafe")


def _validate_relative_directory_chain(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise _invalid("Evaluator output escaped its private root") from error
    current = root
    for part in relative.parts:
        current /= part
        try:
            observed = current.lstat()
        except OSError as error:
            raise _invalid("Evaluator output directory chain is unavailable") from error
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise _invalid("Evaluator output directory chain is unsafe")


def _discover_output_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        selected = Path(current)
        _validate_relative_directory_chain(root, selected)
        for name in directories:
            _validate_tree_entry(selected / name, directory=True)
        for name in names:
            candidate = selected / name
            _validate_tree_entry(candidate, directory=False)
            files.append(candidate)
            if len(files) > MAX_FILES:
                raise _invalid("Evaluator output file count is invalid")
    return tuple(sorted(files))


def _validate_tree_entry(path: Path, *, directory: bool) -> None:
    try:
        observed = path.lstat()
    except OSError as error:
        raise _invalid("Evaluator output entry is unavailable") from error
    expected = stat.S_ISDIR(observed.st_mode) if directory else stat.S_ISREG(observed.st_mode)
    if stat.S_ISLNK(observed.st_mode) or not expected:
        raise _invalid("Evaluator output tree contains an unsafe entry")


def _publication_parent(workspace: Path) -> tuple[Path, bool]:
    parent = workspace / "evaluator"
    try:
        observed = parent.lstat()
    except FileNotFoundError:
        try:
            parent.mkdir(mode=0o700, exist_ok=False)
        except OSError as error:
            raise _invalid("Cannot create evaluator publication directory") from error
        return parent, True
    except OSError as error:
        raise _invalid("Evaluator publication directory is unavailable") from error
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) & 0o300 != 0o300
    ):
        raise _invalid("Evaluator publication directory is unsafe")
    return parent, False


def _publication_target(destination: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise _invalid("Evaluator output relative path is unsafe")
    current = destination
    for part in relative.parts[:-1]:
        current /= part
        try:
            current.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            try:
                observed = current.lstat()
            except OSError as error:
                raise _invalid("Evaluator publication path is unavailable") from error
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise _invalid("Evaluator publication path is unsafe")
        except OSError as error:
            raise _invalid("Cannot create evaluator publication path") from error
    return current / relative.name


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise _invalid("Evaluator output ended before its declared size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise _invalid("Evaluator output exceeds its declared size")
    return b"".join(chunks)


def _seal(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        selected = Path(current)
        for name in files:
            child = selected / name
            _validate_tree_entry(child, directory=False)
            child.chmod(0o400)
        for name in directories:
            child = selected / name
            _validate_tree_entry(child, directory=True)
            child.chmod(0o500)
        selected.chmod(0o500)


def _remove_publication(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        path.unlink()
        return
    for current, directories, files in os.walk(path, topdown=False):
        selected = Path(current)
        for name in files:
            child = selected / name
            if child.is_symlink():
                child.unlink()
            else:
                child.chmod(0o600)
        for name in directories:
            child = selected / name
            if child.is_symlink():
                child.unlink()
            else:
                child.chmod(0o700)
        selected.chmod(0o700)
    shutil.rmtree(path)


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_output_publication_invalid")


__all__ = ["PublishedEvaluatorOutputs", "publish_evaluator_outputs"]
