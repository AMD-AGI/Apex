"""Deterministic publication of an exact offline evaluator dataset CAS."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from apex.core import ConfigurationError, canonical_json_bytes

from .evaluator_artifact_receipt import EvaluatorArtifactReceipt
from .evaluator_dataset import EvaluatorDatasetFile, EvaluatorDatasetReceipt
from .evaluator_dataset_cas import RECEIPT_NAME, verify_evaluator_dataset_root


MAX_DATASET_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class EvaluatorDatasetMaterializationInput:
    """One downloaded file whose expected identity came from the policy lock."""

    split: str
    path: str
    size_bytes: int
    sha256: str
    source: Path

    def __post_init__(self) -> None:
        try:
            EvaluatorDatasetFile(
                self.split,
                EvaluatorArtifactReceipt(self.path, self.size_bytes, self.sha256),
            )
        except ValueError as error:
            raise ValueError("Evaluator dataset input is invalid") from error

    @property
    def receipt(self) -> EvaluatorDatasetFile:
        return EvaluatorDatasetFile(
            self.split,
            EvaluatorArtifactReceipt(self.path, self.size_bytes, self.sha256),
        )


def materialize_evaluator_dataset_cas(
    destination: Path,
    *,
    repository: str,
    dataset_path: str,
    dataset_name: str,
    revision: str,
    files: tuple[EvaluatorDatasetMaterializationInput, ...],
) -> EvaluatorDatasetReceipt:
    """Copy verified files into a sealed CAS and atomically publish it."""

    _validate_inputs(destination, files)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        files_root = stage / "files"
        files_root.mkdir(mode=0o700)
        for item in files:
            _copy_verified(item, files_root)
        receipt = EvaluatorDatasetReceipt(
            repository=repository,
            path=dataset_path,
            name=dataset_name,
            revision=revision,
            files=tuple(item.receipt for item in files),
        )
        _write_receipt(stage / RECEIPT_NAME, receipt)
        _seal(stage)
        os.replace(stage, destination)
    except Exception:
        _remove_stage(stage)
        raise
    observed = verify_evaluator_dataset_root(
        destination,
        expected_repository=repository,
        expected_path=dataset_path,
        expected_name=dataset_name,
        expected_revision=revision,
        expected_splits=tuple(sorted({item.split for item in files})),
    )
    if observed != receipt:
        raise _invalid("Published evaluator dataset receipt differs")
    return observed


def verify_dataset_receipt_against_inputs(
    receipt: EvaluatorDatasetReceipt,
    files: tuple[EvaluatorDatasetMaterializationInput, ...],
) -> None:
    """Require an existing CAS to contain exactly the locked input set."""

    expected = tuple(item.receipt for item in files)
    if receipt.files != expected:
        raise _invalid("Evaluator dataset CAS differs from its file lock")


def _validate_inputs(
    destination: Path, files: tuple[EvaluatorDatasetMaterializationInput, ...]
) -> None:
    ordered = tuple(sorted(files, key=lambda item: (item.split, item.path)))
    if (
        destination.exists()
        or destination.is_symlink()
        or not files
        or files != ordered
        or len({item.path for item in files}) != len(files)
        or sum(item.size_bytes for item in files) > MAX_DATASET_BYTES
    ):
        raise _invalid("Evaluator dataset materialization inputs are invalid")


def _copy_verified(
    item: EvaluatorDatasetMaterializationInput, files_root: Path
) -> None:
    target = files_root.joinpath(*PurePosixPath(item.path).parts)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_fd = _open_source(item.source)
    target_fd = -1
    try:
        before = os.fstat(source_fd)
        target_fd = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        digest, size = _copy_bytes(source_fd, target_fd, item.size_bytes)
        os.fsync(target_fd)
        after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
        if target_fd >= 0:
            os.close(target_fd)
    if (
        _stat_identity(before) != _stat_identity(after)
        or size != item.size_bytes
        or digest != item.sha256
    ):
        raise _invalid("Downloaded evaluator dataset file differs from its lock")


def _open_source(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        observed = os.fstat(descriptor)
    except OSError as error:
        raise _invalid("Cannot open downloaded evaluator dataset file") from error
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        os.close(descriptor)
        raise _invalid("Downloaded evaluator dataset file is unsafe")
    return descriptor


def _copy_bytes(source: int, target: int, expected: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while size <= expected:
        chunk = os.read(source, min(1024 * 1024, expected + 1 - size))
        if not chunk:
            break
        _write_all(target, chunk)
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise _invalid("Cannot write evaluator dataset CAS")
        view = view[written:]


def _write_receipt(path: Path, receipt: EvaluatorDatasetReceipt) -> None:
    payload = canonical_json_bytes(receipt.to_dict()) + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal(root: Path) -> None:
    for current, directories, names in os.walk(root, topdown=False):
        selected = Path(current)
        for name in names:
            (selected / name).chmod(0o400)
        for name in directories:
            (selected / name).chmod(0o500)
        selected.chmod(0o500)


def _remove_stage(stage: Path) -> None:
    if not stage.exists():
        return
    for current, directories, names in os.walk(stage, topdown=False):
        selected = Path(current)
        for name in names:
            (selected / name).chmod(0o600)
        for name in directories:
            (selected / name).chmod(0o700)
        selected.chmod(0o700)
    shutil.rmtree(stage)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_dataset_materialization_invalid")


__all__ = [
    "EvaluatorDatasetMaterializationInput",
    "MAX_DATASET_BYTES",
    "materialize_evaluator_dataset_cas",
    "verify_dataset_receipt_against_inputs",
]
