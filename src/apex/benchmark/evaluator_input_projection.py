"""Run-scoped, reverified inputs for the Docker evaluator sidecar."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import ConfigurationError, canonical_json_bytes, sha256_json
from apex.runtime import LmEvalRuntimeReceipt
from apex.runtime.lm_eval_runtime import canonical_json, collect_runtime_files

from .evaluator_dataset import EvaluatorDatasetReceipt
from .evaluator_dataset_materialization import (
    EvaluatorDatasetMaterializationInput,
    materialize_evaluator_dataset_cas,
)


PROJECTION_SCHEMA = "apex.evaluator-sidecar-input-projection/v1"
_MANIFEST = "lm_eval_runtime_manifest.json"


@dataclass(frozen=True, slots=True)
class EvaluatorSidecarInputProjection:
    """Exact run-local mounts and the receipt that binds their identities."""

    root: Path
    dataset_mount: Path
    runtime_mount: Path
    launcher_path: Path
    receipt_path: Path
    receipt_sha256: str


def materialize_evaluator_sidecar_inputs(
    authority_root: Path,
    *,
    dataset_root: Path,
    dataset_receipt: EvaluatorDatasetReceipt,
    runtime_receipt: LmEvalRuntimeReceipt,
    launcher_source: Path,
    launcher_sha256: str,
) -> EvaluatorSidecarInputProjection:
    """Copy trusted inputs onto the run filesystem and reverify every byte."""

    root = authority_root / "sidecar-inputs"
    if root.exists() or root.is_symlink():
        raise _invalid("Evaluator sidecar input projection already exists")
    root.mkdir(mode=0o700)
    try:
        dataset = _project_dataset(root, dataset_root, dataset_receipt)
        runtime = _project_runtime(root, runtime_receipt)
        launcher = _project_launcher(root, launcher_source, launcher_sha256)
        receipt = _receipt(
            authority_root,
            dataset,
            runtime,
            launcher,
            launcher_sha256,
            dataset_receipt,
            runtime_receipt,
        )
        receipt_path = _write_new(root / "projection_receipt.json", receipt)
        root.chmod(0o500)
    except Exception:
        _remove_owned_tree(root)
        raise
    return EvaluatorSidecarInputProjection(
        root=root.resolve(strict=True),
        dataset_mount=(dataset / "files").resolve(strict=True),
        runtime_mount=runtime.resolve(strict=True),
        launcher_path=launcher.resolve(strict=True),
        receipt_path=receipt_path,
        receipt_sha256=str(receipt["projection_sha256"]),
    )


def _project_dataset(
    root: Path,
    source: Path,
    expected: EvaluatorDatasetReceipt,
) -> Path:
    destination = root / "dataset"
    files = tuple(
        EvaluatorDatasetMaterializationInput(
            item.split,
            item.artifact.path,
            item.artifact.size_bytes,
            item.artifact.sha256,
            source / "files" / PurePosixPath(item.artifact.path),
        )
        for item in expected.files
    )
    observed = materialize_evaluator_dataset_cas(
        destination,
        repository=expected.repository,
        dataset_path=expected.path,
        dataset_name=expected.name,
        revision=expected.revision,
        files=files,
    )
    if observed != expected:
        raise _invalid("Projected evaluator dataset differs from its receipt")
    return destination


def _project_runtime(root: Path, expected: LmEvalRuntimeReceipt) -> Path:
    destination = root / "runtime"
    source = expected.root.resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise _invalid("Evaluator runtime source is unsafe")
    try:
        shutil.copytree(source, destination, symlinks=True)
    except OSError as error:
        raise _invalid("Cannot project evaluator runtime") from error
    _verify_runtime_projection(destination, expected)
    return destination


def _project_launcher(root: Path, source: Path, expected_sha256: str) -> Path:
    destination = root / "evaluator_sidecar_entry.py"
    source_fd = -1
    target_fd = -1
    try:
        source_fd = os.open(
            source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        before = os.fstat(source_fd)
        target_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o400,
        )
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise _invalid("Cannot project evaluator launcher")
                view = view[written:]
        os.fsync(target_fd)
        after = os.fstat(source_fd)
    except OSError as error:
        raise _invalid("Cannot project evaluator launcher") from error
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        if source_fd >= 0:
            os.close(source_fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or size != before.st_size
        or digest.hexdigest() != expected_sha256
    ):
        raise _invalid("Projected evaluator launcher differs from its contract")
    return destination


def _verify_runtime_projection(
    root: Path, expected: LmEvalRuntimeReceipt
) -> None:
    try:
        observed = root.lstat()
        children = {item.name for item in root.iterdir()}
        manifest_path = root / _MANIFEST
        manifest_stat = manifest_path.lstat()
        manifest_content = manifest_path.read_bytes()
        manifest = json.loads(manifest_content)
        files = collect_runtime_files(root / "site-packages")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _invalid("Projected evaluator runtime is unreadable") from error
    digest = hashlib.sha256(
        canonical_json({"identity": dict(expected.identity), "files": files})
    ).hexdigest()
    valid = (
        stat.S_ISDIR(observed.st_mode)
        and not root.is_symlink()
        and stat.S_IMODE(observed.st_mode) & 0o222 == 0
        and children == {_MANIFEST, "site-packages"}
        and stat.S_ISREG(manifest_stat.st_mode)
        and not manifest_path.is_symlink()
        and manifest_stat.st_nlink == 1
        and stat.S_IMODE(manifest_stat.st_mode) & 0o222 == 0
        and hashlib.sha256(manifest_content).hexdigest() == expected.manifest_sha256
        and isinstance(manifest, Mapping)
        and manifest.get("runtime_sha256") == expected.runtime_sha256
        and manifest.get("identity") == dict(expected.identity)
        and manifest.get("files") == files
        and len(files) == expected.file_count
        and digest == expected.runtime_sha256
    )
    if not valid:
        raise _invalid("Projected evaluator runtime differs from its receipt")


def _receipt(
    authority_root: Path,
    dataset: Path,
    runtime: Path,
    launcher: Path,
    launcher_sha256: str,
    dataset_receipt: EvaluatorDatasetReceipt,
    runtime_receipt: LmEvalRuntimeReceipt,
) -> dict[str, object]:
    payload = {
        "schema": PROJECTION_SCHEMA,
        "dataset": {
            "mount": dataset.relative_to(authority_root).as_posix() + "/files",
            "receipt_sha256": dataset_receipt.sha256,
            "revision": dataset_receipt.revision,
        },
        "runtime": {
            "mount": runtime.relative_to(authority_root).as_posix(),
            "runtime_sha256": runtime_receipt.runtime_sha256,
            "manifest_sha256": runtime_receipt.manifest_sha256,
            "lock_sha256": runtime_receipt.lock_sha256,
            "file_count": runtime_receipt.file_count,
        },
        "launcher": {
            "path": launcher.relative_to(authority_root).as_posix(),
            "sha256": launcher_sha256,
        },
        "verified": True,
    }
    return {**payload, "projection_sha256": sha256_json(payload)}


def _write_new(path: Path, value: Mapping[str, Any]) -> Path:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
    )
    try:
        payload = canonical_json_bytes(value) + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise _invalid("Cannot write evaluator input projection receipt")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path.resolve(strict=True)


def _remove_owned_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, names in os.walk(root, topdown=False):
        selected = Path(current)
        selected.chmod(0o700)
        for name in names:
            child = selected / name
            if not child.is_symlink():
                child.chmod(0o600)
            child.unlink()
        for name in directories:
            child = selected / name
            if child.is_symlink():
                child.unlink()
            else:
                child.chmod(0o700)
                child.rmdir()
    root.rmdir()


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_input_projection_invalid")


__all__ = [
    "EvaluatorSidecarInputProjection",
    "PROJECTION_SCHEMA",
    "materialize_evaluator_sidecar_inputs",
]
