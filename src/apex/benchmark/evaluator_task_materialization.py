"""Deterministic private task projection with a locked dataset revision."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from apex.core import ConfigurationError, sha256_bytes, sha256_file, sha256_json


SCHEMA = "apex.evaluator-task-materialization/v1"
METHOD = "apex_yaml_dataset_revision_projection_v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EvaluatorTaskMaterializationReceipt:
    """Source/effective task identity for one private evaluator projection."""

    source_commit: str
    source_tree: str
    source_path: str
    source_sha256: str
    effective_path: str
    effective_sha256: str
    dataset_revision: str
    dataset_receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            not _COMMIT.fullmatch(self.source_commit)
            or not _COMMIT.fullmatch(self.source_tree)
            or not _safe_relative(self.source_path)
            or not _safe_relative(self.effective_path)
            or not _DIGEST.fullmatch(self.source_sha256)
            or not _DIGEST.fullmatch(self.effective_sha256)
            or not _COMMIT.fullmatch(self.dataset_revision)
            or not _DIGEST.fullmatch(self.dataset_receipt_sha256)
        ):
            raise ValueError("Evaluator task materialization receipt is invalid")

    @property
    def materializer_sha256(self) -> str:
        return sha256_json({"schema": SCHEMA, "method": METHOD})

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "method": METHOD,
            "materializer_sha256": self.materializer_sha256,
            "source": {
                "commit": self.source_commit,
                "tree": self.source_tree,
                "path": self.source_path,
                "sha256": self.source_sha256,
            },
            "effective": {
                "path": self.effective_path,
                "sha256": self.effective_sha256,
            },
            "dataset_revision": self.dataset_revision,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.sha256}


def materialize_evaluator_task(
    source_root: Path,
    destination_root: Path,
    *,
    source_commit: str,
    source_tree: str,
    definition_path: str,
    definition_sha256: str,
    dataset_path: str,
    dataset_name: str,
    dataset_revision: str,
    dataset_receipt_sha256: str,
    dataset_files: Mapping[str, tuple[str, ...]],
) -> EvaluatorTaskMaterializationReceipt:
    """Write one immutable private YAML; never mutate the dependency checkout."""

    if not _DIGEST.fullmatch(dataset_receipt_sha256):
        raise _invalid("Evaluator dataset receipt digest is invalid")
    source = _source_file(source_root, definition_path, definition_sha256)
    document = _load_task(source)
    _validate_source_task(document, dataset_path, dataset_name)
    effective = _effective_task(document, dataset_revision, dataset_files)
    payload = yaml.safe_dump(
        effective,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    ).encode("utf-8")
    output = _create_output(destination_root, payload)
    _validate_effective_task(output, dataset_revision, dataset_files)
    return EvaluatorTaskMaterializationReceipt(
        source_commit=source_commit,
        source_tree=source_tree,
        source_path=definition_path,
        source_sha256=definition_sha256,
        effective_path="task/gsm8k.yaml",
        effective_sha256=sha256_bytes(payload),
        dataset_revision=dataset_revision,
        dataset_receipt_sha256=dataset_receipt_sha256,
    )


def _source_file(root: Path, relative: str, expected_sha256: str) -> Path:
    if not _safe_relative(relative) or not _DIGEST.fullmatch(expected_sha256):
        raise _invalid("Evaluator source task identity is invalid")
    base = root.resolve(strict=True)
    selected = base.joinpath(*PurePosixPath(relative).parts)
    try:
        observed = selected.lstat()
        resolved = selected.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as error:
        raise _invalid("Evaluator source task is unsafe") from error
    if (
        selected.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or sha256_file(resolved) != expected_sha256
    ):
        raise _invalid("Evaluator source task differs from its lock")
    return resolved


def _load_task(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise _invalid("Evaluator task YAML is invalid") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _invalid("Evaluator task YAML must be a string-keyed mapping")
    return value


def _validate_source_task(
    value: Mapping[str, Any], dataset_path: str, dataset_name: str
) -> None:
    if (
        value.get("task") != "gsm8k"
        or value.get("dataset_path") != dataset_path
        or value.get("dataset_name") != dataset_name
    ):
        raise _invalid("Evaluator task conflicts with its policy lock")


def _effective_task(
    value: Mapping[str, Any],
    revision: str,
    dataset_files: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    if not _COMMIT.fullmatch(revision) or not _valid_dataset_files(dataset_files):
        raise _invalid("Evaluator dataset revision is invalid")
    effective = dict(value)
    raw_kwargs = effective.get("dataset_kwargs")
    if raw_kwargs is not None and not isinstance(raw_kwargs, Mapping):
        raise _invalid("Evaluator dataset kwargs are invalid")
    kwargs = dict(raw_kwargs or {})
    existing = kwargs.get("revision")
    if existing is not None and existing != revision:
        raise _invalid("Evaluator task dataset revision conflicts with its lock")
    kwargs["revision"] = revision
    kwargs["data_files"] = {
        split: list(paths) for split, paths in sorted(dataset_files.items())
    }
    effective["dataset_kwargs"] = kwargs
    effective["dataset_path"] = "parquet"
    effective.pop("dataset_name", None)
    return effective


def _create_output(root: Path, payload: bytes) -> Path:
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        task_root = root / "task"
        task_root.mkdir(mode=0o700)
        output = task_root / "gsm8k.yaml"
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise _invalid("Cannot materialize evaluator task") from error
    return output


def _validate_effective_task(
    output: Path,
    revision: str,
    dataset_files: Mapping[str, tuple[str, ...]],
) -> None:
    value = _load_task(output)
    kwargs = value.get("dataset_kwargs")
    expected_files = {
        split: list(paths) for split, paths in sorted(dataset_files.items())
    }
    if (
        value.get("task") != "gsm8k"
        or value.get("dataset_path") != "parquet"
        or "dataset_name" in value
        or not isinstance(kwargs, Mapping)
        or kwargs.get("revision") != revision
        or kwargs.get("data_files") != expected_files
    ):
        raise _invalid("Materialized evaluator task omitted its dataset revision")


def _valid_dataset_files(value: Mapping[str, tuple[str, ...]]) -> bool:
    if not value or tuple(value) != tuple(sorted(value)):
        return False
    for split, paths in value.items():
        if not split or not paths:
            return False
        for path in paths:
            if (
                not isinstance(path, str)
                or not path.startswith("/evaluator/dataset/")
                or any(character in path for character in "\r\n\0")
                or "/../" in path
            ):
                return False
    return True


def _safe_relative(value: str) -> bool:
    pure = PurePosixPath(value)
    return bool(
        value
        and not pure.is_absolute()
        and pure.parts
        and all(part not in {"", ".", ".."} for part in pure.parts)
    )


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_task_materialization_invalid")


__all__ = [
    "EvaluatorTaskMaterializationReceipt",
    "METHOD",
    "SCHEMA",
    "materialize_evaluator_task",
]
