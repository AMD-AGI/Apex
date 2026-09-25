"""Strict lock for the formal lm-eval task and dataset identity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from apex.core import sha256_file

from .repositories import BootstrapError


SCHEMA = "apex.evaluator-policy-lock/v2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_TOP_KEYS = frozenset(
    {"schema", "policy_id", "primary_metric", "sample_logging_required", "task", "dataset"}
)
_TASK_KEYS = frozenset(
    {"name", "definition_dependency", "definition_path", "definition_sha256"}
)
_DATASET_KEYS = frozenset(
    {"repository", "path", "name", "revision", "splits", "files"}
)
_DATASET_FILE_KEYS = frozenset({"split", "path", "size_bytes", "sha256"})


@dataclass(frozen=True, slots=True)
class EvaluatorDatasetLockFile:
    """One exact file selected from the immutable dataset revision."""

    split: str
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if (
            not self.split
            or pure.is_absolute()
            or not pure.parts
            or any(part in {"", ".", ".."} for part in pure.parts)
            or isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or not _SHA256.fullmatch(self.sha256)
        ):
            raise ValueError("evaluator dataset lock file is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class EvaluatorPolicyLock:
    """Content-bound task definition and immutable dataset revision."""

    path: Path
    lock_sha256: str
    policy_id: str
    primary_metric: str
    sample_logging_required: bool
    task_name: str
    task_definition_path: str
    task_definition_sha256: str
    dataset_repository: str
    dataset_path: str
    dataset_name: str
    dataset_revision: str
    dataset_splits: tuple[str, ...]
    dataset_files: tuple[EvaluatorDatasetLockFile, ...]

    def env(self) -> dict[str, str]:
        return {
            "MAGPIE_EVAL_POLICY_ID": self.policy_id,
            "MAGPIE_EVAL_TASKS": self.task_name,
            "MAGPIE_EVAL_PRIMARY_METRIC": self.primary_metric,
            "MAGPIE_EVAL_TASK_DEFINITION_PATH": self.task_definition_path,
            "MAGPIE_EVAL_TASK_DEFINITION_SHA256": self.task_definition_sha256,
            "MAGPIE_EVAL_DATASET_PATH": self.dataset_path,
            "MAGPIE_EVAL_DATASET_NAME": self.dataset_name,
            "MAGPIE_EVAL_DATASET_REVISION": self.dataset_revision,
            "EVAL_TASKS_DIR": self.task_definition_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "path": str(self.path),
            "lock_sha256": self.lock_sha256,
            "policy_id": self.policy_id,
            "primary_metric": self.primary_metric,
            "sample_logging_required": self.sample_logging_required,
            "task": {
                "name": self.task_name,
                "definition_path": self.task_definition_path,
                "definition_sha256": self.task_definition_sha256,
            },
            "dataset": {
                "repository": self.dataset_repository,
                "path": self.dataset_path,
                "name": self.dataset_name,
                "revision": self.dataset_revision,
                "splits": list(self.dataset_splits),
                "files": [item.to_dict() for item in self.dataset_files],
            },
        }


def load_evaluator_policy_lock(
    path: Path, *, inferencex_root: Path | None = None
) -> EvaluatorPolicyLock:
    """Load the lock and rehash its exact task definition from InferenceX."""

    selected = path.resolve(strict=True)
    if path.is_symlink() or not selected.is_file():
        raise BootstrapError("evaluator policy lock is not a regular file")
    try:
        raw = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid evaluator policy lock: {error}") from error
    top = _mapping(raw, _TOP_KEYS, "evaluator policy lock")
    if top["schema"] != SCHEMA:
        raise BootstrapError("unsupported evaluator policy lock schema")
    task = _mapping(top["task"], _TASK_KEYS, "evaluator task")
    dataset = _mapping(top["dataset"], _DATASET_KEYS, "evaluator dataset")
    _validate_values(top, task, dataset)
    if inferencex_root is not None:
        task_path = _task_path(inferencex_root, str(task["definition_path"]))
        if sha256_file(task_path) != task["definition_sha256"]:
            raise BootstrapError("evaluator task definition digest differs")
        _validate_task_yaml(task_path, task, dataset)
    files = tuple(_dataset_file(value) for value in dataset["files"])
    return EvaluatorPolicyLock(
        selected,
        sha256_file(selected),
        str(top["policy_id"]),
        str(top["primary_metric"]),
        bool(top["sample_logging_required"]),
        str(task["name"]),
        str(task["definition_path"]),
        str(task["definition_sha256"]),
        str(dataset["repository"]),
        str(dataset["path"]),
        str(dataset["name"]),
        str(dataset["revision"]),
        tuple(dataset["splits"]),
        files,
    )


def _mapping(value: object, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise BootstrapError(f"{label} fields differ")
    return value  # type: ignore[return-value]


def _validate_values(
    top: Mapping[str, Any], task: Mapping[str, Any], dataset: Mapping[str, Any]
) -> None:
    texts = (
        top["policy_id"], top["primary_metric"], task["name"],
        task["definition_dependency"], dataset["repository"], dataset["path"],
        dataset["name"],
    )
    splits = dataset["splits"]
    raw_files = dataset["files"]
    try:
        files = tuple(_dataset_file(value) for value in raw_files)
    except (TypeError, ValueError) as error:
        raise BootstrapError("evaluator dataset files are invalid") from error
    if (
        any(not isinstance(value, str) or not value for value in texts)
        or top["sample_logging_required"] is not True
        or task["definition_dependency"] != "inferencex"
        or not _SHA256.fullmatch(str(task["definition_sha256"]))
        or not _GIT.fullmatch(str(dataset["revision"]))
        or not isinstance(splits, list)
        or splits != sorted(set(splits))
        or any(not isinstance(item, str) or not item for item in splits)
        or not isinstance(raw_files, list)
        or not files
        or files != tuple(sorted(files, key=lambda item: (item.split, item.path)))
        or len({item.path for item in files}) != len(files)
        or tuple(sorted({item.split for item in files})) != tuple(splits)
    ):
        raise BootstrapError("evaluator policy lock values are invalid")


def _dataset_file(value: object) -> EvaluatorDatasetLockFile:
    item = _mapping(value, _DATASET_FILE_KEYS, "evaluator dataset file")
    if (
        not isinstance(item["split"], str)
        or not isinstance(item["path"], str)
        or isinstance(item["size_bytes"], bool)
        or not isinstance(item["size_bytes"], int)
        or not isinstance(item["sha256"], str)
    ):
        raise ValueError("evaluator dataset lock file types are invalid")
    return EvaluatorDatasetLockFile(
        split=item["split"],
        path=item["path"],
        size_bytes=item["size_bytes"],
        sha256=item["sha256"],
    )


def _task_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise BootstrapError("evaluator task path is unsafe")
    base = root.resolve(strict=True)
    selected = base.joinpath(*pure.parts)
    resolved = selected.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise BootstrapError("evaluator task path escapes InferenceX") from error
    if selected.is_symlink() or not resolved.is_file():
        raise BootstrapError("evaluator task definition is not a regular file")
    return resolved


def _validate_task_yaml(
    path: Path, task: Mapping[str, Any], dataset: Mapping[str, Any]
) -> None:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise BootstrapError("evaluator task definition is invalid") from error
    if not isinstance(value, Mapping) or any(
        value.get(key) != expected
        for key, expected in (
            ("task", task["name"]),
            ("dataset_path", dataset["path"]),
            ("dataset_name", dataset["name"]),
        )
    ):
        raise BootstrapError("evaluator task definition conflicts with its lock")


__all__ = [
    "EvaluatorDatasetLockFile",
    "EvaluatorPolicyLock",
    "SCHEMA",
    "load_evaluator_policy_lock",
]
