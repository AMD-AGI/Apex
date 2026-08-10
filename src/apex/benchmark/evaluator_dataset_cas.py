"""Strict verification of an offline evaluator dataset CAS."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError

from .evaluator_dataset import EvaluatorDatasetReceipt
from .quality_artifacts import load_declared_quality_artifacts


RECEIPT_NAME = "evaluator_dataset_receipt.json"


def load_evaluator_dataset_receipt(path: Path) -> EvaluatorDatasetReceipt:
    """Load one immutable, nonlinked dataset receipt document."""

    try:
        observed = path.lstat()
        content = path.read_bytes()
    except OSError as error:
        raise _invalid("Cannot read evaluator dataset receipt") from error
    if (
        path.name != RECEIPT_NAME
        or path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) & 0o222
        or not 0 < len(content) <= 8 * 1024 * 1024
    ):
        raise _invalid("Evaluator dataset receipt file is unsafe")
    try:
        value = json.loads(content)
        if not isinstance(value, Mapping):
            raise ValueError("receipt must be an object")
        return EvaluatorDatasetReceipt.from_mapping(value)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise _invalid("Evaluator dataset receipt is invalid") from error


def verify_evaluator_dataset_cas(
    root: Path,
    receipt: EvaluatorDatasetReceipt,
    *,
    expected_repository: str,
    expected_path: str,
    expected_name: str,
    expected_revision: str,
    expected_splits: tuple[str, ...],
) -> EvaluatorDatasetReceipt:
    """Rehash every declared read-only file and bind it to the policy lock."""

    observed_splits = tuple(sorted({item.split for item in receipt.files}))
    expected = tuple(sorted(set(expected_splits)))
    matches = (
        receipt.repository == expected_repository
        and receipt.path == expected_path
        and receipt.name == expected_name
        and receipt.revision == expected_revision
        and observed_splits == expected
        and expected == expected_splits
    )
    if not matches:
        raise _invalid("Evaluator dataset CAS differs from its policy lock")
    artifacts = tuple(item.artifact.to_dict() for item in receipt.files)
    try:
        loaded = load_declared_quality_artifacts(
            root,
            artifacts,
            max_files=min(256, len(artifacts)),
            max_total_bytes=128 * 1024 * 1024,
            require_read_only=True,
        )
    except Exception as error:
        if isinstance(error, ConfigurationError):
            raise
        raise _invalid("Evaluator dataset CAS verification failed") from error
    if len(loaded) != len(receipt.files):
        raise _invalid("Evaluator dataset CAS file set is incomplete")
    return receipt


def verify_evaluator_dataset_root(
    root: Path,
    *,
    expected_repository: str,
    expected_path: str,
    expected_name: str,
    expected_revision: str,
    expected_splits: tuple[str, ...],
) -> EvaluatorDatasetReceipt:
    """Load and fully verify the conventional dataset CAS layout."""

    try:
        observed = root.lstat()
        selected = root.resolve(strict=True)
    except OSError as error:
        raise _invalid("Evaluator dataset CAS root is unsafe") from error
    if (
        root.is_symlink()
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) & 0o222
    ):
        raise _invalid("Evaluator dataset CAS root is unsafe")
    receipt = load_evaluator_dataset_receipt(selected / RECEIPT_NAME)
    return verify_evaluator_dataset_cas(
        selected / "files",
        receipt,
        expected_repository=expected_repository,
        expected_path=expected_path,
        expected_name=expected_name,
        expected_revision=expected_revision,
        expected_splits=expected_splits,
    )


def dataset_receipt_from_mapping(value: Mapping[str, Any]) -> EvaluatorDatasetReceipt:
    """Normalize an injected receipt without granting it filesystem authority."""

    try:
        return EvaluatorDatasetReceipt.from_mapping(value)
    except (TypeError, ValueError) as error:
        raise _invalid("Evaluator dataset receipt is invalid") from error


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_dataset_cas_invalid")


__all__ = [
    "RECEIPT_NAME",
    "dataset_receipt_from_mapping",
    "load_evaluator_dataset_receipt",
    "verify_evaluator_dataset_cas",
    "verify_evaluator_dataset_root",
]
