from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from apex.benchmark.evaluator_dataset_cas import (
    RECEIPT_NAME,
    verify_evaluator_dataset_root,
)
from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_dataset import (
    EvaluatorDatasetFile,
    EvaluatorDatasetReceipt,
)
from apex.core import ConfigurationError, sha256_file


_REPOSITORY = "https://huggingface.co/datasets/openai/gsm8k"
_REVISION = "1" * 40


def _cas(tmp_path: Path) -> tuple[Path, EvaluatorDatasetReceipt]:
    root = tmp_path / "dataset-cas"
    files = root / "files"
    test = files / "test" / "data.parquet"
    train = files / "train" / "data.parquet"
    test.parent.mkdir(parents=True)
    train.parent.mkdir(parents=True)
    test.write_bytes(b"test-dataset")
    train.write_bytes(b"train-dataset")
    receipt = EvaluatorDatasetReceipt(
        repository=_REPOSITORY,
        path="openai/gsm8k",
        name="main",
        revision=_REVISION,
        files=(
            EvaluatorDatasetFile(
                "test",
                EvaluatorArtifactReceipt(
                    "test/data.parquet", test.stat().st_size, sha256_file(test)
                ),
            ),
            EvaluatorDatasetFile(
                "train",
                EvaluatorArtifactReceipt(
                    "train/data.parquet", train.stat().st_size, sha256_file(train)
                ),
            ),
        ),
    )
    manifest = root / RECEIPT_NAME
    manifest.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    for path in (test, train, test.parent, train.parent, files, manifest, root):
        path.chmod(0o400 if path.is_file() else 0o500)
    return root, receipt


def _verify(root: Path):
    return verify_evaluator_dataset_root(
        root,
        expected_repository=_REPOSITORY,
        expected_path="openai/gsm8k",
        expected_name="main",
        expected_revision=_REVISION,
        expected_splits=("test", "train"),
    )


def test_verifies_exact_read_only_offline_dataset_cas(tmp_path: Path) -> None:
    root, receipt = _cas(tmp_path)

    assert _verify(root) == receipt


def test_rejects_policy_revision_or_split_drift(tmp_path: Path) -> None:
    root, _ = _cas(tmp_path)

    with pytest.raises(ConfigurationError, match="policy lock"):
        verify_evaluator_dataset_root(
            root,
            expected_repository=_REPOSITORY,
            expected_path="openai/gsm8k",
            expected_name="main",
            expected_revision="2" * 40,
            expected_splits=("test", "train"),
        )
    with pytest.raises(ConfigurationError, match="policy lock"):
        verify_evaluator_dataset_root(
            root,
            expected_repository=_REPOSITORY,
            expected_path="openai/gsm8k",
            expected_name="main",
            expected_revision=_REVISION,
            expected_splits=("test",),
        )


def test_rejects_writable_or_tampered_dataset_file(tmp_path: Path) -> None:
    root, _ = _cas(tmp_path)
    target = root / "files" / "test" / "data.parquet"
    target.chmod(0o600)

    with pytest.raises(ConfigurationError, match="verification failed"):
        _verify(root)

    root.chmod(0o700)
    target.chmod(0o600)
    target.write_bytes(b"tampered")
    target.chmod(0o400)
    root.chmod(0o500)
    with pytest.raises(ConfigurationError, match="verification failed"):
        _verify(root)


def test_rejects_linked_manifest_or_dataset_file(tmp_path: Path) -> None:
    root, _ = _cas(tmp_path)
    root.chmod(0o700)
    manifest = root / RECEIPT_NAME
    backup = root / "receipt.backup"
    manifest.rename(backup)
    manifest.symlink_to(backup)
    root.chmod(0o500)
    with pytest.raises(ConfigurationError, match="unsafe"):
        _verify(root)

    root.chmod(0o700)
    manifest.unlink()
    backup.rename(manifest)
    manifest.chmod(0o400)
    root.chmod(0o500)
    files = root / "files"
    files.chmod(0o700)
    target = files / "test" / "data.parquet"
    target.parent.chmod(0o700)
    target.chmod(0o600)
    linked = files / "test" / "linked.parquet"
    os.link(target, linked)
    target.chmod(0o400)
    target.parent.chmod(0o500)
    files.chmod(0o500)
    with pytest.raises(ConfigurationError, match="verification failed"):
        _verify(root)


def test_rejects_receipt_digest_tampering(tmp_path: Path) -> None:
    root, _ = _cas(tmp_path)
    root.chmod(0o700)
    manifest = root / RECEIPT_NAME
    manifest.chmod(0o600)
    value = json.loads(manifest.read_text())
    value["sha256"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    manifest.chmod(0o400)
    root.chmod(0o500)

    with pytest.raises(ConfigurationError, match="invalid"):
        _verify(root)
