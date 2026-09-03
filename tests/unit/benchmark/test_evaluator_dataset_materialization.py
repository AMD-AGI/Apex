from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pytest

from apex.benchmark.evaluator_dataset_cas import verify_evaluator_dataset_root
from apex.benchmark.evaluator_dataset_materialization import (
    EvaluatorDatasetMaterializationInput,
    materialize_evaluator_dataset_cas,
    verify_dataset_receipt_against_inputs,
)
from apex.core import ConfigurationError


_REPOSITORY = "https://huggingface.co/datasets/openai/gsm8k"
_REVISION = "7" * 40


def _inputs(tmp_path: Path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    values = []
    for split, payload in (("test", b"PAR1-test-PAR1"), ("train", b"PAR1-train-PAR1")):
        source = downloads / f"{split}.parquet"
        source.write_bytes(payload)
        values.append(
            EvaluatorDatasetMaterializationInput(
                split=split,
                path=f"main/{split}.parquet",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                source=source,
            )
        )
    return tuple(values)


def _materialize(tmp_path: Path):
    destination = tmp_path / "cas"
    inputs = _inputs(tmp_path)
    receipt = materialize_evaluator_dataset_cas(
        destination,
        repository=_REPOSITORY,
        dataset_path="openai/gsm8k",
        dataset_name="main",
        revision=_REVISION,
        files=inputs,
    )
    return destination, inputs, receipt


def test_materializes_exact_sealed_dataset_cas(tmp_path: Path) -> None:
    root, inputs, receipt = _materialize(tmp_path)

    observed = verify_evaluator_dataset_root(
        root,
        expected_repository=_REPOSITORY,
        expected_path="openai/gsm8k",
        expected_name="main",
        expected_revision=_REVISION,
        expected_splits=("test", "train"),
    )
    verify_dataset_receipt_against_inputs(observed, inputs)
    assert observed == receipt
    assert stat.S_IMODE(root.stat().st_mode) == 0o500
    assert all(
        stat.S_IMODE(path.stat().st_mode) == (0o400 if path.is_file() else 0o500)
        for path in root.rglob("*")
    )


def test_rejects_tampered_or_linked_download(tmp_path: Path) -> None:
    inputs = list(_inputs(tmp_path))
    inputs[0] = EvaluatorDatasetMaterializationInput(
        inputs[0].split,
        inputs[0].path,
        inputs[0].size_bytes,
        "0" * 64,
        inputs[0].source,
    )
    with pytest.raises(ConfigurationError, match="differs from its lock"):
        materialize_evaluator_dataset_cas(
            tmp_path / "tampered-cas",
            repository=_REPOSITORY,
            dataset_path="openai/gsm8k",
            dataset_name="main",
            revision=_REVISION,
            files=tuple(inputs),
        )

    linked = tmp_path / "linked.parquet"
    linked.symlink_to(inputs[0].source)
    inputs[0] = EvaluatorDatasetMaterializationInput(
        inputs[0].split,
        inputs[0].path,
        inputs[0].size_bytes,
        inputs[0].sha256,
        linked,
    )
    with pytest.raises(ConfigurationError, match="Cannot open"):
        materialize_evaluator_dataset_cas(
            tmp_path / "linked-cas",
            repository=_REPOSITORY,
            dataset_path="openai/gsm8k",
            dataset_name="main",
            revision=_REVISION,
            files=tuple(inputs),
        )


def test_never_overwrites_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "cas"
    destination.mkdir()

    with pytest.raises(ConfigurationError, match="inputs are invalid"):
        materialize_evaluator_dataset_cas(
            destination,
            repository=_REPOSITORY,
            dataset_path="openai/gsm8k",
            dataset_name="main",
            revision=_REVISION,
            files=_inputs(tmp_path),
        )


def test_rejects_symlinked_cas_root(tmp_path: Path) -> None:
    root, _, _ = _materialize(tmp_path)
    linked = tmp_path / "linked-cas"
    linked.symlink_to(root, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="root is unsafe"):
        verify_evaluator_dataset_root(
            linked,
            expected_repository=_REPOSITORY,
            expected_path="openai/gsm8k",
            expected_name="main",
            expected_revision=_REVISION,
            expected_splits=("test", "train"),
        )
