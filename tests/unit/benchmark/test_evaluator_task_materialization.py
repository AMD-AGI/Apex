from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from apex.benchmark.evaluator_task_materialization import (
    materialize_evaluator_task,
)
from apex.core import ConfigurationError, sha256_file


_REVISION = "1" * 40


def _source(tmp_path: Path, text: str | None = None) -> tuple[Path, Path]:
    root = tmp_path / "inferencex"
    task = root / "utils" / "evals" / "gsm8k.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(
        text
        or "task: gsm8k\ndataset_path: openai/gsm8k\ndataset_name: main\n",
        encoding="utf-8",
    )
    return root, task


def _materialize(source: Path, task: Path, destination: Path):
    return materialize_evaluator_task(
        source,
        destination,
        source_commit="2" * 40,
        source_tree="3" * 40,
        definition_path="utils/evals/gsm8k.yaml",
        definition_sha256=sha256_file(task),
        dataset_path="openai/gsm8k",
        dataset_name="main",
        dataset_revision=_REVISION,
        dataset_receipt_sha256="4" * 64,
        dataset_files={
            "test": ("/evaluator/dataset/main/test.parquet",),
            "train": ("/evaluator/dataset/main/train.parquet",),
        },
    )


def test_materializes_revision_without_changing_source(tmp_path: Path) -> None:
    source, task = _source(tmp_path)
    before = task.read_bytes()
    destination = tmp_path / "authority"

    receipt = _materialize(source, task, destination)
    output = destination / "task" / "gsm8k.yaml"
    value = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert task.read_bytes() == before
    assert value["dataset_path"] == "parquet"
    assert "dataset_name" not in value
    assert value["dataset_kwargs"] == {
        "revision": _REVISION,
        "data_files": {
            "test": ["/evaluator/dataset/main/test.parquet"],
            "train": ["/evaluator/dataset/main/train.parquet"],
        },
    }
    assert receipt.source_sha256 == sha256_file(task)
    assert receipt.effective_sha256 == sha256_file(output)
    assert receipt.to_dict()["receipt_sha256"] == receipt.sha256
    assert stat.S_IMODE(output.stat().st_mode) == 0o400


def test_materialization_is_content_deterministic_across_roots(tmp_path: Path) -> None:
    source, task = _source(tmp_path)

    first = _materialize(source, task, tmp_path / "first")
    second = _materialize(source, task, tmp_path / "second")

    assert first == second
    assert first.sha256 == second.sha256
    assert (tmp_path / "first/task/gsm8k.yaml").read_bytes() == (
        tmp_path / "second/task/gsm8k.yaml"
    ).read_bytes()


def test_rejects_source_digest_drift_and_conflicting_revision(tmp_path: Path) -> None:
    source, task = _source(tmp_path)
    digest = sha256_file(task)
    task.write_text(task.read_text() + "metadata: {}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="differs"):
        materialize_evaluator_task(
            source,
            tmp_path / "drift",
            source_commit="2" * 40,
            source_tree="3" * 40,
            definition_path="utils/evals/gsm8k.yaml",
            definition_sha256=digest,
            dataset_path="openai/gsm8k",
            dataset_name="main",
            dataset_revision=_REVISION,
            dataset_receipt_sha256="4" * 64,
            dataset_files={
                "test": ("/evaluator/dataset/main/test.parquet",),
                "train": ("/evaluator/dataset/main/train.parquet",),
            },
        )

    source2, task2 = _source(
        tmp_path / "conflict",
        "task: gsm8k\ndataset_path: openai/gsm8k\ndataset_name: main\n"
        "dataset_kwargs: {revision: '0000000000000000000000000000000000000000'}\n",
    )
    with pytest.raises(ConfigurationError, match="conflicts"):
        _materialize(source2, task2, tmp_path / "conflict-output")


def test_rejects_linked_source_or_existing_destination(tmp_path: Path) -> None:
    source, task = _source(tmp_path)
    link = source / "utils" / "evals" / "linked.yaml"
    link.symlink_to(task)

    with pytest.raises(ConfigurationError, match="differs"):
        materialize_evaluator_task(
            source,
            tmp_path / "linked-output",
            source_commit="2" * 40,
            source_tree="3" * 40,
            definition_path="utils/evals/linked.yaml",
            definition_sha256=sha256_file(task),
            dataset_path="openai/gsm8k",
            dataset_name="main",
            dataset_revision=_REVISION,
            dataset_receipt_sha256="4" * 64,
            dataset_files={
                "test": ("/evaluator/dataset/main/test.parquet",),
                "train": ("/evaluator/dataset/main/train.parquet",),
            },
        )

    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(ConfigurationError, match="Cannot materialize"):
        _materialize(source, task, destination)


def test_output_is_not_hardlinked_to_source(tmp_path: Path) -> None:
    source, task = _source(tmp_path)
    destination = tmp_path / "authority"
    _materialize(source, task, destination)
    output = destination / "task" / "gsm8k.yaml"

    assert output.stat().st_ino != task.stat().st_ino
    assert output.stat().st_nlink == 1
    assert os.access(output, os.R_OK)
