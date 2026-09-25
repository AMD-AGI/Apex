from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.core import sha256_file
from apex.runtime import load_evaluator_policy_lock
from apex.runtime.repositories import BootstrapError


_TASK = (
    "task: gsm8k\n"
    "dataset_path: openai/gsm8k\n"
    "dataset_name: main\n"
    "test_split: test\n"
)


def _case(tmp_path: Path) -> tuple[Path, Path]:
    inferencex = tmp_path / "InferenceX"
    task = inferencex / "utils" / "evals" / "gsm8k.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(_TASK, encoding="utf-8")
    lock = tmp_path / "evaluator_policy.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": "apex.evaluator-policy-lock/v2",
                "policy_id": "apex-lm-eval-gsm8k-v2",
                "primary_metric": "exact_match,strict-match",
                "sample_logging_required": True,
                "task": {
                    "name": "gsm8k",
                    "definition_dependency": "inferencex",
                    "definition_path": "utils/evals/gsm8k.yaml",
                    "definition_sha256": sha256_file(task),
                },
                "dataset": {
                    "repository": "https://huggingface.co/datasets/openai/gsm8k",
                    "path": "openai/gsm8k",
                    "name": "main",
                    "revision": "7" * 40,
                    "splits": ["test", "train"],
                    "files": [
                        {
                            "split": "test",
                            "path": "main/test.parquet",
                            "size_bytes": 4,
                            "sha256": "8" * 64,
                        },
                        {
                            "split": "train",
                            "path": "main/train.parquet",
                            "size_bytes": 5,
                            "sha256": "9" * 64,
                        },
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return lock, inferencex


def test_loads_and_rehashes_task_and_dataset_identity(tmp_path: Path) -> None:
    lock, inferencex = _case(tmp_path)

    receipt = load_evaluator_policy_lock(lock, inferencex_root=inferencex)

    assert receipt.lock_sha256 == sha256_file(lock)
    assert receipt.task_definition_sha256 == sha256_file(
        inferencex / "utils/evals/gsm8k.yaml"
    )
    assert receipt.dataset_revision == "7" * 40
    assert tuple(item.split for item in receipt.dataset_files) == ("test", "train")
    assert receipt.env()["EVAL_TASKS_DIR"] == "utils/evals/gsm8k.yaml"


@pytest.mark.parametrize("mutation", ("digest", "dataset", "escape"))
def test_rejects_task_or_dataset_drift(tmp_path: Path, mutation: str) -> None:
    lock, inferencex = _case(tmp_path)
    value = json.loads(lock.read_text(encoding="utf-8"))
    if mutation == "digest":
        value["task"]["definition_sha256"] = "0" * 64
    elif mutation == "dataset":
        value["dataset"]["path"] = "other/dataset"
    else:
        value["task"]["definition_path"] = "../outside.yaml"
    lock.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(BootstrapError):
        load_evaluator_policy_lock(lock, inferencex_root=inferencex)
