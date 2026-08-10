from __future__ import annotations

from pathlib import Path

import pytest

from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_contract_factory import (
    build_lm_eval_execution_contract,
)
from apex.benchmark.evaluator_dataset import (
    EvaluatorDatasetFile,
    EvaluatorDatasetReceipt,
)
from apex.benchmark.evaluator_policy import EvaluatorPolicy
from apex.benchmark.evaluator_task_materialization import (
    EvaluatorTaskMaterializationReceipt,
)
from apex.core import ConfigurationError, sha256_file
from apex.ports import BenchmarkPass, MagpieAttestationRequest


def _policy() -> EvaluatorPolicy:
    return EvaluatorPolicy(
        policy_id="apex-lm-eval-gsm8k-v2",
        tasks="gsm8k",
        task_definition_path="utils/evals/gsm8k.yaml",
        task_definition_sha256="1" * 64,
        dataset_path="openai/gsm8k",
        dataset_name="main",
        dataset_revision="2" * 40,
        primary_metric="exact_match,strict-match",
        max_length=2248,
        max_gen_tokens=480,
    )


def _dataset() -> EvaluatorDatasetReceipt:
    return EvaluatorDatasetReceipt(
        repository="https://huggingface.co/datasets/openai/gsm8k",
        path="openai/gsm8k",
        name="main",
        revision="2" * 40,
        files=(
            EvaluatorDatasetFile(
                "test", EvaluatorArtifactReceipt("test/data", 1, "3" * 64)
            ),
            EvaluatorDatasetFile(
                "train", EvaluatorArtifactReceipt("train/data", 1, "4" * 64)
            ),
        ),
    )


def _task() -> EvaluatorTaskMaterializationReceipt:
    return EvaluatorTaskMaterializationReceipt(
        source_commit="5" * 40,
        source_tree="6" * 40,
        source_path="utils/evals/gsm8k.yaml",
        source_sha256="1" * 64,
        effective_path="task/gsm8k.yaml",
        effective_sha256="7" * 64,
        dataset_revision="2" * 40,
        dataset_receipt_sha256=_dataset().sha256,
    )


def _request(tmp_path: Path, **changes) -> MagpieAttestationRequest:
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    policy = _policy()
    values = {
        "run_id": "baseline-measurement",
        "pass_type": BenchmarkPass.MEASUREMENT,
        "config_path": config,
        "run_root": tmp_path / "run",
        "benchmark_argv": ("python", "-m", "Magpie"),
        "config_sha256": sha256_file(config),
        "execution_mode": "docker",
        "lifecycle": "one_shot",
        "requested_image": "serving/image:fixed",
        "gpu_lease": {},
        "evaluator_policy": policy.to_dict(),
        "evaluator_policy_lock": {
            "schema": "apex.evaluator-policy-lock/v2",
            "lock_sha256": "8" * 64,
            "policy_id": policy.policy_id,
            "primary_metric": policy.primary_metric,
            "sample_logging_required": True,
            "task": {
                "name": "gsm8k",
                "definition_path": policy.task_definition_path,
                "definition_sha256": policy.task_definition_sha256,
            },
            "dataset": {
                "repository": "https://huggingface.co/datasets/openai/gsm8k",
                "path": policy.dataset_path,
                "name": policy.dataset_name,
                "revision": policy.dataset_revision,
                "splits": ["test", "train"],
                "files": [item.to_dict() for item in _dataset().files],
            },
        },
        "lm_eval_runtime": {
            "sha256": "9" * 64,
            "manifest_sha256": "a" * 64,
            "lock_sha256": "b" * 64,
            "identity": {
                "base_image_id": "sha256:" + "c" * 64,
                "base_image_repo_digest": "example/eval@sha256:" + "d" * 64,
            },
        },
        "model": "Qwen/example",
        "evaluator_endpoint_port": 8888,
        "evaluator_concurrent_requests": 64,
        "evaluator_timeout_seconds": 3600,
    }
    values.update(changes)
    return MagpieAttestationRequest(**values)


def test_builds_complete_contract_from_verified_receipts(tmp_path: Path) -> None:
    contract = build_lm_eval_execution_contract(
        _request(tmp_path), task=_task(), dataset=_dataset()
    )

    value = contract.to_dict()
    assert value["run"]["pass_type"] == "measurement"
    assert value["task"]["effective_definition_sha256"] == "7" * 64
    assert value["dataset"]["revision"] == "2" * 40
    assert value["runtime"]["manifest_sha256"] == "a" * 64
    assert value["command"]["evaluator_argv"] == list(contract.argv)
    assert value["command"]["container_argv"] == list(contract.sidecar_argv)
    assert value["command"]["cwd"] == "/authority"
    assert value["outputs"]["sample_logging_required"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"execution_mode": "local"},
        {"lifecycle": "reuse"},
        {"pass_type": BenchmarkPass.DIAGNOSTIC},
        {"model": ""},
    ],
)
def test_rejects_non_docker_or_incomplete_formal_request(
    tmp_path: Path, change: dict[str, object]
) -> None:
    with pytest.raises(ConfigurationError, match="incomplete"):
        build_lm_eval_execution_contract(
            _request(tmp_path, **change), task=_task(), dataset=_dataset()
        )


def test_rejects_dataset_or_task_swap(tmp_path: Path) -> None:
    dataset = _dataset()
    swapped = EvaluatorDatasetReceipt(
        dataset.repository,
        dataset.path,
        dataset.name,
        "e" * 40,
        dataset.files,
    )
    with pytest.raises(ConfigurationError, match="policy lock"):
        build_lm_eval_execution_contract(
            _request(tmp_path), task=_task(), dataset=swapped
        )

    task = _task()
    swapped_task = EvaluatorTaskMaterializationReceipt(
        task.source_commit,
        task.source_tree,
        task.source_path,
        "f" * 64,
        task.effective_path,
        task.effective_sha256,
        task.dataset_revision,
        task.dataset_receipt_sha256,
    )
    with pytest.raises(ConfigurationError, match="policy lock"):
        build_lm_eval_execution_contract(
            _request(tmp_path), task=swapped_task, dataset=dataset
        )
