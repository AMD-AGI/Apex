from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_dataset import (
    EvaluatorDatasetFile,
    EvaluatorDatasetReceipt,
)
from apex.benchmark.evaluator_policy import EvaluatorPolicy
from apex.benchmark.evaluator_preparation import LmEvalExecutionPreparer
from apex.core import ConfigurationError, sha256_file
from apex.ports import BenchmarkPass, MagpieAttestationRequest
from apex.runtime import (
    DependencyReceipt,
    EvaluatorDatasetLockFile,
    EvaluatorPolicyLock,
    LmEvalRuntimeReceipt,
)


def _dataset_root(tmp_path: Path, revision: str) -> Path:
    root = tmp_path / "dataset"
    files = root / "files"
    records = []
    for split, content in (("test", b"test"), ("train", b"train")):
        path = files / split / "data"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        records.append(
            EvaluatorDatasetFile(
                split,
                EvaluatorArtifactReceipt(
                    f"{split}/data", len(content), hashlib.sha256(content).hexdigest()
                ),
            )
        )
    receipt = EvaluatorDatasetReceipt(
        "https://huggingface.co/datasets/openai/gsm8k",
        "openai/gsm8k",
        "main",
        revision,
        tuple(records),
    )
    manifest = root / "evaluator_dataset_receipt.json"
    manifest.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    for path in (*files.rglob("*"), files, manifest, root):
        path.chmod(0o400 if path.is_file() else 0o500)
    return root


def _receipt(tmp_path: Path) -> DependencyReceipt:
    inferencex = tmp_path / "inferencex"
    task = inferencex / "utils" / "evals" / "gsm8k.yaml"
    task.parent.mkdir(parents=True)
    task.write_text(
        "task: gsm8k\ndataset_path: openai/gsm8k\ndataset_name: main\n",
        encoding="utf-8",
    )
    benchmark_library = inferencex / "benchmarks" / "benchmark_lib.sh"
    benchmark_library.parent.mkdir()
    benchmark_library.write_text(
        "run_eval() { return 9; }\nappend_lm_eval_summary() { return 9; }\n",
        encoding="utf-8",
    )
    magpie = tmp_path / "magpie"
    scripts = magpie / "Magpie" / "scripts" / "benchmark"
    scripts.mkdir(parents=True)
    (scripts / "vllm_mi300x.sh").write_text("run_eval --framework lm-eval\n")
    policy_lock = EvaluatorPolicyLock(
        path=tmp_path / "policy.lock.json",
        lock_sha256="1" * 64,
        policy_id="apex-lm-eval-gsm8k-v2",
        primary_metric="exact_match,strict-match",
        sample_logging_required=True,
        task_name="gsm8k",
        task_definition_path="utils/evals/gsm8k.yaml",
        task_definition_sha256=sha256_file(task),
        dataset_repository="https://huggingface.co/datasets/openai/gsm8k",
        dataset_path="openai/gsm8k",
        dataset_name="main",
        dataset_revision="2" * 40,
        dataset_splits=("test", "train"),
        dataset_files=(
            EvaluatorDatasetLockFile(
                "test", "test/data", 4, hashlib.sha256(b"test").hexdigest()
            ),
            EvaluatorDatasetLockFile(
                "train", "train/data", 5, hashlib.sha256(b"train").hexdigest()
            ),
        ),
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    lm_eval = LmEvalRuntimeReceipt(
        runtime,
        "3" * 64,
        "4" * 64,
        {
            "base_image_id": "sha256:" + "5" * 64,
            "base_image_repo_digest": "example/eval@sha256:" + "6" * 64,
        },
        1,
        "7" * 64,
    )
    return DependencyReceipt(
        "apex.dependencies.receipt/v1",
        "8" * 64,
        Path("/python"),
        {"inferencex": inferencex, "magpie": magpie},
        {"inferencex": "9" * 40, "magpie": "b" * 40},
        {
            "dependencies": {
                "inferencex": {"tree": "a" * 40},
                "magpie": {"tree": "c" * 40},
            }
        },
        lm_eval_runtime=lm_eval,
        evaluator_policy=policy_lock,
    )


def _request(tmp_path: Path, receipt: DependencyReceipt) -> MagpieAttestationRequest:
    config = tmp_path / "config.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    lock = receipt.evaluator_policy
    runtime = receipt.lm_eval_runtime
    assert lock is not None and runtime is not None
    policy = EvaluatorPolicy(
        lock.policy_id,
        lock.task_name,
        lock.task_definition_path,
        lock.task_definition_sha256,
        lock.dataset_path,
        lock.dataset_name,
        lock.dataset_revision,
        lock.primary_metric,
        2248,
        480,
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    return MagpieAttestationRequest(
        "baseline-measurement",
        BenchmarkPass.MEASUREMENT,
        config,
        run_root,
        ("python", "-m", "Magpie"),
        sha256_file(config),
        "docker",
        "one_shot",
        "serving/image:fixed",
        {},
        evaluator_policy=policy.to_dict(),
        evaluator_policy_lock=lock.to_dict(),
        lm_eval_runtime=runtime.to_dict(),
        model="Qwen/example",
        evaluator_endpoint_port=8888,
        evaluator_concurrent_requests=64,
        evaluator_timeout_seconds=3600,
    )


def test_prepares_private_immutable_inputs_without_source_mutation(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    request = _request(tmp_path, receipt)
    source = receipt.root("inferencex") / "utils/evals/gsm8k.yaml"
    before = source.read_bytes()

    prepared = LmEvalExecutionPreparer(
        receipt, _dataset_root(tmp_path, "2" * 40)
    ).prepare(request)

    assert source.read_bytes() == before
    assert prepared.task_mount == prepared.authority_root / "task-materialization/task"
    assert prepared.dataset_mount.name == "files"
    assert prepared.runtime_mount == receipt.lm_eval_runtime.root
    assert prepared.runtime_receipt == receipt.lm_eval_runtime
    assert prepared.output_root.is_dir()
    assert prepared.contract.config_sha256 == request.config_sha256
    assert prepared.contract.dataset_revision == "2" * 40
    assert stat.S_IMODE(prepared.contract_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(prepared.task_receipt_path.stat().st_mode) == 0o400
    assert prepared.launch_config_path.is_file()
    assert prepared.inferencex_projection.root.is_dir()
    assert prepared.launch_config_receipt.canonical_config_sha256 == request.config_sha256


def test_rejects_dataset_revision_drift_before_private_materialization(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    request = _request(tmp_path, receipt)

    with pytest.raises(ConfigurationError, match="dataset CAS"):
        LmEvalExecutionPreparer(
            receipt, _dataset_root(tmp_path, "f" * 40)
        ).prepare(request)

    assert not (request.run_root / "authority").exists()


def test_rejects_request_receipt_swap(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    request = _request(tmp_path, receipt)
    request = replace(request, lm_eval_runtime={"sha256": "0" * 64})

    with pytest.raises(ConfigurationError, match="runtime differs"):
        LmEvalExecutionPreparer(
            receipt, _dataset_root(tmp_path, "2" * 40)
        ).prepare(request)
