from __future__ import annotations

import pytest

from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_dataset import (
    EvaluatorDatasetFile,
    EvaluatorDatasetReceipt,
)
from apex.benchmark.evaluator_execution import (
    LmEvalExecutionContract,
    LmEvalExecutionReceipt,
    validate_execution_binding,
    validate_receipt_against_contract,
)


def _artifact(path: str, marker: str) -> EvaluatorArtifactReceipt:
    return EvaluatorArtifactReceipt(path, 7, marker * 64)


def _dataset() -> EvaluatorDatasetReceipt:
    return EvaluatorDatasetReceipt(
        repository="https://huggingface.co/datasets/openai/gsm8k",
        path="openai/gsm8k",
        name="main",
        revision="1" * 40,
        files=(
            EvaluatorDatasetFile("test", _artifact("test/data.parquet", "2")),
            EvaluatorDatasetFile("train", _artifact("train/data.parquet", "3")),
        ),
    )


def _contract(dataset: EvaluatorDatasetReceipt) -> LmEvalExecutionContract:
    return LmEvalExecutionContract(
        run_id="baseline-measurement",
        config_sha256="0" * 64,
        model="Qwen/example",
        endpoint_port=8888,
        policy_sha256="4" * 64,
        policy_lock_sha256="d" * 64,
        task_definition_sha256="5" * 64,
        effective_task_definition_sha256="e" * 64,
        task_materialization_receipt_sha256="f" * 64,
        dataset_receipt_sha256=dataset.sha256,
        dataset_revision=dataset.revision,
        runtime_sha256="6" * 64,
        runtime_manifest_sha256="a" * 64,
        runtime_lock_sha256="b" * 64,
        launcher_sha256="0" * 64,
        image_repo_digest="example/evaluator@sha256:" + "7" * 64,
        image_id="sha256:" + "8" * 64,
        max_length=2248,
        max_gen_tokens=480,
        concurrent_requests=64,
        timeout_seconds=3600,
    )


def _receipt(contract: LmEvalExecutionContract) -> LmEvalExecutionReceipt:
    return LmEvalExecutionReceipt(
        contract_sha256=contract.sha256,
        config_sha256=contract.config_sha256,
        policy_sha256=contract.policy_sha256,
        policy_lock_sha256=contract.policy_lock_sha256,
        task_definition_sha256=contract.task_definition_sha256,
        effective_task_definition_sha256=contract.effective_task_definition_sha256,
        task_materialization_receipt_sha256=contract.task_materialization_receipt_sha256,
        dataset_receipt_sha256=contract.dataset_receipt_sha256,
        dataset_revision=contract.dataset_revision,
        runtime_sha256=contract.runtime_sha256,
        runtime_manifest_sha256=contract.runtime_manifest_sha256,
        runtime_lock_sha256=contract.runtime_lock_sha256,
        launcher_sha256=contract.launcher_sha256,
        image_repo_digest=contract.image_repo_digest,
        image_id="sha256:" + "8" * 64,
        container_id="9" * 64,
        listener_receipt_sha256="a" * 64,
        sidecar_spec_sha256="e" * 64,
        created_observation_sha256="f" * 64,
        exited_observation_sha256="0" * 64,
        broker_receipt_sha256="1" * 64,
        container_cleanup_sha256="2" * 64,
        runtime_probe_sha256="b" * 64,
        runtime_publication_sha256="3" * 64,
        result_artifacts=(_artifact("lm_eval/results.json", "c"),),
        sample_artifacts=(_artifact("lm_eval/samples_gsm8k.jsonl", "d"),),
    )


def test_freezes_offline_dataset_and_exact_sidecar_invocation() -> None:
    dataset = _dataset()
    contract = _contract(dataset)

    assert dataset.to_dict()["offline"] is True
    assert dataset.to_dict()["sha256"] == dataset.sha256
    assert contract.argv[:5] == (
        "python3",
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
    )
    assert "--log_samples" in contract.argv
    assert "num_concurrent=64" in contract.argv[-3]
    assert contract.environment == {
        "HF_HOME": "/tmp/huggingface",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "OPENAI_API_KEY": "EMPTY",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "/evaluator/runtime/site-packages",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/tmp/xdg",
    }
    assert "base_url=http://127.0.0.1:18080" in contract.argv[-3]
    assert contract.sidecar_argv[-len(contract.argv) :] == contract.argv
    assert contract.to_dict()["security"] == {
        "network_mode": "none",
        "read_only_root": True,
        "gpu_devices": [],
        "cap_drop": ["ALL"],
        "no_new_privileges": True,
    }


def test_execution_receipt_round_trips_and_binds_outputs() -> None:
    contract = _contract(_dataset())
    receipt = _receipt(contract)
    value = receipt.to_dict()

    assert LmEvalExecutionReceipt.from_mapping(value) == receipt
    assert validate_receipt_against_contract(receipt, contract) is None
    assert validate_execution_binding(
        value,
        expected_policy={
            "sha256": contract.policy_sha256,
            "task_definition_sha256": contract.task_definition_sha256,
        },
        expected_runtime_sha256=contract.runtime_sha256,
        expected_image_repo_digest=contract.image_repo_digest,
        result_artifacts=tuple(item.to_dict() for item in receipt.result_artifacts),
        sample_artifacts=tuple(item.to_dict() for item in receipt.sample_artifacts),
    ) is None


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("policy_sha256", "quality_evaluator_policy_mismatch"),
        ("task_definition_sha256", "quality_evaluator_task_definition_mismatch"),
        ("runtime_sha256", "quality_evaluator_runtime_mismatch"),
        ("image_repo_digest", "quality_evaluator_image_mismatch"),
    ],
)
def test_execution_binding_rejects_expected_identity_drift(
    field: str, reason: str
) -> None:
    contract = _contract(_dataset())
    receipt = _receipt(contract)
    policy = {
        "sha256": contract.policy_sha256,
        "task_definition_sha256": contract.task_definition_sha256,
    }
    runtime = contract.runtime_sha256
    image = contract.image_repo_digest
    if field == "policy_sha256":
        policy["sha256"] = "e" * 64
    elif field in policy:
        policy[field] = "e" * 64
    elif field == "runtime_sha256":
        runtime = "e" * 64
    else:
        image = "example/other@sha256:" + "e" * 64

    assert validate_execution_binding(
        receipt.to_dict(),
        expected_policy=policy,
        expected_runtime_sha256=runtime,
        expected_image_repo_digest=image,
        result_artifacts=tuple(item.to_dict() for item in receipt.result_artifacts),
        sample_artifacts=tuple(item.to_dict() for item in receipt.sample_artifacts),
    ) == reason


def test_execution_receipt_rejects_digest_and_output_tampering() -> None:
    contract = _contract(_dataset())
    receipt = _receipt(contract)
    value = receipt.to_dict()
    value["receipt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="digest"):
        LmEvalExecutionReceipt.from_mapping(value)

    assert validate_execution_binding(
        receipt.to_dict(),
        expected_policy={
            "sha256": contract.policy_sha256,
            "task_definition_sha256": contract.task_definition_sha256,
        },
        expected_runtime_sha256=contract.runtime_sha256,
        expected_image_repo_digest=contract.image_repo_digest,
        result_artifacts=({"path": "lm_eval/results.json", "size_bytes": 8, "sha256": "c" * 64},),
        sample_artifacts=tuple(item.to_dict() for item in receipt.sample_artifacts),
    ) == "quality_evaluator_result_receipt_mismatch"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EvaluatorArtifactReceipt("../escape", 1, "1" * 64),
        lambda: EvaluatorArtifactReceipt("result.json", 0, "1" * 64),
        lambda: LmEvalExecutionContract(
            run_id="run",
            config_sha256="0" * 64,
            model="model",
            endpoint_port=8888,
            policy_sha256="1" * 64,
            policy_lock_sha256="5" * 64,
            task_definition_sha256="2" * 64,
            effective_task_definition_sha256="6" * 64,
            task_materialization_receipt_sha256="7" * 64,
            dataset_receipt_sha256="3" * 64,
            dataset_revision="8" * 40,
            runtime_sha256="4" * 64,
            runtime_manifest_sha256="9" * 64,
            runtime_lock_sha256="a" * 64,
            launcher_sha256="b" * 64,
            image_repo_digest="mutable:latest",
            image_id="sha256:" + "b" * 64,
            max_length=100,
            max_gen_tokens=100,
            concurrent_requests=1,
            timeout_seconds=3600,
        ),
    ],
)
def test_rejects_unsafe_or_mutable_contract_inputs(factory) -> None:
    with pytest.raises(ValueError):
        factory()
