"""Compose a frozen lm-eval sidecar contract from verified Apex receipts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError, sha256_file
from apex.ports import BenchmarkPass, MagpieAttestationRequest

from .evaluator_dataset import EvaluatorDatasetReceipt
from .evaluator_execution import LmEvalExecutionContract
from .evaluator_policy import EvaluatorPolicy
from .evaluator_task_materialization import EvaluatorTaskMaterializationReceipt


_DIGEST = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


def build_lm_eval_execution_contract(
    request: MagpieAttestationRequest,
    *,
    task: EvaluatorTaskMaterializationReceipt,
    dataset: EvaluatorDatasetReceipt,
) -> LmEvalExecutionContract:
    """Bind request, policy, task, dataset, runtime, and exact image identity."""

    _validate_request(request)
    policy = _policy(request.evaluator_policy)
    lock = _mapping(request.evaluator_policy_lock, "evaluator policy lock")
    runtime = _mapping(request.lm_eval_runtime, "lm-eval runtime receipt")
    _validate_policy_lock(policy, lock, task, dataset)
    identity = _mapping(runtime.get("identity"), "lm-eval runtime identity")
    _validate_runtime(runtime, identity)
    assert request.model is not None
    assert request.evaluator_endpoint_port is not None
    assert request.evaluator_concurrent_requests is not None
    assert request.evaluator_timeout_seconds is not None
    return LmEvalExecutionContract(
        run_id=request.run_id,
        config_sha256=request.config_sha256,
        model=request.model,
        endpoint_port=request.evaluator_endpoint_port,
        policy_sha256=policy.digest,
        policy_lock_sha256=str(lock["lock_sha256"]),
        task_definition_sha256=policy.task_definition_sha256,
        effective_task_definition_sha256=task.effective_sha256,
        task_materialization_receipt_sha256=task.sha256,
        dataset_receipt_sha256=dataset.sha256,
        dataset_revision=dataset.revision,
        runtime_sha256=str(runtime["sha256"]),
        runtime_manifest_sha256=str(runtime["manifest_sha256"]),
        runtime_lock_sha256=str(runtime["lock_sha256"]),
        launcher_sha256=sha256_file(
            Path(__file__).with_name("evaluator_sidecar_entry.py")
        ),
        image_repo_digest=str(identity["base_image_repo_digest"]),
        image_id=str(identity["base_image_id"]),
        max_length=policy.max_length,
        max_gen_tokens=policy.max_gen_tokens,
        concurrent_requests=request.evaluator_concurrent_requests,
        timeout_seconds=request.evaluator_timeout_seconds,
    )


def _validate_request(request: MagpieAttestationRequest) -> None:
    integers = (
        request.evaluator_endpoint_port,
        request.evaluator_concurrent_requests,
        request.evaluator_timeout_seconds,
    )
    if (
        request.pass_type is not BenchmarkPass.MEASUREMENT
        or request.execution_mode != "docker"
        or request.lifecycle != "one_shot"
        or sha256_file(request.config_path) != request.config_sha256
        or not isinstance(request.model, str)
        or not request.model
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers
        )
    ):
        raise _invalid("Evaluator execution request is incomplete")


def _policy(value: object) -> EvaluatorPolicy:
    policy = _mapping(value, "evaluator policy")
    fields = {
        key: policy.get(key)
        for key in (
            "policy_id", "tasks", "task_definition_path",
            "task_definition_sha256", "dataset_path", "dataset_name",
            "dataset_revision", "primary_metric", "max_length",
            "max_gen_tokens",
        )
    }
    try:
        typed = EvaluatorPolicy(**fields)  # type: ignore[arg-type]
    except (ConfigurationError, TypeError, ValueError) as error:
        raise _invalid("Evaluator policy is invalid") from error
    if typed.to_dict() != dict(policy):
        raise _invalid("Evaluator policy digest differs")
    return typed


def _validate_policy_lock(
    policy: EvaluatorPolicy,
    lock: Mapping[str, Any],
    task: EvaluatorTaskMaterializationReceipt,
    dataset: EvaluatorDatasetReceipt,
) -> None:
    task_lock = _mapping(lock.get("task"), "evaluator task lock")
    dataset_lock = _mapping(lock.get("dataset"), "evaluator dataset lock")
    matches = (
        _digest(lock.get("lock_sha256"))
        and lock.get("schema") == "apex.evaluator-policy-lock/v2"
        and lock.get("policy_id") == policy.policy_id
        and lock.get("primary_metric") == policy.primary_metric
        and lock.get("sample_logging_required") is True
        and task_lock.get("name") == policy.tasks
        and task_lock.get("definition_path") == policy.task_definition_path
        and task_lock.get("definition_sha256") == policy.task_definition_sha256
        and task.source_sha256 == policy.task_definition_sha256
        and task.dataset_revision == policy.dataset_revision
        and task.dataset_receipt_sha256 == dataset.sha256
        and dataset_lock.get("repository") == dataset.repository
        and dataset_lock.get("path") == policy.dataset_path == dataset.path
        and dataset_lock.get("name") == policy.dataset_name == dataset.name
        and dataset_lock.get("revision") == policy.dataset_revision == dataset.revision
        and tuple(dataset_lock.get("splits", ()))
        == tuple(sorted({item.split for item in dataset.files}))
        and dataset_lock.get("files")
        == [item.to_dict() for item in dataset.files]
    )
    if not matches:
        raise _invalid("Evaluator inputs differ from the policy lock")


def _validate_runtime(
    runtime: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    if (
        not all(
            _digest(runtime.get(key))
            for key in ("sha256", "manifest_sha256", "lock_sha256")
        )
        or not _IMAGE_ID.fullmatch(str(identity.get("base_image_id", "")))
        or not _REPO_DIGEST.fullmatch(
            str(identity.get("base_image_repo_digest", ""))
        )
    ):
        raise _invalid("lm-eval runtime identity is invalid")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid(f"{label} is missing")
    return value


def _digest(value: object) -> bool:
    return isinstance(value, str) and bool(_DIGEST.fullmatch(value))


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_execution_contract_invalid")


__all__ = ["build_lm_eval_execution_contract"]
