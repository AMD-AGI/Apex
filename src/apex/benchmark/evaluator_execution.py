"""Typed authority contract and receipt for exact-image lm-eval execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from apex.core import ConfigurationError, sha256_json, validate_identifier

from .evaluator_artifact_receipt import EvaluatorArtifactReceipt


EXECUTION_CONTRACT_SCHEMA = "apex.lm-eval-execution-contract/v2"
EXECUTION_RECEIPT_SCHEMA = "apex.lm-eval-execution-receipt/v3"
EXECUTION_AUTHORITY = "apex_exact_image_lm_eval_sidecar"
EXECUTION_MODE = "exact_image_sidecar"
SIDECAR_PROXY_PORT = 18080
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class LmEvalExecutionContract:
    """Fixed sidecar invocation frozen before Magpie starts."""

    run_id: str
    config_sha256: str
    model: str
    endpoint_port: int
    policy_sha256: str
    policy_lock_sha256: str
    task_definition_sha256: str
    effective_task_definition_sha256: str
    task_materialization_receipt_sha256: str
    dataset_receipt_sha256: str
    dataset_revision: str
    runtime_sha256: str
    runtime_manifest_sha256: str
    runtime_lock_sha256: str
    launcher_sha256: str
    image_repo_digest: str
    image_id: str
    max_length: int
    max_gen_tokens: int
    concurrent_requests: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field_name="evaluator run_id")
        integers = (
            self.endpoint_port,
            self.max_length,
            self.max_gen_tokens,
            self.concurrent_requests,
            self.timeout_seconds,
        )
        if (
            not self.model.strip()
            or any(character in self.model for character in "\r\n\0")
            or any(isinstance(value, bool) or not isinstance(value, int) for value in integers)
            or not 1 <= self.endpoint_port <= 65535
            or self.max_length <= 1
            or not 0 < self.max_gen_tokens < self.max_length
            or not 0 < self.concurrent_requests <= 4096
            or not 0 < self.timeout_seconds <= 24 * 60 * 60
            or not all(
                _digest(value)
                for value in (
                    self.config_sha256,
                    self.policy_sha256,
                    self.policy_lock_sha256,
                    self.task_definition_sha256,
                    self.effective_task_definition_sha256,
                    self.task_materialization_receipt_sha256,
                    self.dataset_receipt_sha256,
                    self.runtime_sha256,
                    self.runtime_manifest_sha256,
                    self.runtime_lock_sha256,
                    self.launcher_sha256,
                )
            )
            or not _COMMIT.fullmatch(self.dataset_revision)
            or not _REPO_DIGEST.fullmatch(self.image_repo_digest)
            or not _IMAGE_ID.fullmatch(self.image_id)
        ):
            raise ValueError("lm-eval execution contract is invalid")

    @property
    def argv(self) -> tuple[str, ...]:
        model_args = (
            f"model={self.model},base_url=http://127.0.0.1:{SIDECAR_PROXY_PORT}"
            "/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,"
            f"num_concurrent={self.concurrent_requests},timeout=1800,"
            f"tokenized_requests=False,max_length={self.max_length}"
        )
        generation = (
            f"max_tokens={self.max_gen_tokens},temperature=0,top_p=1"
        )
        return (
            "python3",
            "-m",
            "lm_eval",
            "--model",
            "local-chat-completions",
            "--apply_chat_template",
            "--tasks",
            "/evaluator/task/gsm8k.yaml",
            "--output_path",
            "/authority/output",
            "--log_samples",
            "--model_args",
            model_args,
            "--gen_kwargs",
            generation,
        )

    @property
    def environment(self) -> Mapping[str, str]:
        return {
            "HF_HOME": "/tmp/huggingface",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "OPENAI_API_KEY": "EMPTY",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/evaluator/runtime/site-packages",
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": "/tmp/xdg",
        }

    @property
    def sidecar_connection_limit(self) -> int:
        return min(512, max(16, self.concurrent_requests * 2))

    @property
    def sidecar_argv(self) -> tuple[str, ...]:
        return (
            "python3",
            "/evaluator/launcher/evaluator_sidecar_entry.py",
            "--unix-socket",
            "/evaluator/broker/serving.sock",
            "--proxy-port",
            str(SIDECAR_PROXY_PORT),
            "--max-connections",
            str(self.sidecar_connection_limit),
            "--runtime-probe",
            "/authority/runtime_probe.json",
            "--",
            *self.argv,
        )

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": EXECUTION_CONTRACT_SCHEMA,
            "authority": EXECUTION_AUTHORITY,
            "execution_mode": EXECUTION_MODE,
            "run": {
                "run_id": self.run_id,
                "pass_type": "measurement",
                "config_sha256": self.config_sha256,
            },
            "policy": {
                "sha256": self.policy_sha256,
                "lock_sha256": self.policy_lock_sha256,
            },
            "task": {
                "source_definition_sha256": self.task_definition_sha256,
                "effective_definition_sha256": self.effective_task_definition_sha256,
                "materialization_receipt_sha256": self.task_materialization_receipt_sha256,
            },
            "dataset": {
                "revision": self.dataset_revision,
                "receipt_sha256": self.dataset_receipt_sha256,
            },
            "runtime": {
                "sha256": self.runtime_sha256,
                "manifest_sha256": self.runtime_manifest_sha256,
                "lock_sha256": self.runtime_lock_sha256,
                "launcher_sha256": self.launcher_sha256,
            },
            "image": {
                "repo_digest": self.image_repo_digest,
                "image_id": self.image_id,
            },
            "command": {
                "evaluator_argv": list(self.argv),
                "evaluator_argv_sha256": sha256_json(list(self.argv)),
                "container_argv": list(self.sidecar_argv),
                "container_argv_sha256": sha256_json(list(self.sidecar_argv)),
                "cwd": "/authority",
                "environment": dict(self.environment),
                "environment_sha256": sha256_json(dict(self.environment)),
                "timeout_seconds": self.timeout_seconds,
            },
            "outputs": {
                "root": "/authority/output",
                "sample_logging_required": True,
                "max_files": 256,
                "max_total_bytes": 128 * 1024 * 1024,
            },
            "security": {
                "network_mode": "none",
                "read_only_root": True,
                "gpu_devices": [],
                "cap_drop": ["ALL"],
                "no_new_privileges": True,
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class LmEvalExecutionReceipt:
    """Authority-observed sidecar result bound to exact output artifacts."""

    contract_sha256: str
    config_sha256: str
    policy_sha256: str
    policy_lock_sha256: str
    task_definition_sha256: str
    effective_task_definition_sha256: str
    task_materialization_receipt_sha256: str
    dataset_receipt_sha256: str
    dataset_revision: str
    runtime_sha256: str
    runtime_manifest_sha256: str
    runtime_lock_sha256: str
    launcher_sha256: str
    image_repo_digest: str
    image_id: str
    container_id: str
    listener_receipt_sha256: str
    sidecar_spec_sha256: str
    created_observation_sha256: str
    exited_observation_sha256: str
    broker_receipt_sha256: str
    container_cleanup_sha256: str
    runtime_probe_sha256: str
    runtime_publication_sha256: str
    result_artifacts: tuple[EvaluatorArtifactReceipt, ...]
    sample_artifacts: tuple[EvaluatorArtifactReceipt, ...]

    def __post_init__(self) -> None:
        digests = (
            self.contract_sha256,
            self.config_sha256,
            self.policy_sha256,
            self.policy_lock_sha256,
            self.task_definition_sha256,
            self.effective_task_definition_sha256,
            self.task_materialization_receipt_sha256,
            self.dataset_receipt_sha256,
            self.runtime_sha256,
            self.runtime_manifest_sha256,
            self.runtime_lock_sha256,
            self.launcher_sha256,
            self.container_id,
            self.listener_receipt_sha256,
            self.sidecar_spec_sha256,
            self.created_observation_sha256,
            self.exited_observation_sha256,
            self.broker_receipt_sha256,
            self.container_cleanup_sha256,
            self.runtime_probe_sha256,
            self.runtime_publication_sha256,
        )
        artifacts = (*self.result_artifacts, *self.sample_artifacts)
        if (
            not all(_digest(value) for value in digests)
            or not _REPO_DIGEST.fullmatch(self.image_repo_digest)
            or not _IMAGE_ID.fullmatch(self.image_id)
            or not _COMMIT.fullmatch(self.dataset_revision)
            or not self.result_artifacts
            or not self.sample_artifacts
            or len({item.path for item in artifacts}) != len(artifacts)
        ):
            raise ValueError("lm-eval execution receipt is invalid")

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": EXECUTION_RECEIPT_SCHEMA,
            "authority": EXECUTION_AUTHORITY,
            "execution_mode": EXECUTION_MODE,
            "status": "succeeded",
            "verified": True,
            "contract_sha256": self.contract_sha256,
            "config_sha256": self.config_sha256,
            "policy_sha256": self.policy_sha256,
            "policy_lock_sha256": self.policy_lock_sha256,
            "task_definition_sha256": self.task_definition_sha256,
            "effective_task_definition_sha256": self.effective_task_definition_sha256,
            "task_materialization_receipt_sha256": self.task_materialization_receipt_sha256,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "dataset_revision": self.dataset_revision,
            "runtime_sha256": self.runtime_sha256,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "launcher_sha256": self.launcher_sha256,
            "image_repo_digest": self.image_repo_digest,
            "image_id": self.image_id,
            "container_id": self.container_id,
            "listener_receipt_sha256": self.listener_receipt_sha256,
            "sidecar_spec_sha256": self.sidecar_spec_sha256,
            "created_observation_sha256": self.created_observation_sha256,
            "exited_observation_sha256": self.exited_observation_sha256,
            "broker_receipt_sha256": self.broker_receipt_sha256,
            "container_cleanup_sha256": self.container_cleanup_sha256,
            "runtime_probe_sha256": self.runtime_probe_sha256,
            "runtime_publication_sha256": self.runtime_publication_sha256,
            "result_artifacts": [item.to_dict() for item in self.result_artifacts],
            "sample_artifacts": [item.to_dict() for item in self.sample_artifacts],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LmEvalExecutionReceipt:
        fields = {
            "schema", "authority", "execution_mode", "status", "verified",
            "contract_sha256", "config_sha256", "policy_sha256",
            "policy_lock_sha256", "task_definition_sha256",
            "effective_task_definition_sha256",
            "task_materialization_receipt_sha256", "dataset_receipt_sha256",
            "dataset_revision", "runtime_sha256", "runtime_manifest_sha256",
            "runtime_lock_sha256", "launcher_sha256", "image_repo_digest",
            "image_id", "container_id", "listener_receipt_sha256",
            "sidecar_spec_sha256", "created_observation_sha256",
            "exited_observation_sha256", "broker_receipt_sha256",
            "container_cleanup_sha256",
            "runtime_probe_sha256", "runtime_publication_sha256",
            "result_artifacts", "sample_artifacts",
            "receipt_sha256",
        }
        if (
            set(value) != fields
            or value.get("schema") != EXECUTION_RECEIPT_SCHEMA
            or value.get("authority") != EXECUTION_AUTHORITY
            or value.get("execution_mode") != EXECUTION_MODE
            or value.get("status") != "succeeded"
            or value.get("verified") is not True
        ):
            raise ValueError("lm-eval execution receipt envelope is invalid")
        results = _artifact_sequence(value.get("result_artifacts"))
        samples = _artifact_sequence(value.get("sample_artifacts"))
        receipt = cls(
            contract_sha256=str(value["contract_sha256"]),
            config_sha256=str(value["config_sha256"]),
            policy_sha256=str(value["policy_sha256"]),
            policy_lock_sha256=str(value["policy_lock_sha256"]),
            task_definition_sha256=str(value["task_definition_sha256"]),
            effective_task_definition_sha256=str(
                value["effective_task_definition_sha256"]
            ),
            task_materialization_receipt_sha256=str(
                value["task_materialization_receipt_sha256"]
            ),
            dataset_receipt_sha256=str(value["dataset_receipt_sha256"]),
            dataset_revision=str(value["dataset_revision"]),
            runtime_sha256=str(value["runtime_sha256"]),
            runtime_manifest_sha256=str(value["runtime_manifest_sha256"]),
            runtime_lock_sha256=str(value["runtime_lock_sha256"]),
            launcher_sha256=str(value["launcher_sha256"]),
            image_repo_digest=str(value["image_repo_digest"]),
            image_id=str(value["image_id"]),
            container_id=str(value["container_id"]),
            listener_receipt_sha256=str(value["listener_receipt_sha256"]),
            sidecar_spec_sha256=str(value["sidecar_spec_sha256"]),
            created_observation_sha256=str(value["created_observation_sha256"]),
            exited_observation_sha256=str(value["exited_observation_sha256"]),
            broker_receipt_sha256=str(value["broker_receipt_sha256"]),
            container_cleanup_sha256=str(value["container_cleanup_sha256"]),
            runtime_probe_sha256=str(value["runtime_probe_sha256"]),
            runtime_publication_sha256=str(value["runtime_publication_sha256"]),
            result_artifacts=results,
            sample_artifacts=samples,
        )
        if value.get("receipt_sha256") != receipt.sha256:
            raise ValueError("lm-eval execution receipt digest is invalid")
        return receipt


def validate_receipt_against_contract(
    receipt: LmEvalExecutionReceipt, contract: LmEvalExecutionContract
) -> str | None:
    """Reject a sidecar receipt swapped across any frozen evaluator input."""

    checks = (
        (receipt.contract_sha256, contract.sha256, "evaluator_contract_mismatch"),
        (receipt.config_sha256, contract.config_sha256, "evaluator_config_mismatch"),
        (receipt.policy_sha256, contract.policy_sha256, "evaluator_policy_mismatch"),
        (receipt.policy_lock_sha256, contract.policy_lock_sha256, "evaluator_policy_lock_mismatch"),
        (receipt.task_definition_sha256, contract.task_definition_sha256, "evaluator_task_mismatch"),
        (receipt.effective_task_definition_sha256, contract.effective_task_definition_sha256, "evaluator_effective_task_mismatch"),
        (receipt.task_materialization_receipt_sha256, contract.task_materialization_receipt_sha256, "evaluator_task_materialization_mismatch"),
        (receipt.dataset_receipt_sha256, contract.dataset_receipt_sha256, "evaluator_dataset_mismatch"),
        (receipt.dataset_revision, contract.dataset_revision, "evaluator_dataset_revision_mismatch"),
        (receipt.runtime_sha256, contract.runtime_sha256, "evaluator_runtime_mismatch"),
        (receipt.runtime_manifest_sha256, contract.runtime_manifest_sha256, "evaluator_runtime_manifest_mismatch"),
        (receipt.runtime_lock_sha256, contract.runtime_lock_sha256, "evaluator_runtime_lock_mismatch"),
        (receipt.launcher_sha256, contract.launcher_sha256, "evaluator_launcher_mismatch"),
        (receipt.image_repo_digest, contract.image_repo_digest, "evaluator_image_digest_mismatch"),
        (receipt.image_id, contract.image_id, "evaluator_image_id_mismatch"),
    )
    return next((error for observed, expected, error in checks if observed != expected), None)


def validate_execution_binding(
    value: object,
    *,
    expected_policy: Mapping[str, Any],
    expected_runtime_sha256: str | None,
    expected_image_repo_digest: str | None,
    result_artifacts: tuple[Mapping[str, Any], ...],
    sample_artifacts: tuple[Mapping[str, Any], ...],
) -> str | None:
    """Bind one authority receipt to frozen policy/runtime and parsed outputs."""

    if not isinstance(value, Mapping):
        return "quality_evaluator_execution_receipt_missing"
    try:
        receipt = LmEvalExecutionReceipt.from_mapping(value)
    except (KeyError, TypeError, ValueError):
        return "quality_evaluator_execution_receipt_invalid"
    expected_policy_sha = expected_policy.get("sha256")
    checks = (
        (receipt.policy_sha256, expected_policy_sha, "quality_evaluator_policy_mismatch"),
        (
            receipt.task_definition_sha256,
            expected_policy.get("task_definition_sha256"),
            "quality_evaluator_task_definition_mismatch",
        ),
        (
            receipt.runtime_sha256,
            expected_runtime_sha256,
            "quality_evaluator_runtime_mismatch",
        ),
        (
            receipt.image_repo_digest,
            expected_image_repo_digest,
            "quality_evaluator_image_mismatch",
        ),
    )
    for observed, expected, error in checks:
        if expected is not None and observed != expected:
            return error
    if receipt.policy_sha256 != expected_policy_sha:
        return "quality_evaluator_policy_mismatch"
    if [item.to_dict() for item in receipt.result_artifacts] != list(result_artifacts):
        return "quality_evaluator_result_receipt_mismatch"
    if [item.to_dict() for item in receipt.sample_artifacts] != list(sample_artifacts):
        return "quality_evaluator_sample_receipt_mismatch"
    return None


def _artifact_sequence(value: object) -> tuple[EvaluatorArtifactReceipt, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("Evaluator artifact sequence is invalid")
    return tuple(
        EvaluatorArtifactReceipt.from_mapping(item)
        if isinstance(item, Mapping)
        else _invalid_artifact()
        for item in value
    )


def _invalid_artifact() -> EvaluatorArtifactReceipt:
    raise ValueError("Evaluator artifact declaration is invalid")


def _digest(value: object) -> bool:
    return isinstance(value, str) and bool(_DIGEST.fullmatch(value))


def require_execution_binding(error: str | None) -> None:
    """Raise a stable configuration error for contract construction callers."""

    if error:
        raise ConfigurationError(error, error)


__all__ = [
    "EXECUTION_AUTHORITY",
    "EXECUTION_CONTRACT_SCHEMA",
    "EXECUTION_MODE",
    "EXECUTION_RECEIPT_SCHEMA",
    "LmEvalExecutionContract",
    "LmEvalExecutionReceipt",
    "require_execution_binding",
    "validate_execution_binding",
    "validate_receipt_against_contract",
]
