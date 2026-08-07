"""Typed evaluator-only policy derived from an immutable workload identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from apex.core import ConfigurationError, sha256_json


QWEN_CONFIG_SHA256 = "f97bda8e04655fbd1410bafb34072ec072de416ea7e24551d2618281e75deafb"
QWEN_MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"


@dataclass(frozen=True, slots=True)
class EvaluatorPolicy:
    """One evaluator request contract, separate from server admission."""

    policy_id: str
    tasks: str
    primary_metric: str
    max_length: int
    max_gen_tokens: int

    def __post_init__(self) -> None:
        if (
            not self.policy_id
            or not self.tasks
            or not self.primary_metric
            or self.max_length < 1
            or self.max_gen_tokens < 1
            or self.max_gen_tokens >= self.max_length
        ):
            raise ConfigurationError(
                "Evaluator policy is incomplete", "invalid_evaluator_policy"
            )

    @property
    def digest(self) -> str:
        return sha256_json({"schema": "apex.evaluator-policy/v1", **asdict(self)})

    def env(self) -> dict[str, str]:
        return {
            "MAGPIE_EVAL_POLICY_ID": self.policy_id,
            "MAGPIE_EVAL_TASKS": self.tasks,
            "MAGPIE_EVAL_PRIMARY_METRIC": self.primary_metric,
            "MAGPIE_EVAL_MAX_LENGTH": str(self.max_length),
            "MAGPIE_EVAL_MAX_GEN_TOKENS": str(self.max_gen_tokens),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sha256": self.digest}


def qwen_evaluator_policy() -> EvaluatorPolicy:
    """Construct the reviewed policy without import-time object creation."""

    return EvaluatorPolicy(
        policy_id="qwen3-next-80b-gsm8k-strict-v1",
        tasks="gsm8k",
        primary_metric="exact_match,strict-match",
        max_length=2248,
        max_gen_tokens=480,
    )


def resolve_evaluator_policy(
    original_sha256: str, benchmark: Mapping[str, Any]
) -> EvaluatorPolicy | None:
    """Resolve only reviewed identities; never infer an output budget."""

    if original_sha256 != QWEN_CONFIG_SHA256:
        return None
    envs = benchmark.get("envs")
    valid = (
        benchmark.get("framework") == "vllm"
        and benchmark.get("model") == QWEN_MODEL_ID
        and isinstance(envs, Mapping)
        and int(envs.get("MAX_MODEL_LEN", 0)) == 2248
        and str(envs.get("MAGPIE_EVAL_TASKS", "gsm8k")) == "gsm8k"
    )
    if not valid:
        raise ConfigurationError(
            "Reviewed Qwen evaluator policy does not match workload fields",
            "qwen_evaluator_policy_mismatch",
        )
    return qwen_evaluator_policy()


__all__ = [
    "EvaluatorPolicy",
    "QWEN_CONFIG_SHA256",
    "QWEN_MODEL_ID",
    "qwen_evaluator_policy",
    "resolve_evaluator_policy",
]
