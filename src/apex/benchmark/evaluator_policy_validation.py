"""Validation helpers for evaluator policy bindings in benchmark views."""

from __future__ import annotations

from typing import Any, Mapping

from apex.core import ConfigurationError

from .evaluator_policy import EvaluatorPolicy

_POLICY_FIELDS = frozenset(
    (
        "policy_id",
        "tasks",
        "task_definition_path",
        "task_definition_sha256",
        "dataset_path",
        "dataset_name",
        "dataset_revision",
        "primary_metric",
        "max_length",
        "max_gen_tokens",
        "sha256",
    )
)


def validate_evaluator_policy(
    benchmark: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Require the typed policy, its digest, and its resolved environment."""

    quality = metadata.get("quality_contract")
    policy = quality.get("evaluator_policy") if isinstance(quality, Mapping) else None
    if policy is None:
        if isinstance(quality, Mapping) and quality.get("kind") in {
            "lm_eval",
            "trace_only",
        }:
            raise _mismatch("lm-eval benchmark view lacks its evaluator policy")
        return
    if not isinstance(policy, Mapping) or set(policy) != _POLICY_FIELDS:
        raise _mismatch("Benchmark evaluator policy metadata is incomplete")
    typed = _typed_policy(policy)
    if dict(policy) != typed.to_dict():
        raise _mismatch("Benchmark evaluator policy digest is invalid")
    _validate_policy_environment(benchmark, policy)


def _typed_policy(policy: Mapping[str, Any]) -> EvaluatorPolicy:
    try:
        return EvaluatorPolicy(
            policy_id=policy["policy_id"],
            tasks=policy["tasks"],
            task_definition_path=policy["task_definition_path"],
            task_definition_sha256=policy["task_definition_sha256"],
            dataset_path=policy["dataset_path"],
            dataset_name=policy["dataset_name"],
            dataset_revision=policy["dataset_revision"],
            primary_metric=policy["primary_metric"],
            max_length=policy["max_length"],
            max_gen_tokens=policy["max_gen_tokens"],
        )
    except (ConfigurationError, KeyError, TypeError, ValueError) as error:
        raise _mismatch("Benchmark evaluator policy metadata is invalid") from error


def _validate_policy_environment(
    benchmark: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    expected = {
        "MAGPIE_EVAL_POLICY_ID": policy.get("policy_id"),
        "MAGPIE_EVAL_TASKS": policy.get("tasks"),
        "MAGPIE_EVAL_TASK_DEFINITION_PATH": policy.get("task_definition_path"),
        "MAGPIE_EVAL_TASK_DEFINITION_SHA256": policy.get(
            "task_definition_sha256"
        ),
        "MAGPIE_EVAL_DATASET_PATH": policy.get("dataset_path"),
        "MAGPIE_EVAL_DATASET_NAME": policy.get("dataset_name"),
        "MAGPIE_EVAL_DATASET_REVISION": policy.get("dataset_revision"),
        "MAGPIE_EVAL_PRIMARY_METRIC": policy.get("primary_metric"),
        "MAGPIE_EVAL_MAX_LENGTH": str(policy.get("max_length")),
        "MAGPIE_EVAL_MAX_GEN_TOKENS": str(policy.get("max_gen_tokens")),
    }
    envs = benchmark.get("envs")
    if not isinstance(envs, Mapping) or any(
        str(envs.get(name)) != str(value) for name, value in expected.items()
    ):
        raise _mismatch(
            "Benchmark evaluator policy differs from its resolved receipt"
        )


def _mismatch(message: str) -> ConfigurationError:
    return ConfigurationError(message, "benchmark_evaluator_policy_mismatch")


__all__ = ["validate_evaluator_policy"]
