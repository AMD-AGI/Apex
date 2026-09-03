"""Typed Apex evaluator policy consumed from the frozen scoring view."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from apex.core import ConfigurationError, sha256_json


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class EvaluatorPolicy:
    """One evaluator request contract, separate from server admission."""

    policy_id: str
    tasks: str
    task_definition_path: str
    task_definition_sha256: str
    dataset_path: str
    dataset_name: str
    dataset_revision: str
    primary_metric: str
    max_length: int
    max_gen_tokens: int

    def __post_init__(self) -> None:
        if (
            not self.policy_id
            or not self.tasks
            or not self.task_definition_path
            or not _SHA256.fullmatch(self.task_definition_sha256)
            or not self.dataset_path
            or not self.dataset_name
            or not _GIT.fullmatch(self.dataset_revision)
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
        return sha256_json({"schema": "apex.evaluator-policy/v2", **asdict(self)})

    def env(self) -> dict[str, str]:
        return {
            "MAGPIE_EVAL_POLICY_ID": self.policy_id,
            "MAGPIE_EVAL_TASKS": self.tasks,
            "MAGPIE_EVAL_TASK_DEFINITION_PATH": self.task_definition_path,
            "MAGPIE_EVAL_TASK_DEFINITION_SHA256": self.task_definition_sha256,
            "MAGPIE_EVAL_DATASET_PATH": self.dataset_path,
            "MAGPIE_EVAL_DATASET_NAME": self.dataset_name,
            "MAGPIE_EVAL_DATASET_REVISION": self.dataset_revision,
            "MAGPIE_EVAL_PRIMARY_METRIC": self.primary_metric,
            "MAGPIE_EVAL_MAX_LENGTH": str(self.max_length),
            "MAGPIE_EVAL_MAX_GEN_TOKENS": str(self.max_gen_tokens),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sha256": self.digest}


def evaluator_policy_from_scoring(
    benchmark: Mapping[str, Any],
) -> EvaluatorPolicy | None:
    """Parse explicit Apex-owned fields without model or config routing."""

    envs = benchmark.get("envs")
    if not isinstance(envs, Mapping) or "RUN_EVAL" not in envs:
        return None
    fields = {
        "policy_id": envs.get("MAGPIE_EVAL_POLICY_ID"),
        "tasks": envs.get("MAGPIE_EVAL_TASKS"),
        "task_definition_path": envs.get("MAGPIE_EVAL_TASK_DEFINITION_PATH"),
        "task_definition_sha256": envs.get("MAGPIE_EVAL_TASK_DEFINITION_SHA256"),
        "dataset_path": envs.get("MAGPIE_EVAL_DATASET_PATH"),
        "dataset_name": envs.get("MAGPIE_EVAL_DATASET_NAME"),
        "dataset_revision": envs.get("MAGPIE_EVAL_DATASET_REVISION"),
        "primary_metric": envs.get("MAGPIE_EVAL_PRIMARY_METRIC"),
        "max_length": envs.get("MAGPIE_EVAL_MAX_LENGTH"),
        "max_gen_tokens": envs.get("MAGPIE_EVAL_MAX_GEN_TOKENS"),
    }
    try:
        return EvaluatorPolicy(
            policy_id=str(fields["policy_id"] or ""),
            tasks=str(fields["tasks"] or ""),
            task_definition_path=str(fields["task_definition_path"] or ""),
            task_definition_sha256=str(fields["task_definition_sha256"] or ""),
            dataset_path=str(fields["dataset_path"] or ""),
            dataset_name=str(fields["dataset_name"] or ""),
            dataset_revision=str(fields["dataset_revision"] or ""),
            primary_metric=str(fields["primary_metric"] or ""),
            max_length=_strict_int(fields["max_length"]),
            max_gen_tokens=_strict_int(fields["max_gen_tokens"]),
        )
    except (ConfigurationError, TypeError, ValueError) as error:
        raise ConfigurationError(
            "Apex scoring view lacks an explicit evaluator policy",
            "invalid_evaluator_policy",
        ) from error


def _strict_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("evaluator integer field is invalid")
    text = str(value)
    if not text.isdigit() or text.startswith("0"):
        raise ValueError("evaluator integer field is invalid")
    return int(text)


__all__ = [
    "EvaluatorPolicy",
    "evaluator_policy_from_scoring",
]
