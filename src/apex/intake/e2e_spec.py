"""Kernel-only end-to-end optimization request contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from apex.core import AgentBackendName, ContractError


_DATASET_SPLITS = {"train", "validation", "heldout"}
_DATA_VISIBILITIES = {"public", "private", "heldout_private"}


@dataclass(frozen=True, slots=True)
class RegressionGates:
    """Frozen workload quality and tail-latency constraints."""

    accuracy_regression_pct: float = 0.0
    ttft_p99_regression_pct: float = 5.0
    tpot_p99_regression_pct: float = 2.0

    def __post_init__(self) -> None:
        if self.accuracy_regression_pct != 0:
            raise ContractError("Accuracy regression is not permitted", "accuracy_regression_forbidden")
        if (
            not math.isfinite(self.ttft_p99_regression_pct)
            or not math.isfinite(self.tpot_p99_regression_pct)
            or not 0 <= self.ttft_p99_regression_pct <= 5
            or not 0 <= self.tpot_p99_regression_pct <= 2
        ):
            raise ContractError(
                "Tail-latency regression gates must be finite and no weaker than 5% TTFT / 2% TPOT",
                "invalid_regression_gate",
            )


@dataclass(frozen=True, slots=True)
class MetricGoal:
    """Primary metric direction and frozen regression gates."""

    primary: str = "throughput"
    direction: str = "maximize"
    gates: RegressionGates = field(default_factory=RegressionGates)

    def __post_init__(self) -> None:
        if self.primary != "throughput" or self.direction != "maximize":
            raise ContractError(
                "E2E v1 supports only primary=throughput with direction=maximize",
                "unsupported_metric_goal",
            )


@dataclass(frozen=True, slots=True)
class E2EOptimizeSpec:
    """Input for the kernel-only workload optimization state machine."""

    schema_version: int
    config_path: Path
    results_dir: Path
    agent_backend: AgentBackendName = AgentBackendName.CODEX
    agent_model: str | None = None
    agent_effort: str | None = None
    scope: str = "kernels"
    gpu_arch: str = "gfx950"
    goal: MetricGoal = field(default_factory=MetricGoal)
    deployment_hints: Mapping[str, Any] = field(default_factory=dict)
    max_iterations: int = 3
    max_kernels: int = 10
    max_turns: int = 25
    agent_timeout_seconds: int = 3600
    context_input_tokens: int = 16_000
    context_response_token_allocation: int = 8_000
    dataset_split: str = "train"
    data_visibility: str = "public"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractError("Unsupported E2EOptimizeSpec schema_version", "unsupported_schema")
        if not self.config_path.is_absolute() or not self.results_dir.is_absolute():
            raise ContractError("config_path and results_dir must be absolute", "path_not_absolute")
        if self.scope != "kernels":
            raise ContractError("E2E scope is fixed to kernels", "unsupported_e2e_scope")
        if self.agent_model is not None and not self.agent_model.strip():
            raise ContractError("Agent model may not be empty", "invalid_agent_options")
        if self.agent_effort is not None and not self.agent_effort.strip():
            raise ContractError("Agent effort may not be empty", "invalid_agent_options")
        if min(
            self.max_iterations,
            self.max_kernels,
            self.max_turns,
            self.agent_timeout_seconds,
            self.context_input_tokens,
            self.context_response_token_allocation,
        ) <= 0:
            raise ContractError("E2E budgets must be positive", "invalid_budget")
        if self.dataset_split not in _DATASET_SPLITS:
            raise ContractError("Invalid dataset split", "invalid_dataset_split")
        if self.data_visibility not in _DATA_VISIBILITIES:
            raise ContractError("Invalid data visibility", "invalid_data_visibility")
        if self.data_visibility == "heldout_private" and self.dataset_split != "heldout":
            raise ContractError(
                "heldout_private visibility requires the heldout split",
                "invalid_data_partition",
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "E2EOptimizeSpec":
        if "context_output_tokens" in data:
            raise ContractError(
                "context_output_tokens implied an unenforced execution limit; use "
                "context_response_token_allocation",
                "superseded_context_budget_field",
            )
        goal_data = dict(data.get("goal", {}))
        gates_data = dict(goal_data.get("gates", {}))
        goal = MetricGoal(
            primary=str(goal_data.get("primary", "throughput")),
            direction=str(goal_data.get("direction", "maximize")),
            gates=RegressionGates(
                accuracy_regression_pct=float(gates_data.get("accuracy_regression_pct", 0)),
                ttft_p99_regression_pct=float(gates_data.get("ttft_p99_regression_pct", 5)),
                tpot_p99_regression_pct=float(gates_data.get("tpot_p99_regression_pct", 2)),
            ),
        )
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            config_path=Path(str(data.get("config_path", ""))),
            results_dir=Path(str(data.get("results_dir", ""))),
            agent_backend=AgentBackendName(str(data.get("agent_backend", "codex"))),
            agent_model=str(data["agent_model"]) if data.get("agent_model") else None,
            agent_effort=str(data["agent_effort"]) if data.get("agent_effort") else None,
            scope=str(data.get("scope", "kernels")),
            gpu_arch=str(data.get("gpu_arch", "gfx950")),
            goal=goal,
            deployment_hints=dict(data.get("deployment_hints", {})),
            max_iterations=int(data.get("max_iterations", 3)),
            max_kernels=int(data.get("max_kernels", 10)),
            max_turns=int(data.get("max_turns", 25)),
            agent_timeout_seconds=int(data.get("agent_timeout_seconds", 3600)),
            context_input_tokens=int(data.get("context_input_tokens", 16_000)),
            context_response_token_allocation=int(
                data.get("context_response_token_allocation", 8_000)
            ),
            dataset_split=str(data.get("dataset_split", "train")),
            data_visibility=str(data.get("data_visibility", "public")),
        )

    @classmethod
    def from_file(cls, path: Path) -> "E2EOptimizeSpec":
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(content)
        if not isinstance(data, Mapping):
            raise ContractError("E2E spec document must contain an object", "invalid_e2e_spec")
        return cls.from_mapping(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_path": str(self.config_path),
            "results_dir": str(self.results_dir),
            "agent_backend": self.agent_backend.value,
            "agent_model": self.agent_model,
            "agent_effort": self.agent_effort,
            "scope": self.scope,
            "gpu_arch": self.gpu_arch,
            "goal": {
                "primary": self.goal.primary,
                "direction": self.goal.direction,
                "gates": {
                    "accuracy_regression_pct": self.goal.gates.accuracy_regression_pct,
                    "ttft_p99_regression_pct": self.goal.gates.ttft_p99_regression_pct,
                    "tpot_p99_regression_pct": self.goal.gates.tpot_p99_regression_pct,
                },
            },
            "deployment_hints": dict(self.deployment_hints),
            "max_iterations": self.max_iterations,
            "max_kernels": self.max_kernels,
            "max_turns": self.max_turns,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "context_input_tokens": self.context_input_tokens,
            "context_response_token_allocation": self.context_response_token_allocation,
            "dataset_split": self.dataset_split,
            "data_visibility": self.data_visibility,
        }
