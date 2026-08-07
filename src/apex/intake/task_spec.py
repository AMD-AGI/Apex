"""Structured single-kernel task contract used by humans and orchestrators."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml

from apex.core import AgentBackendName, ContractError, validate_identifier


_TRUSTED_RECIPE_PROVENANCE = {"trusted_registry", "external_evaluator"}
_RECOGNIZED_LANGUAGES = {"python", "triton", "hip"}
_DATASET_SPLITS = {"train", "validation", "heldout"}
_DATA_VISIBILITIES = {"public", "private", "heldout_private"}


def _relative_source_path(value: str, *, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ContractError(
            f"{field_name} must be a workspace-relative path without '..': {value!r}",
            "unsafe_source_path",
        )
    if any(part in {"", "."} for part in path.parts):
        raise ContractError(f"Invalid {field_name}: {value!r}", "unsafe_source_path")
    return path.as_posix()


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(f"{field_name} must be a list of strings", "invalid_task_spec")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise ContractError(f"{field_name} contains an empty value", "invalid_task_spec")
    return result


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractError(f"{field_name} must be an object", "invalid_task_spec")
    return value


def _scope_terms(value: Any, *, field_name: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                item.strip().lower()
                for item in _string_tuple(value, field_name=field_name)
            }
        )
    )


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """A subprocess argv contract; shell strings are deliberately unsupported."""

    argv: tuple[str, ...]
    timeout_seconds: int = 600
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv or any(not isinstance(arg, str) or not arg for arg in self.argv):
            raise ContractError("command argv must contain non-empty strings", "invalid_command")
        if self.timeout_seconds <= 0:
            raise ContractError("command timeout must be positive", "invalid_command")
        _relative_source_path(self.cwd, field_name="command cwd") if self.cwd != "." else None
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.env.items()):
            raise ContractError("command env must map strings to strings", "invalid_command")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CommandSpec":
        argv = data.get("argv")
        if isinstance(argv, (str, bytes)):
            raise ContractError("command must use argv, not a shell string", "shell_command_forbidden")
        return cls(
            argv=_string_tuple(argv, field_name="command argv"),
            timeout_seconds=int(data.get("timeout_seconds", 600)),
            cwd=str(data.get("cwd", ".")),
            env={str(k): str(v) for k, v in dict(data.get("env", {})).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "timeout_seconds": self.timeout_seconds,
            "cwd": self.cwd,
            "env": dict(sorted(self.env.items())),
        }


@dataclass(frozen=True, slots=True)
class TaskRecipe:
    """Trusted recipe identity; ``fixed_hip`` is reserved but not executable in V1."""

    kind: str
    recipe_id: str
    sha256: str
    provenance: str

    def __post_init__(self) -> None:
        if self.kind not in {"python_triton", "fixed_hip"}:
            raise ContractError(f"Unsupported recipe kind: {self.kind}", "unsupported_recipe")
        validate_identifier(self.recipe_id, field_name="recipe_id")
        digest = self.sha256.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ContractError("recipe sha256 must be a 64-hex digest", "invalid_recipe_hash")
        if self.provenance not in _TRUSTED_RECIPE_PROVENANCE:
            raise ContractError("recipe provenance is not trusted", "untrusted_recipe")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskRecipe":
        return cls(
            kind=str(data.get("kind", "")),
            recipe_id=str(data.get("recipe_id", "")),
            sha256=str(data.get("sha256", "")),
            provenance=str(data.get("provenance", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "recipe_id": self.recipe_id,
            "sha256": self.sha256,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class DeliverySpec:
    """Delivery is always a bundle; applying it is a trusted CLI action."""

    mode: str = "bundle"

    def __post_init__(self) -> None:
        if self.mode != "bundle":
            raise ContractError("Only bundle delivery is supported", "unsupported_delivery_mode")


@dataclass(frozen=True, slots=True)
class AgentOptions:
    """Backend-neutral model and reasoning controls."""

    model: str | None = None
    effort: str | None = None

    def __post_init__(self) -> None:
        if self.model is not None and not self.model.strip():
            raise ContractError("agent model may not be empty", "invalid_agent_options")
        if self.effort is not None and not self.effort.strip():
            raise ContractError("agent effort may not be empty", "invalid_agent_options")


@dataclass(frozen=True, slots=True)
class TaskBudget:
    """Frozen agent/process budget supplied equally to comparison agents."""

    max_iterations: int = 1
    max_turns: int = 25
    timeout_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.max_iterations <= 0 or self.max_turns <= 0 or self.timeout_seconds <= 0:
            raise ContractError("task budget values must be positive", "invalid_task_budget")


@dataclass(frozen=True, slots=True)
class TaskScope:
    """Trusted applicability facts used for scoped advisory retrieval."""

    dtype: tuple[str, ...] = ()
    regime: tuple[str, ...] = ()
    framework: tuple[str, ...] = ()
    versions: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskScope":
        versions = _mapping(data.get("versions"), field_name="scope.versions")
        return cls(
            dtype=_scope_terms(data.get("dtype"), field_name="scope.dtype"),
            regime=_scope_terms(data.get("regime"), field_name="scope.regime"),
            framework=_scope_terms(data.get("framework"), field_name="scope.framework"),
            versions=tuple(
                sorted((str(key).strip(), str(value).strip()) for key, value in versions.items())
            ),
        )

    def __post_init__(self) -> None:
        values = (*self.dtype, *self.regime, *self.framework)
        if any(not item for item in values):
            raise ContractError("task scope contains an empty term", "invalid_task_scope")
        if any(not key or not value for key, value in self.versions):
            raise ContractError("task scope versions are incomplete", "invalid_task_scope")
        if len(dict(self.versions)) != len(self.versions):
            raise ContractError("task scope versions are duplicated", "invalid_task_scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": list(self.dtype),
            "regime": list(self.regime),
            "framework": list(self.framework),
            "versions": dict(self.versions),
        }


@dataclass(frozen=True, slots=True)
class KernelMeasurementSpec:
    """Trusted location and aggregation policy for raw invocation evidence."""

    report_path: str
    aggregation: str = "equal_case"
    schema: str = "apex.kernel-measurement/v1"
    policy_id: str = "kernel_invocation_nearest_rank_v1"
    sample_unit: str = "kernel_invocation"
    quantile_method: str = "nearest_rank_v1"
    min_valid_samples: int = 300
    min_tail_observations: int = 3
    warmup_samples: int = 20
    keep_srobust_threshold: float = 1.05
    confidence_srobust_floor: float = 1.0
    worst_case_srobust_floor: float = 1.0
    max_cv: float = 0.10
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 1729
    bootstrap_repetitions: int = 1000
    min_bootstrap_units: int = 2

    def __post_init__(self) -> None:
        _relative_source_path(self.report_path, field_name="measurement report path")
        if self.aggregation not in {"equal_case", "workload_weighted"}:
            raise ContractError(
                "Unsupported kernel measurement aggregation",
                "invalid_measurement_contract",
            )
        if self.schema != "apex.kernel-measurement/v1":
            raise ContractError(
                "Unsupported kernel measurement schema",
                "invalid_measurement_contract",
            )
        if (
            self.policy_id != "kernel_invocation_nearest_rank_v1"
            or self.sample_unit != "kernel_invocation"
            or self.quantile_method != "nearest_rank_v1"
        ):
            raise ContractError(
                "Unsupported canonical measurement policy",
                "invalid_measurement_contract",
            )
        if self.min_valid_samples < 300 or self.min_tail_observations < 1:
            raise ContractError(
                "Canonical measurement requires at least 300 valid samples",
                "invalid_measurement_contract",
            )
        if math.ceil(self.min_valid_samples * 0.01) < self.min_tail_observations:
            raise ContractError("Measurement tail minimum is invalid", "invalid_measurement_contract")
        thresholds = (
            self.keep_srobust_threshold,
            self.confidence_srobust_floor,
            self.worst_case_srobust_floor,
            self.max_cv,
        )
        if any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise ContractError("Measurement thresholds are invalid", "invalid_measurement_contract")
        if (
            self.keep_srobust_threshold < 1.05
            or self.confidence_srobust_floor < 1.0
            or self.worst_case_srobust_floor < 1.0
            or self.max_cv > 0.10
        ):
            raise ContractError(
                "Measurement policy cannot weaken canonical promotion gates",
                "invalid_measurement_contract",
            )
        if not 0.95 <= self.bootstrap_confidence_level < 1:
            raise ContractError("Bootstrap confidence is invalid", "invalid_measurement_contract")
        if (
            self.warmup_samples < 0
            or self.bootstrap_seed < 0
            or self.bootstrap_repetitions < 100
            or self.min_bootstrap_units < 2
        ):
            raise ContractError("Bootstrap policy is invalid", "invalid_measurement_contract")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KernelMeasurementSpec":
        return cls(
            report_path=str(data.get("report_path", "")),
            aggregation=str(data.get("aggregation", "equal_case")),
            schema=str(data.get("schema", "apex.kernel-measurement/v1")),
            policy_id=str(data.get("policy_id", "kernel_invocation_nearest_rank_v1")),
            sample_unit=str(data.get("sample_unit", "kernel_invocation")),
            quantile_method=str(data.get("quantile_method", "nearest_rank_v1")),
            min_valid_samples=int(data.get("min_valid_samples", 300)),
            min_tail_observations=int(data.get("min_tail_observations", 3)),
            warmup_samples=int(data.get("warmup_samples", 20)),
            keep_srobust_threshold=float(data.get("keep_srobust_threshold", 1.05)),
            confidence_srobust_floor=float(data.get("confidence_srobust_floor", 1.0)),
            worst_case_srobust_floor=float(data.get("worst_case_srobust_floor", 1.0)),
            max_cv=float(data.get("max_cv", 0.10)),
            bootstrap_confidence_level=float(data.get("bootstrap_confidence_level", 0.95)),
            bootstrap_seed=int(data.get("bootstrap_seed", 1729)),
            bootstrap_repetitions=int(data.get("bootstrap_repetitions", 1000)),
            min_bootstrap_units=int(data.get("min_bootstrap_units", 2)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "report_path": self.report_path,
            "aggregation": self.aggregation,
            "policy_id": self.policy_id,
            "sample_unit": self.sample_unit,
            "quantile_method": self.quantile_method,
            "min_valid_samples": self.min_valid_samples,
            "min_tail_observations": self.min_tail_observations,
            "warmup_samples": self.warmup_samples,
            "keep_srobust_threshold": self.keep_srobust_threshold,
            "confidence_srobust_floor": self.confidence_srobust_floor,
            "worst_case_srobust_floor": self.worst_case_srobust_floor,
            "max_cv": self.max_cv,
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "min_bootstrap_units": self.min_bootstrap_units,
        }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Validated caller-neutral input for optimizing an existing kernel."""

    schema_version: int
    task_id: str
    workspace: Path
    results_dir: Path
    instructions: str
    language: str
    editable_files: tuple[str, ...]
    target_functions: tuple[str, ...]
    commands: Mapping[str, CommandSpec]
    gpu_arch: str = "gfx950"
    mode: str = "optimize_existing"
    agent_backend: AgentBackendName = AgentBackendName.CODEX
    agent_options: AgentOptions = field(default_factory=AgentOptions)
    budget: TaskBudget = field(default_factory=TaskBudget)
    scope: TaskScope = field(default_factory=TaskScope)
    measurement: KernelMeasurementSpec | None = None
    recipe: TaskRecipe | None = None
    delivery: DeliverySpec = field(default_factory=DeliverySpec)
    dataset_split: str = "train"
    data_visibility: str = "public"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractError("Unsupported TaskSpec schema_version", "unsupported_schema")
        validate_identifier(self.task_id, field_name="task_id")
        if self.mode != "optimize_existing":
            raise ContractError(f"Unsupported task mode: {self.mode}", "unsupported_task_mode")
        if self.language not in _RECOGNIZED_LANGUAGES:
            raise ContractError(f"Unsupported language: {self.language}", "unsupported_language")
        if self.language == "hip":
            raise ContractError(
                "Standalone HIP execution is unavailable in V1",
                "hip_execution_unavailable",
            )
        if not self.workspace.is_absolute() or not self.results_dir.is_absolute():
            raise ContractError("workspace and results_dir must be absolute", "path_not_absolute")
        if not self.instructions.strip():
            raise ContractError("instructions must not be empty", "empty_instructions")
        if not self.editable_files:
            raise ContractError("editable_files must not be empty", "missing_editable_files")
        for path in self.editable_files:
            _relative_source_path(path, field_name="editable file")
        if not self.target_functions:
            raise ContractError("target_functions must not be empty", "missing_target_functions")
        required = {"compile", "correctness", "performance"}
        missing = sorted(required.difference(self.commands))
        if missing:
            raise ContractError(f"Missing task commands: {', '.join(missing)}", "missing_task_commands")
        self._validate_recipe()
        if self.measurement and self.measurement.report_path in self.editable_files:
            raise ContractError(
                "Measurement report cannot be agent-editable",
                "measurement_report_editable",
            )
        if self.dataset_split not in _DATASET_SPLITS:
            raise ContractError("Invalid dataset split", "invalid_dataset_split")
        if self.data_visibility not in _DATA_VISIBILITIES:
            raise ContractError("Invalid data visibility", "invalid_data_visibility")
        if self.data_visibility == "heldout_private" and self.dataset_split != "heldout":
            raise ContractError(
                "heldout_private visibility requires the heldout split",
                "invalid_data_partition",
            )

    def _validate_recipe(self) -> None:
        if self.recipe is not None and self.recipe.kind != "python_triton":
            raise ContractError("Python/Triton requires a python_triton recipe", "recipe_language_mismatch")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskSpec":
        commands_data = _mapping(data.get("commands"), field_name="commands")
        agent_data = _mapping(data.get("agent_options"), field_name="agent_options")
        budget_data = _mapping(data.get("budget"), field_name="budget")
        scope_data = _mapping(data.get("scope"), field_name="scope")
        recipe_data = data.get("recipe")
        measurement_data = data.get("measurement")
        delivery_data = _mapping(data.get("delivery"), field_name="delivery")
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            task_id=str(data.get("task_id", "")),
            workspace=Path(str(data.get("workspace", ""))),
            results_dir=Path(str(data.get("results_dir", ""))),
            instructions=str(data.get("instructions", "")),
            language=str(data.get("language", "")),
            editable_files=tuple(
                _relative_source_path(item, field_name="editable file")
                for item in _string_tuple(data.get("editable_files"), field_name="editable_files")
            ),
            target_functions=_string_tuple(data.get("target_functions"), field_name="target_functions"),
            commands={str(name): CommandSpec.from_mapping(command) for name, command in commands_data.items()},
            gpu_arch=str(data.get("gpu_arch", "gfx950")),
            mode=str(data.get("mode", "optimize_existing")),
            agent_backend=AgentBackendName(str(data.get("agent_backend", "codex"))),
            agent_options=AgentOptions(
                model=str(agent_data["model"]) if agent_data.get("model") else None,
                effort=str(agent_data["effort"]) if agent_data.get("effort") else None,
            ),
            budget=TaskBudget(
                max_iterations=int(budget_data.get("max_iterations", 1)),
                max_turns=int(budget_data.get("max_turns", 25)),
                timeout_seconds=int(budget_data.get("timeout_seconds", 3600)),
            ),
            scope=TaskScope.from_mapping(scope_data),
            measurement=(
                KernelMeasurementSpec.from_mapping(measurement_data)
                if isinstance(measurement_data, Mapping)
                else None
            ),
            recipe=TaskRecipe.from_mapping(recipe_data) if isinstance(recipe_data, Mapping) else None,
            delivery=DeliverySpec(str(delivery_data.get("mode", "bundle"))),
            dataset_split=str(data.get("dataset_split", "train")),
            data_visibility=str(data.get("data_visibility", "public")),
        )

    @classmethod
    def from_file(cls, path: Path) -> "TaskSpec":
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(content)
        if not isinstance(data, Mapping):
            raise ContractError("TaskSpec document must contain an object", "invalid_task_spec")
        return cls.from_mapping(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "workspace": str(self.workspace),
            "results_dir": str(self.results_dir),
            "instructions": self.instructions,
            "language": self.language,
            "editable_files": list(self.editable_files),
            "target_functions": list(self.target_functions),
            "commands": {name: command.to_dict() for name, command in sorted(self.commands.items())},
            "gpu_arch": self.gpu_arch,
            "mode": self.mode,
            "agent_backend": self.agent_backend.value,
            "agent_options": {"model": self.agent_options.model, "effort": self.agent_options.effort},
            "budget": {
                "max_iterations": self.budget.max_iterations,
                "max_turns": self.budget.max_turns,
                "timeout_seconds": self.budget.timeout_seconds,
            },
            "scope": self.scope.to_dict(),
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "recipe": self.recipe.to_dict() if self.recipe else None,
            "delivery": {"mode": self.delivery.mode},
            "dataset_split": self.dataset_split,
            "data_visibility": self.data_visibility,
        }


@dataclass(frozen=True, slots=True)
class ResolvedTaskSpec:
    """TaskSpec with evaluator-owned workspace paths and baseline hashes."""

    task: TaskSpec
    workspace: Path
    editable_paths: tuple[Path, ...]
    baseline_file_hashes: Mapping[str, str]
    resolution_hash: str
