from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from apex.core import AgentBackendName, ContractError
from apex.intake import E2EOptimizeSpec, TaskResolver, TaskSpec


def _commands() -> dict[str, dict[str, object]]:
    return {
        "compile": {"argv": ["python", "scripts/compile.py"]},
        "correctness": {"argv": ["python", "scripts/correctness.py"]},
        "performance": {"argv": ["python", "scripts/performance.py"]},
    }


def _task_mapping(workspace: Path, results_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": "rms-norm",
        "workspace": str(workspace),
        "results_dir": str(results_dir),
        "instructions": "Optimize rms_norm while preserving its API.",
        "language": "triton",
        "editable_files": ["source/kernel.py"],
        "target_functions": ["rms_norm"],
        "commands": _commands(),
    }


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "source").mkdir(parents=True)
    (workspace / "source" / "kernel.py").write_text("def rms_norm(x):\n    return x\n", encoding="utf-8")
    return workspace


def test_task_spec_defaults_to_codex_and_bundle(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    task = TaskSpec.from_mapping(_task_mapping(workspace, tmp_path / "results"))

    assert task.agent_backend is AgentBackendName.CODEX
    assert task.delivery.mode == "bundle"
    assert task.mode == "optimize_existing"
    assert task.dataset_split == "train"
    assert task.data_visibility == "public"


def test_e2e_spec_rejects_superseded_unenforced_output_limit(tmp_path: Path) -> None:
    with pytest.raises(ContractError) as raised:
        E2EOptimizeSpec.from_mapping(
            {
                "config_path": str(tmp_path / "benchmark.yaml"),
                "results_dir": str(tmp_path / "results"),
                "context_output_tokens": 8000,
            }
        )

    assert raised.value.reason_code == "superseded_context_budget_field"


def test_task_spec_round_trip_json(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_task_mapping(workspace, tmp_path / "results")), encoding="utf-8")

    task = TaskSpec.from_file(task_path)

    assert TaskSpec.from_mapping(task.to_dict()).to_dict() == task.to_dict()


def test_task_spec_preserves_matched_agent_options_and_budget(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["agent_options"] = {"model": "gpt-5.5", "effort": "xhigh"}
    data["budget"] = {"max_iterations": 2, "max_turns": 40, "timeout_seconds": 3600}
    data["scope"] = {
        "dtype": ["FP16", "bf16"],
        "regime": ["Decode"],
        "framework": ["vLLM"],
        "versions": {"rocm": "7.2"},
    }

    task = TaskSpec.from_mapping(data)

    assert task.agent_options.model == "gpt-5.5"
    assert task.agent_options.effort == "xhigh"
    assert task.budget.max_iterations == 2
    assert task.budget.max_turns == 40
    assert task.budget.timeout_seconds == 3600
    assert task.scope.dtype == ("bf16", "fp16")
    assert task.scope.regime == ("decode",)
    assert task.scope.framework == ("vllm",)
    assert task.scope.versions == (("rocm", "7.2"),)


def test_measurement_contract_is_trusted_and_not_editable(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["measurement"] = {
        "schema": "apex.kernel-measurement/v1",
        "report_path": "build/apex_kernel_measurement.json",
        "aggregation": "equal_case",
    }
    task = TaskSpec.from_mapping(data)
    assert task.measurement is not None
    assert task.measurement.report_path == "build/apex_kernel_measurement.json"
    assert task.measurement.keep_srobust_threshold == 1.05
    assert task.measurement.bootstrap_seed == 1729
    assert task.measurement.bootstrap_repetitions == 1000

    data["measurement"]["report_path"] = "source/kernel.py"
    with pytest.raises(ContractError) as editable:
        TaskSpec.from_mapping(data)
    assert editable.value.reason_code == "measurement_report_editable"


def test_measurement_statistics_policy_is_frozen_in_task_round_trip(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["measurement"] = {
        "report_path": "build/timings.json",
        "keep_srobust_threshold": 1.08,
        "confidence_srobust_floor": 1.01,
        "worst_case_srobust_floor": 1.0,
        "max_cv": 0.05,
        "bootstrap_confidence_level": 0.99,
        "bootstrap_seed": 91,
        "bootstrap_repetitions": 2000,
        "min_bootstrap_units": 3,
    }

    task = TaskSpec.from_mapping(data)
    serialized = task.to_dict()

    assert serialized["measurement"] == task.measurement.to_dict()  # type: ignore[union-attr]
    assert TaskSpec.from_mapping(serialized).to_dict() == serialized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("keep_srobust_threshold", 0.0),
        ("keep_srobust_threshold", 1.049),
        ("confidence_srobust_floor", 0.99),
        ("worst_case_srobust_floor", 0.99),
        ("max_cv", 0.11),
        ("max_cv", float("nan")),
        ("bootstrap_confidence_level", 0.90),
        ("bootstrap_confidence_level", 1.0),
        ("bootstrap_repetitions", 99),
        ("min_bootstrap_units", 1),
    ],
)
def test_invalid_measurement_statistics_policy_fails_closed(
    tmp_path: Path, field: str, value: object
) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["measurement"] = {"report_path": "build/timings.json", field: value}

    with pytest.raises(ContractError) as raised:
        TaskSpec.from_mapping(data)

    assert raised.value.reason_code == "invalid_measurement_contract"


def test_resolver_rejects_stale_measurement_report(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    report = workspace / "build" / "apex_kernel_measurement.json"
    report.parent.mkdir()
    report.write_text('{"stale": true}', encoding="utf-8")
    data = _task_mapping(workspace, tmp_path / "results")
    data["measurement"] = {
        "report_path": "build/apex_kernel_measurement.json",
    }

    with pytest.raises(ContractError) as raised:
        TaskResolver().resolve(TaskSpec.from_mapping(data))

    assert raised.value.reason_code == "stale_measurement_report"


def test_resolver_rejects_symlinked_measurement_parent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "build").symlink_to(outside, target_is_directory=True)
    data = _task_mapping(workspace, tmp_path / "results")
    data["measurement"] = {
        "report_path": "build/apex_kernel_measurement.json",
    }

    with pytest.raises(ContractError) as raised:
        TaskResolver().resolve(TaskSpec.from_mapping(data))

    assert raised.value.reason_code == "measurement_report_path_escape"


def test_shell_command_string_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["commands"] = {**_commands(), "compile": {"argv": "python compile.py && touch hacked"}}

    with pytest.raises(ContractError, match="argv") as raised:
        TaskSpec.from_mapping(data)

    assert raised.value.reason_code == "shell_command_forbidden"


@pytest.mark.parametrize("source", ["../outside.py", "/tmp/outside.py", "source/../outside.py"])
def test_editable_path_escape_is_rejected(tmp_path: Path, source: str) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["editable_files"] = [source]

    with pytest.raises(ContractError) as raised:
        TaskSpec.from_mapping(data)

    assert raised.value.reason_code == "unsafe_source_path"


def test_hip_execution_is_unavailable_without_a_recipe(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["language"] = "hip"

    with pytest.raises(ContractError) as raised:
        TaskSpec.from_mapping(data)

    assert raised.value.reason_code == "hip_execution_unavailable"


def test_hip_execution_is_unavailable_even_with_a_complete_fixed_recipe(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, tmp_path / "results")
    data["language"] = "hip"
    data["recipe"] = {
        "kind": "fixed_hip",
        "recipe_id": "aka-hip-v1",
        "sha256": "1" * 64,
        "provenance": "external_evaluator",
    }
    data["commands"] = {
        **_commands(),
        "build": {"argv": ["cmake", "--build", "build"]},
        "deploy": {"argv": ["python", "scripts/deploy.py"]},
        "engagement": {"argv": ["python", "scripts/engagement.py"]},
    }

    with pytest.raises(ContractError) as raised:
        TaskSpec.from_mapping(data)

    assert raised.value.reason_code == "hip_execution_unavailable"


def test_resolver_hashes_evaluator_owned_source(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    task = TaskSpec.from_mapping(_task_mapping(workspace, tmp_path / "results"))

    resolved = TaskResolver().resolve(task)

    assert resolved.workspace == workspace.resolve()
    assert set(resolved.baseline_file_hashes) == {"source/kernel.py"}
    assert len(resolved.baseline_file_hashes["source/kernel.py"]) == 64
    assert len(resolved.resolution_hash) == 64


def test_resolver_rejects_results_inside_source_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    data = _task_mapping(workspace, workspace / "results")

    with pytest.raises(ContractError) as raised:
        TaskResolver().resolve(TaskSpec.from_mapping(data))

    assert raised.value.reason_code == "results_inside_workspace"


def test_resolver_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    original = workspace / "source" / "kernel.py"
    symlink = workspace / "source" / "linked.py"
    symlink.symlink_to(original)
    data = _task_mapping(workspace, tmp_path / "results")
    data["editable_files"] = ["source/linked.py"]
    with pytest.raises(ContractError) as raised:
        TaskResolver().resolve(TaskSpec.from_mapping(data))
    assert raised.value.reason_code == "source_symlink"

    symlink.unlink()
    os.link(original, workspace / "source" / "linked.py")
    with pytest.raises(ContractError) as raised:
        TaskResolver().resolve(TaskSpec.from_mapping(data))
    assert raised.value.reason_code == "source_hardlink"


def test_e2e_spec_is_kernel_only_and_has_frozen_default_gates(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    spec = E2EOptimizeSpec.from_mapping(
        {"config_path": str(config), "results_dir": str(tmp_path / "results")}
    )

    assert spec.scope == "kernels"
    assert spec.agent_backend is AgentBackendName.CODEX
    assert spec.goal.primary == "throughput"
    assert spec.goal.gates.ttft_p99_regression_pct == 5
    assert spec.goal.gates.tpot_p99_regression_pct == 2
    assert spec.dataset_split == "train"
    assert spec.data_visibility == "public"

    with pytest.raises(ContractError) as raised:
        E2EOptimizeSpec.from_mapping(
            {
                "config_path": str(config),
                "results_dir": str(tmp_path / "results"),
                "scope": "all",
            }
        )
    assert raised.value.reason_code == "unsupported_e2e_scope"


@pytest.mark.parametrize(
    "gates",
    [
        {"ttft_p99_regression_pct": 5.01},
        {"tpot_p99_regression_pct": 2.01},
        {"ttft_p99_regression_pct": float("nan")},
        {"tpot_p99_regression_pct": float("inf")},
    ],
)
def test_e2e_spec_rejects_weakened_or_nonfinite_tail_gates(
    tmp_path: Path, gates: dict[str, float]
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")

    with pytest.raises(ContractError) as raised:
        E2EOptimizeSpec.from_mapping(
            {
                "config_path": str(config),
                "results_dir": str(tmp_path / "results"),
                "goal": {"gates": gates},
            }
        )

    assert raised.value.reason_code == "invalid_regression_gate"


@pytest.mark.parametrize(
    "goal",
    [
        {"primary": "ttft", "direction": "minimize"},
        {"primary": "throughput", "direction": "minimize"},
        {"primary": "latency", "direction": "maximize"},
    ],
)
def test_e2e_spec_rejects_unimplemented_metric_goals(
    tmp_path: Path, goal: dict[str, str]
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")

    with pytest.raises(ContractError) as raised:
        E2EOptimizeSpec.from_mapping(
            {
                "config_path": str(config),
                "results_dir": str(tmp_path / "results"),
                "goal": goal,
            }
        )

    assert raised.value.reason_code == "unsupported_metric_goal"


def test_e2e_spec_freezes_agent_model_and_effort(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {}\n", encoding="utf-8")
    spec = E2EOptimizeSpec.from_mapping(
        {
            "config_path": str(config),
            "results_dir": str(tmp_path / "results"),
            "agent_model": "gpt-5.5",
            "agent_effort": "xhigh",
        }
    )

    assert spec.agent_model == "gpt-5.5"
    assert spec.agent_effort == "xhigh"
    assert spec.to_dict()["agent_effort"] == "xhigh"


@pytest.mark.parametrize("kind", ["kernel", "e2e"])
def test_specs_validate_dataset_partition(tmp_path: Path, kind: str) -> None:
    if kind == "kernel":
        data = _task_mapping(_workspace(tmp_path), tmp_path / "results")
        build = TaskSpec.from_mapping
    else:
        config = tmp_path / "benchmark.yaml"
        config.write_text("benchmark: {}\n", encoding="utf-8")
        data = {"config_path": str(config), "results_dir": str(tmp_path / "results")}
        build = E2EOptimizeSpec.from_mapping

    data["dataset_split"] = "training"
    with pytest.raises(ContractError) as invalid_split:
        build(data)
    assert invalid_split.value.reason_code == "invalid_dataset_split"

    data["dataset_split"] = "train"
    data["data_visibility"] = "heldout_private"
    with pytest.raises(ContractError) as invalid_partition:
        build(data)
    assert invalid_partition.value.reason_code == "invalid_data_partition"

    data["dataset_split"] = "heldout"
    accepted = build(data)
    assert accepted.dataset_split == "heldout"
    assert accepted.data_visibility == "heldout_private"
