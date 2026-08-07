from __future__ import annotations

from pathlib import Path

from apex.cli import app
from apex.core import AgentBackendName
from apex.intake import E2EOptimizeSpec, TaskSpec


def _task(tmp_path: Path) -> TaskSpec:
    return TaskSpec.from_mapping(
        {
            "task_id": "cli-budget",
            "workspace": str(tmp_path / "workspace"),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize kernel",
            "language": "triton",
            "editable_files": ["kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                phase: {"argv": ["true"]}
                for phase in ("compile", "correctness", "performance")
            },
        }
    )


def test_kernel_cli_budget_overrides_are_frozen_into_task(tmp_path: Path) -> None:
    args = app._parser().parse_args(
        [
            "optimize", "kernel", "Optimize kernel.py",
            "--workspace", str(tmp_path / "workspace"),
            "--results", str(tmp_path / "results"),
            "--max-iterations", "3", "--max-turns", "17",
            "--timeout-seconds", "901",
        ]
    )

    task = app._kernel_budget_overrides(_task(tmp_path), args)

    assert task.budget.max_iterations == 3
    assert task.budget.max_turns == 17
    assert task.budget.timeout_seconds == 901


def test_e2e_cli_agent_and_budget_overrides_are_explicit(tmp_path: Path) -> None:
    args = app._parser().parse_args(
        [
            "optimize", "e2e", "--spec", str(tmp_path / "spec.yaml"),
            "--agent-backend", "claude", "--agent-model", "opus",
            "--agent-effort", "high", "--max-iterations", "4",
            "--max-kernels", "6", "--max-turns", "19",
            "--timeout-seconds", "902",
        ]
    )
    spec = E2EOptimizeSpec.from_mapping(
        {
            "config_path": str(tmp_path / "benchmark.yaml"),
            "results_dir": str(tmp_path / "results"),
        }
    )

    updated = app._e2e_overrides(spec, args)

    assert updated.agent_backend is AgentBackendName.CLAUDE
    assert updated.agent_model == "opus"
    assert updated.agent_effort == "high"
    assert (updated.max_iterations, updated.max_kernels) == (4, 6)
    assert (updated.max_turns, updated.agent_timeout_seconds) == (19, 902)
