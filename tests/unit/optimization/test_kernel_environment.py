from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError
from apex.execution import ProcessResult
from apex.intake import CommandSpec, TaskResolver, TaskSpec
from apex.optimization.kernel import CandidateVerifier, candidate_source_digest


class RecordingSupervisor:
    def __init__(self) -> None:
        self.environment: dict[str, str] | None = None

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        self.environment = dict(environment)
        return ProcessResult(tuple(argv), 0, False, "ok", "", False, False, 0.1)


def _resolved(tmp_path: Path, *, command_env: dict[str, str] | None = None):
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "kernel.py").write_text("def kernel(x): return x\n", encoding="utf-8")
    command = {"argv": ["true"], "env": command_env or {}}
    task = TaskSpec.from_mapping(
        {
            "task_id": "environment-test",
            "workspace": str(workspace),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Keep behavior while optimizing the kernel",
            "language": "triton",
            "editable_files": ["kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                "compile": command,
                "correctness": command,
                "performance": command,
            },
        }
    )
    return TaskResolver().resolve(task)


def test_kernel_verifier_uses_gpu_allowlist_and_explicit_safe_command_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("OPENAI_API_KEY", "agent-secret")
    monkeypatch.setenv("BASH_ENV", "/tmp/injected-startup")
    resolved = _resolved(tmp_path, command_env={"KERNEL_CASE": "large"})
    supervisor = RecordingSupervisor()
    digest = candidate_source_digest(
        resolved.workspace, resolved.task.editable_files
    )

    result = CandidateVerifier(supervisor).compile(
        resolved,
        candidate_root=resolved.workspace,
        expected_source_digest=digest,
    )

    assert result.passed
    assert supervisor.environment is not None
    assert supervisor.environment["ROCR_VISIBLE_DEVICES"] == "5"
    assert supervisor.environment["KERNEL_CASE"] == "large"
    assert "OPENAI_API_KEY" not in supervisor.environment
    assert "BASH_ENV" not in supervisor.environment
    assert supervisor.environment["PYTHONNOUSERSITE"] == "1"


def test_kernel_verifier_rejects_python_or_secret_command_env(tmp_path: Path) -> None:
    for command_env in (
        {"PYTHONPATH": "/tmp/import-first"},
        {"ANTHROPIC_API_KEY": "secret"},
    ):
        resolved = _resolved(tmp_path / next(iter(command_env)), command_env=command_env)
        digest = candidate_source_digest(
            resolved.workspace, resolved.task.editable_files
        )
        with pytest.raises(ContractError):
            CandidateVerifier(RecordingSupervisor()).compile(
                resolved,
                candidate_root=resolved.workspace,
                expected_source_digest=digest,
            )
