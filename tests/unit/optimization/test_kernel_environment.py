from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError
from apex.execution import ProcessResult
from apex.intake import CommandSpec, TaskResolver, TaskSpec
from apex.optimization.kernel import CandidateVerifier, candidate_source_digest
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentProcessContainmentReceipt,
)


def _containment() -> AgentProcessContainmentReceipt:
    return AgentProcessContainmentReceipt(
        policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        launcher_path="/usr/bin/bwrap",
        launcher_sha256="b" * 64,
        namespace_init_host_pid=100,
        namespace_init_starttime=200,
        namespace_init_inner_pid=1,
        pid_namespace_inode=300,
        mount_namespace_inode=301,
        ipc_namespace_inode=302,
        user_namespace_inode=303,
        private_procfs_verified=True,
        pidfd_opened=True,
        termination_reason="natural_exit",
        teardown_mode="natural_exit",
        pidfd_sigkill_sent=False,
        namespace_init_exit_verified=True,
        wrapper_exit_verified=True,
        wrapper_force_killed=False,
        terminal_status_verified=True,
        terminal_status_absent_after_sigkill=False,
        status_eof_verified=True,
        namespace_membership_scan_complete=True,
        live_namespace_members_after=(),
    )


class RecordingSupervisor:
    def __init__(self) -> None:
        self.environment: dict[str, str] | None = None

    def run(
        self,
        argv,
        *,
        cwd,
        environment,
        timeout_seconds,
        stdin_text=None,
        require_pid_namespace=False,
    ):
        self.environment = dict(environment)
        assert require_pid_namespace is True
        return ProcessResult(
            tuple(argv),
            0,
            False,
            "ok",
            "",
            False,
            False,
            0.1,
            process_containment=_containment(),
        )


class UncontainedSupervisor:
    def run(self, argv, **kwargs):
        assert kwargs["require_pid_namespace"] is True
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


def test_kernel_verifier_rejects_missing_process_tree_containment(
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path)
    digest = candidate_source_digest(
        resolved.workspace, resolved.task.editable_files
    )

    with pytest.raises(IntegrityError) as raised:
        CandidateVerifier(UncontainedSupervisor()).performance(
            resolved,
            candidate_root=resolved.workspace,
            expected_source_digest=digest,
        )

    assert raised.value.reason_code == "verifier_process_containment_failed"
