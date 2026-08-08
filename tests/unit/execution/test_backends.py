from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

from apex.core import AgentBackendName, ConfigurationError
from apex.execution import AgentRegistry, ProcessResult, build_default_registry
from apex.execution.claude import ClaudeBackend
from apex.execution.codex import CodexBackend
from apex.execution.cursor import CursorBackend
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentCaptureStatus,
    AgentProcessContainmentReceipt,
    AgentRequest,
    AgentTerminationKind,
    STRUCTURED_TURN_CHECKPOINT_POLICY,
)


def _containment(*, stopped: bool = False) -> AgentProcessContainmentReceipt:
    return AgentProcessContainmentReceipt(
        policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        launcher_path="/usr/bin/bwrap",
        launcher_sha256="b" * 64,
        namespace_init_host_pid=1234,
        namespace_init_starttime=5678,
        namespace_init_inner_pid=1,
        pid_namespace_inode=9012,
        mount_namespace_inode=9013,
        ipc_namespace_inode=9014,
        user_namespace_inode=9015,
        private_procfs_verified=True,
        pidfd_opened=True,
        termination_reason="stdout_budget_boundary" if stopped else "natural_exit",
        teardown_mode="pidfd_sigkill" if stopped else "natural_exit",
        pidfd_sigkill_sent=stopped,
        namespace_init_exit_verified=True,
        wrapper_exit_verified=True,
        wrapper_force_killed=False,
        terminal_status_verified=True,
        terminal_status_absent_after_sigkill=False,
        status_eof_verified=True,
        namespace_membership_scan_complete=True,
        live_namespace_members_after=(),
    )


class FakeSupervisor:
    def __init__(
        self,
        stdout: str | None = None,
        *,
        stdout_truncated: bool = False,
        cleanup_succeeded: bool = True,
    ) -> None:
        self.call: dict[str, object] | None = None
        self.stdout = (
            '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
            if stdout is None
            else stdout
        )
        self.stdout_truncated = stdout_truncated
        self.cleanup_succeeded = cleanup_succeeded
        self.pid_namespace_requests: list[bool] = []

    def run(
        self,
        argv,
        *,
        cwd,
        environment,
        timeout_seconds,
        stdin_text=None,
        stdout_budget=None,
        require_pid_namespace=False,
    ):
        self.pid_namespace_requests.append(require_pid_namespace)
        self.call = {
            "argv": tuple(argv),
            "cwd": cwd,
            "environment": environment,
            "timeout_seconds": timeout_seconds,
            "stdin_text": stdin_text,
            "stdout_budget": stdout_budget,
            "require_pid_namespace": require_pid_namespace,
        }
        if tuple(argv)[-1] == "--version":
            return ProcessResult(
                argv=tuple(argv),
                exit_code=0,
                timed_out=False,
                stdout="test-agent 1.2.3\n",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_seconds=0.01,
            )
        observer_stopped = False
        emitted: list[str] = []
        for line in self.stdout.splitlines(keepends=True):
            emitted.append(line)
            if stdout_budget is not None and stdout_budget(line):
                observer_stopped = True
                break
        return ProcessResult(
            argv=tuple(argv),
            exit_code=137 if observer_stopped else 0,
            timed_out=False,
            stdout="".join(emitted),
            stderr="",
            stdout_truncated=self.stdout_truncated,
            stderr_truncated=False,
            duration_seconds=0.1,
            observer_stopped=observer_stopped,
            observer_termination_started=observer_stopped,
            process_containment=_containment(stopped=observer_stopped),
            cleanup_succeeded=self.cleanup_succeeded,
        )


def _request(tmp_path: Path) -> AgentRequest:
    return AgentRequest(
        run_id="run-1",
        attempt_id="attempt-1",
        backend=AgentBackendName.CODEX,
        prompt="Optimize only source/kernel.py",
        workspace=tmp_path,
        allowed_files=("source/kernel.py",),
        model="test-model",
        effort="high",
        timeout_seconds=42,
    )


def test_default_registry_uses_codex() -> None:
    registry = build_default_registry()

    assert registry.default is AgentBackendName.CODEX
    assert registry.get().name is AgentBackendName.CODEX
    assert set(registry.names) == set(AgentBackendName)


def test_registry_rejects_missing_default() -> None:
    with pytest.raises(ConfigurationError):
        AgentRegistry([], default=AgentBackendName.CODEX)


def test_codex_uses_stdin_ephemeral_process_and_hides_other_backend_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apex.execution.codex.require_executable", lambda _: sys.executable)
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-anthropic")
    monkeypatch.setenv("CURSOR_API_KEY", "secret-cursor")
    monkeypatch.setenv("BASH_ENV", "/tmp/injected-startup")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/import-first")
    supervisor = FakeSupervisor()

    result = CodexBackend(supervisor).run(_request(tmp_path))

    assert result.succeeded
    assert result.events[0].kind == "turn.completed"
    assert result.usage is not None
    assert result.usage.input_tokens == 1
    assert result.usage.turn_count == 1
    assert result.effort == "high"
    assert result.invocation is not None
    assert result.invocation.cli_version == "test-agent 1.2.3"
    assert result.invocation.argv == supervisor.call["argv"]
    assert result.invocation.entrypoint_sha256
    assert result.invocation.allowed_files_enforced_by_cli is False
    assert supervisor.pid_namespace_requests == [True, True]
    assert dict(result.invocation.isolation)["sandbox"] == "workspace-write"
    assert supervisor.call is not None
    argv = supervisor.call["argv"]
    assert "--ephemeral" in argv
    assert "--sandbox" in argv
    assert "workspace-write" in argv
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert argv[-1] == "-"
    assert supervisor.call["stdin_text"] == "Optimize only source/kernel.py"
    environment = supervisor.call["environment"]
    assert environment["OPENAI_API_KEY"] == "secret-openai"
    assert "ANTHROPIC_API_KEY" not in environment
    assert "CURSOR_API_KEY" not in environment
    assert "BASH_ENV" not in environment
    assert "LD_PRELOAD" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize(
    ("backend_class", "module", "credential", "blocked_credentials"),
    (
        (
            CodexBackend,
            "apex.execution.codex.require_executable",
            "OPENAI_API_KEY",
            ("ANTHROPIC_API_KEY", "CURSOR_API_KEY"),
        ),
        (
            ClaudeBackend,
            "apex.execution.claude.require_executable",
            "ANTHROPIC_API_KEY",
            ("OPENAI_API_KEY", "CURSOR_API_KEY"),
        ),
        (
            CursorBackend,
            "apex.execution.cursor.require_executable",
            "CURSOR_API_KEY",
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
        ),
    ),
)
def test_each_backend_receives_only_its_own_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_class,
    module: str,
    credential: str,
    blocked_credentials: tuple[str, ...],
) -> None:
    monkeypatch.setattr(module, lambda _: sys.executable)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CURSOR_API_KEY"):
        monkeypatch.setenv(key, f"{key}-value")
    monkeypatch.setenv("PYTHONSTARTUP", "/tmp/injected.py")
    supervisor = FakeSupervisor()

    request = _request(tmp_path)
    if backend_class is CursorBackend:
        request = replace(request, effort=None)
    backend_class(supervisor).run(request)

    assert supervisor.call is not None
    environment = supervisor.call["environment"]
    assert environment[credential] == f"{credential}-value"
    assert all(key not in environment for key in blocked_credentials)
    assert "PYTHONSTARTUP" not in environment


def test_cursor_rejects_unrepresentable_effort_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as raised:
        CursorBackend(FakeSupervisor()).run(_request(tmp_path))

    assert raised.value.reason_code == "agent_effort_unsupported"


@pytest.mark.parametrize(
    ("backend_class", "module", "forbidden", "required"),
    (
        (
            ClaudeBackend,
            "apex.execution.claude.require_executable",
            ("--dangerously-skip-permissions",),
            (
                "--bare",
                "--safe-mode",
                "--disable-slash-commands",
                "--strict-mcp-config",
                "dontAsk",
            ),
        ),
        (
            CursorBackend,
            "apex.execution.cursor.require_executable",
            ("--force", "--approve-mcps"),
            ("--sandbox", "enabled"),
        ),
    ),
)
def test_non_codex_backends_use_fail_closed_cli_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_class,
    module: str,
    forbidden: tuple[str, ...],
    required: tuple[str, ...],
) -> None:
    monkeypatch.setattr(module, lambda _: sys.executable)
    supervisor = FakeSupervisor()
    request = replace(_request(tmp_path), effort=None)

    result = backend_class(supervisor).run(request)

    assert result.succeeded
    assert result.invocation is not None
    assert supervisor.call is not None
    argv = supervisor.call["argv"]
    assert all(value not in argv for value in forbidden)
    assert all(value in argv for value in required)
    assert result.invocation.argv == argv


def test_backend_exact_boundary_is_a_complete_candidate_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apex.execution.claude.require_executable", lambda _: sys.executable)
    stdout = "\n".join(
        (
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"first"}]}}',
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"second"}]}}',
        )
    )
    result = ClaudeBackend(FakeSupervisor(stdout)).run(
        replace(_request(tmp_path), backend=AgentBackendName.CLAUDE, max_turns=1)
    )

    assert result.termination_kind is AgentTerminationKind.EXACT_TURN_BOUNDARY
    assert result.capture_status is AgentCaptureStatus.COMPLETE
    assert result.termination_reason == "max_turns_exact_boundary"
    assert result.observed_turns == 1
    assert result.candidate_capture_allowed
    assert not result.succeeded
    assert result.exit_code == 137
    assert result.invocation is not None
    assert result.invocation.turn_policy == STRUCTURED_TURN_CHECKPOINT_POLICY


def test_backend_success_without_structured_turn_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apex.execution.codex.require_executable", lambda _: sys.executable)

    result = CodexBackend(FakeSupervisor("diagnostic only\n")).run(_request(tmp_path))

    assert result.termination_kind is AgentTerminationKind.INVALID_STREAM
    assert result.termination_reason == "missing_structured_turn_evidence"
    assert not result.candidate_capture_allowed
    assert not result.succeeded


def test_backend_rejects_provider_summary_above_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apex.execution.codex.require_executable", lambda _: sys.executable)
    supervisor = FakeSupervisor('{"type":"result","num_turns":2}\n')

    result = CodexBackend(supervisor).run(replace(_request(tmp_path), max_turns=1))

    assert result.termination_kind is AgentTerminationKind.TURN_OVERRUN
    assert result.termination_reason == "max_turns_overrun"
    assert result.observed_turns == 2
    assert not result.candidate_capture_allowed
    assert result.candidate_rejection_reason == "agent_turn_budget_overrun"


@pytest.mark.parametrize(
    ("supervisor", "capture", "reason"),
    (
        (
            FakeSupervisor(stdout_truncated=True),
            AgentCaptureStatus.OUTPUT_TRUNCATED,
            "agent_output_truncated",
        ),
        (
            FakeSupervisor(cleanup_succeeded=False),
            AgentCaptureStatus.CLEANUP_FAILED,
            "agent_process_cleanup_failed",
        ),
    ),
)
def test_backend_rejects_incomplete_exact_boundary_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supervisor: FakeSupervisor,
    capture: AgentCaptureStatus,
    reason: str,
) -> None:
    monkeypatch.setattr("apex.execution.codex.require_executable", lambda _: sys.executable)

    result = CodexBackend(supervisor).run(replace(_request(tmp_path), max_turns=1))

    assert result.termination_kind is AgentTerminationKind.EXACT_TURN_BOUNDARY
    assert result.capture_status is capture
    assert not result.candidate_capture_allowed
    assert result.candidate_rejection_reason == reason
