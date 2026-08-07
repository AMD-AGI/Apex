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
from apex.ports import AgentRequest


class FakeSupervisor:
    def __init__(self, stdout: str | None = None) -> None:
        self.call: dict[str, object] | None = None
        self.stdout = (
            '{"type":"turn.completed","usage":{"input_tokens":1}}\n'
            if stdout is None
            else stdout
        )

    def run(
        self,
        argv,
        *,
        cwd,
        environment,
        timeout_seconds,
        stdin_text=None,
        stdout_budget=None,
    ):
        self.call = {
            "argv": tuple(argv),
            "cwd": cwd,
            "environment": environment,
            "timeout_seconds": timeout_seconds,
            "stdin_text": stdin_text,
            "stdout_budget": stdout_budget,
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
        budget_exceeded = False
        emitted: list[str] = []
        for line in self.stdout.splitlines(keepends=True):
            emitted.append(line)
            if stdout_budget is not None and stdout_budget(line):
                budget_exceeded = True
                break
        return ProcessResult(
            argv=tuple(argv),
            exit_code=-15 if budget_exceeded else 0,
            timed_out=False,
            stdout="".join(emitted),
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.1,
            budget_exceeded=budget_exceeded,
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


def test_backend_stream_budget_exhaustion_is_typed_failure(
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

    assert result.budget_exceeded
    assert not result.budget_enforcement_failed
    assert result.budget_reason == "max_turns_exceeded"
    assert result.observed_turns == 2
    assert not result.succeeded
    assert result.exit_code == -15


def test_backend_success_without_structured_turn_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("apex.execution.codex.require_executable", lambda _: sys.executable)

    result = CodexBackend(FakeSupervisor("diagnostic only\n")).run(_request(tmp_path))

    assert result.budget_enforcement_failed
    assert result.budget_reason == "missing_structured_turn_evidence"
    assert not result.budget_exceeded
    assert not result.succeeded
