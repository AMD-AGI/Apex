"""Codex CLI adapter."""

from __future__ import annotations

from apex.core import AgentBackendName
from apex.ports import AgentRequest, AgentResult

from .common import (
    execute_agent_cli,
    invocation_environment,
    require_executable,
)
from .supervisor import SubprocessSupervisor


class CodexBackend:
    name = AgentBackendName.CODEX

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor()

    def run(self, request: AgentRequest) -> AgentResult:
        executable = require_executable("codex")
        argv = [
            executable,
            "exec",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "workspace-write",
            "--config",
            'approval_policy="never"',
            "--strict-config",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--ephemeral",
            "-C",
            str(request.workspace),
        ]
        if request.model:
            argv.extend(["--model", request.model])
        if request.effort:
            argv.extend(["--config", f'model_reasoning_effort="{request.effort}"'])
        argv.append("-")
        environment = invocation_environment(
            request.environment,
            credential_key="OPENAI_API_KEY",
        )
        return execute_agent_cli(
            request,
            self._supervisor,
            backend=self.name,
            cli_name="codex",
            executable=executable,
            argv=argv,
            environment=environment,
            prompt_transport="stdin",
            isolation={
                "approval": "never_via_strict_config",
                "execpolicy_rules": "ignored",
                "project_instructions": "backend_default_may_load",
                "response_token_limit": "not_supported_context_advisory_only",
                "sandbox": "workspace-write",
                "session": "ephemeral",
                "user_config": "ignored",
            },
            effort=request.effort,
            stdin_text=request.prompt,
        )
