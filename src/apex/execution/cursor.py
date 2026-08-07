"""Cursor Agent CLI adapter."""

from __future__ import annotations

from apex.core import AgentBackendName, ConfigurationError
from apex.ports import AgentRequest, AgentResult

from .common import (
    execute_agent_cli,
    invocation_environment,
    require_executable,
)
from .supervisor import SubprocessSupervisor


class CursorBackend:
    name = AgentBackendName.CURSOR

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor()

    def run(self, request: AgentRequest) -> AgentResult:
        if request.effort is not None:
            raise ConfigurationError(
                "Cursor Agent has no independent effort option",
                "agent_effort_unsupported",
                {"backend": self.name.value},
            )
        executable = require_executable("cursor-agent")
        argv = [
            executable,
            "--print",
            "--trust",
            "--sandbox",
            "enabled",
            "--output-format",
            "stream-json",
            "--workspace",
            str(request.workspace),
        ]
        if request.model:
            argv.extend(["--model", request.model])
        argv.append(request.prompt)
        environment = invocation_environment(
            request.environment,
            credential_key="CURSOR_API_KEY",
        )
        return execute_agent_cli(
            request,
            self._supervisor,
            backend=self.name,
            cli_name="cursor-agent",
            executable=executable,
            argv=argv,
            environment=environment,
            prompt_transport="argv",
            isolation={
                "approval": "default_non_force",
                "config_sources": "backend_default_may_load",
                "mcp": "not_auto_approved",
                "response_token_limit": "not_supported_context_advisory_only",
                "sandbox": "enabled",
                "session": "new_unresumed_may_persist",
                "workspace_trust": "explicit_headless_trust",
            },
            effort=None,
        )
