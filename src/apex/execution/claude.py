"""Claude Code CLI adapter."""

from __future__ import annotations

from apex.core import AgentBackendName
from apex.ports import AgentRequest, AgentResult

from .common import (
    execute_agent_cli,
    invocation_environment,
    require_executable,
)
from .supervisor import SubprocessSupervisor


class ClaudeBackend:
    name = AgentBackendName.CLAUDE

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor()

    def run(self, request: AgentRequest) -> AgentResult:
        executable = require_executable("claude")
        argv = [
            executable,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--bare",
            "--safe-mode",
            "--disable-slash-commands",
            "--permission-mode",
            "dontAsk",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--strict-mcp-config",
            "--no-session-persistence",
        ]
        if request.model:
            argv.extend(["--model", request.model])
        if request.effort:
            argv.extend(["--effort", request.effort])
        argv.append(request.prompt)
        environment = invocation_environment(
            request.environment,
            credential_key="ANTHROPIC_API_KEY",
        )
        return execute_agent_cli(
            request,
            self._supervisor,
            backend=self.name,
            cli_name="claude",
            executable=executable,
            argv=argv,
            environment=environment,
            prompt_transport="argv",
            isolation={
                "approval": "dontAsk",
                "customizations": "bare_and_safe_mode",
                "filesystem": "built_in_permissions_not_path_allowlist",
                "mcp": "strict_explicit_empty",
                "project_instructions": "disabled_by_safe_mode",
                "response_token_limit": "not_supported_context_advisory_only",
                "sandbox": "claude_permission_policy",
                "session": "not_persisted",
            },
            effort=request.effort,
        )
