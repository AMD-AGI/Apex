"""Native interactive/headless coding sessions, outside formal optimization."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from apex.core import AgentBackendName, ConfigurationError
from apex.ports import (
    CodingSessionOutput,
    CodingSessionRequest,
    KernelEnhancement,
)

from .common import invocation_environment, require_executable
from .skill_assets import KernelSkillPackage


_KERNEL_TERMS = re.compile(
    r"\b(?:amd|gpu|kernel|triton|rocm|hip|aiter|cktile|throughput|latency)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class NativeSessionInvocation:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    stdin_text: str | None
    prompt_transport: str
    kernel_capabilities_enabled: bool
    kernel_skill_ids: tuple[str, ...]
    kernel_skill_digest: str | None
    capability_notices: tuple[str, ...]


class NativeSessionRunner(Protocol):
    def run(self, invocation: NativeSessionInvocation) -> int: ...


class SubprocessSessionRunner:
    def run(self, invocation: NativeSessionInvocation) -> int:
        completed = subprocess.run(
            invocation.argv,
            cwd=invocation.cwd,
            env=dict(invocation.environment),
            input=invocation.stdin_text,
            text=True,
            check=False,
        )
        return completed.returncode


class NativeCodingSessionLauncher:
    """Delegate ordinary coding UX to the selected backend's native session."""

    def __init__(
        self,
        *,
        mcp_command: Sequence[str],
        skill_package: KernelSkillPackage | None = None,
        runner: NativeSessionRunner | None = None,
        executable_resolver: Callable[[str], str] = require_executable,
    ) -> None:
        self._mcp_command = tuple(mcp_command)
        self._skill_package = skill_package
        self._runner = runner or SubprocessSessionRunner()
        self._resolve = executable_resolver

    def launch(self, request: CodingSessionRequest) -> int:
        if not request.workspace.is_dir():
            raise ConfigurationError("Session workspace is not a directory", "workspace_not_directory")
        invocation = self.prepare(request)
        for notice in invocation.capability_notices:
            print(notice, file=sys.stderr)
        return self._runner.run(invocation)

    def prepare(self, request: CodingSessionRequest) -> NativeSessionInvocation:
        requested_enhancement = _should_enable_kernel_capabilities(request)
        enhanced = requested_enhancement
        notices: tuple[str, ...] = ()
        mcp_command = _scoped_mcp_command(self._mcp_command, request)
        skills = self._skill_package if enhanced else None
        if request.backend is AgentBackendName.CODEX:
            executable = self._resolve("codex")
            argv = _codex_argv(
                request,
                executable,
                mcp_command if enhanced else (),
                skills,
            )
            credential = "OPENAI_API_KEY"
        elif request.backend is AgentBackendName.CLAUDE:
            executable = self._resolve("claude")
            argv = _claude_argv(
                request,
                executable,
                mcp_command if enhanced else (),
                skills,
            )
            credential = "ANTHROPIC_API_KEY"
        else:
            if request.effort is not None:
                raise ConfigurationError(
                    "Cursor Agent has no independent effort option",
                    "agent_effort_unsupported",
                )
            executable = self._resolve("cursor-agent")
            argv = _cursor_argv(request, executable, skills)
            credential = "CURSOR_API_KEY"
            if requested_enhancement:
                notices = (
                    "kernel_mcp_bridge_unavailable: Cursor session includes Apex "
                    "skills without Apex MCP tools",
                )
                enhanced = skills is not None
        if requested_enhancement and skills is None:
            notices = (
                *notices,
                "kernel_skill_bridge_unavailable: packaged Apex skills are unavailable",
            )
        return NativeSessionInvocation(
            tuple(argv),
            request.workspace,
            invocation_environment({}, credential_key=credential),
            request.prompt
            if request.output is not CodingSessionOutput.INTERACTIVE
            else None,
            _prompt_transport(request),
            enhanced,
            tuple(sorted(skills.skill_paths)) if skills is not None else (),
            skills.digest if skills is not None else None,
            notices,
        )


def _scoped_mcp_command(
    command: Sequence[str], request: CodingSessionRequest
) -> tuple[str, ...]:
    results = request.results_dir or default_capability_results(request.workspace)
    return (
        *command,
        "--workspace",
        str(request.workspace),
        "--results",
        str(results),
    )


def default_capability_results(workspace: Path) -> Path:
    """Select a stable default outside the source workspace."""

    root = workspace.resolve(strict=True)
    if root.parent != root:
        return root.parent / f".{root.name}.apex-capability-results"
    return Path(tempfile.gettempdir()).resolve() / "apex-root-capability-results"


def _should_enable_kernel_capabilities(request: CodingSessionRequest) -> bool:
    if request.enhancement is KernelEnhancement.PLAIN:
        return False
    if request.enhancement is KernelEnhancement.KERNEL:
        return True
    return bool(request.prompt and _KERNEL_TERMS.search(request.prompt))


def _prompt_transport(request: CodingSessionRequest) -> str:
    if request.prompt is None:
        return "none"
    if request.output is CodingSessionOutput.INTERACTIVE:
        return "argv"
    return "stdin"


def _codex_argv(
    request: CodingSessionRequest,
    executable: str,
    mcp_command: Sequence[str],
    skills: KernelSkillPackage | None,
) -> list[str]:
    argv = [executable]
    if request.output is not CodingSessionOutput.INTERACTIVE:
        argv.append("exec")
    argv.extend(["-C", str(request.workspace)])
    _model_effort(argv, request, codex=True)
    if request.output is CodingSessionOutput.JSONL:
        argv.append("--json")
    if mcp_command:
        argv.extend(_codex_mcp_config(mcp_command))
    if skills is not None:
        argv.extend(["--config", _codex_skill_config(skills)])
    if request.resume_session is not None or request.resume_latest:
        argv.append("resume")
        argv.extend(
            [request.resume_session]
            if request.resume_session is not None
            else ["--last"]
        )
    if request.prompt is not None and request.output is CodingSessionOutput.INTERACTIVE:
        argv.append(request.prompt)
    return argv


def _claude_argv(
    request: CodingSessionRequest,
    executable: str,
    mcp_command: Sequence[str],
    skills: KernelSkillPackage | None,
) -> list[str]:
    argv = [executable]
    if request.output is not CodingSessionOutput.INTERACTIVE:
        argv.extend(
            [
                "--print",
                "--output-format",
                "stream-json" if request.output is CodingSessionOutput.JSONL else "text",
            ]
        )
        if request.output is CodingSessionOutput.JSONL:
            argv.append("--verbose")
    _model_effort(argv, request, codex=False)
    if mcp_command:
        argv.extend(["--mcp-config", _claude_mcp_config(mcp_command)])
    if skills is not None:
        argv.extend(["--plugin-dir", str(skills.root)])
    if request.resume_session is not None:
        argv.extend(["--resume", request.resume_session])
    elif request.resume_latest:
        argv.append("--continue")
    if request.prompt is not None and request.output is CodingSessionOutput.INTERACTIVE:
        argv.append(request.prompt)
    return argv


def _cursor_argv(
    request: CodingSessionRequest,
    executable: str,
    skills: KernelSkillPackage | None,
) -> list[str]:
    argv = [executable, "--workspace", str(request.workspace)]
    if request.output is not CodingSessionOutput.INTERACTIVE:
        argv.extend(
            [
                "--print",
                "--trust",
                "--output-format",
                "stream-json" if request.output is CodingSessionOutput.JSONL else "text",
            ]
        )
    if request.model:
        argv.extend(["--model", request.model])
    if skills is not None:
        argv.extend(["--plugin-dir", str(skills.root)])
    if request.resume_session is not None:
        argv.extend(["--resume", request.resume_session])
    elif request.resume_latest:
        argv.append("--continue")
    if request.prompt is not None and request.output is CodingSessionOutput.INTERACTIVE:
        argv.append(request.prompt)
    return argv


def _model_effort(argv: list[str], request: CodingSessionRequest, *, codex: bool) -> None:
    if request.model:
        argv.extend(["--model", request.model])
    if request.effort:
        if codex:
            argv.extend(["--config", f'model_reasoning_effort="{request.effort}"'])
        else:
            argv.extend(["--effort", request.effort])


def _codex_mcp_config(command: Sequence[str]) -> list[str]:
    return [
        "--config",
        f"mcp_servers.apex.command={json.dumps(command[0])}",
        "--config",
        f"mcp_servers.apex.args={json.dumps(list(command[1:]))}",
    ]


def _codex_skill_config(skills: KernelSkillPackage) -> str:
    entries = ",".join(
        "{path=" + json.dumps(str(path)) + ",enabled=true}"
        for _, path in sorted(skills.skill_paths.items())
    )
    return f"skills.config=[{entries}]"


def _claude_mcp_config(command: Sequence[str]) -> str:
    return json.dumps(
        {
            "mcpServers": {
                "apex": {"command": command[0], "args": list(command[1:])}
            }
        },
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "default_capability_results",
    "NativeCodingSessionLauncher",
    "NativeSessionInvocation",
    "NativeSessionRunner",
    "SubprocessSessionRunner",
]
