"""Shared normalization helpers for command-line agent adapters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
from typing import Mapping

from apex.core import AgentBackendName, DependencyError, sha256_file
from apex.ports import AgentInvocationReceipt, AgentRequest, AgentResult

from .environment import (
    AGENT_CONFIG_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    build_subprocess_environment,
)
from .supervisor import SubprocessSupervisor
from .transcript import parse_agent_output
from .turn_budget import AgentTurnBudget, TURN_POLICY


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise DependencyError(f"Agent CLI is not installed: {name}", "agent_cli_missing", {"cli": name})
    return path


def invocation_environment(
    overrides: Mapping[str, str], *, credential_key: str
) -> dict[str, str]:
    """Build one backend-scoped environment with only its own credential."""

    return build_subprocess_environment(
        overrides,
        inherit=(
            *AGENT_CONFIG_ENVIRONMENT_KEYS,
            *GPU_RUNTIME_ENVIRONMENT_KEYS,
            *HF_RUNTIME_ENVIRONMENT_KEYS,
        ),
        inherit_secrets=(credential_key,),
        allow_override_secrets=(credential_key,),
    )


def invocation_receipt(
    request: AgentRequest,
    *,
    cli_name: str,
    cli_version: str,
    executable: str,
    argv: list[str],
    prompt_transport: str,
    turn_policy: str,
    isolation: Mapping[str, str],
) -> AgentInvocationReceipt:
    """Bind an invocation to exact entrypoint bytes and explicit policy claims."""

    discovered = Path(executable).absolute()
    resolved = discovered.resolve(strict=True)
    metadata = resolved.stat()
    digest = _entrypoint_digest(
        str(resolved),
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    return AgentInvocationReceipt(
        cli_name=cli_name,
        cli_version=cli_version,
        executable_path=str(discovered),
        resolved_executable_path=str(resolved),
        entrypoint_sha256=digest,
        argv=tuple(argv),
        workspace=str(request.workspace),
        prompt_transport=prompt_transport,
        requested_allowed_files=request.allowed_files,
        allowed_files_enforced_by_cli=False,
        max_turns=request.max_turns,
        turn_policy=turn_policy,
        isolation=tuple(sorted(isolation.items())),
    )


@lru_cache(maxsize=16)
def _entrypoint_digest(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return sha256_file(Path(path))


def resolve_cli_version(
    supervisor: SubprocessSupervisor,
    *,
    executable: str,
    workspace: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> str:
    """Return the CLI's own bounded version output or fail provenance closed."""

    result = supervisor.run(
        [executable, "--version"],
        cwd=workspace,
        environment=environment,
        timeout_seconds=min(timeout_seconds, 30),
    )
    output = result.stdout.strip() or result.stderr.strip()
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
        or not output
        or len(output) > 512
    ):
        raise DependencyError(
            "Agent CLI version could not be identified",
            "agent_cli_identity_failed",
            {"executable": executable},
        )
    return output


def execute_agent_cli(
    request: AgentRequest,
    supervisor: SubprocessSupervisor,
    *,
    backend: AgentBackendName,
    cli_name: str,
    executable: str,
    argv: list[str],
    environment: Mapping[str, str],
    prompt_transport: str,
    isolation: Mapping[str, str],
    effort: str | None,
    stdin_text: str | None = None,
) -> AgentResult:
    """Execute one isolated structured CLI stream with common budget evidence."""

    cli_version = resolve_cli_version(
        supervisor,
        executable=executable,
        workspace=request.workspace,
        environment=environment,
        timeout_seconds=request.timeout_seconds,
    )
    budget = AgentTurnBudget(request.max_turns)
    invocation = invocation_receipt(
        request,
        cli_name=cli_name,
        cli_version=cli_version,
        executable=executable,
        argv=argv,
        prompt_transport=prompt_transport,
        turn_policy=TURN_POLICY,
        isolation=isolation,
    )
    process = supervisor.run(
        argv,
        cwd=request.workspace,
        environment=environment,
        timeout_seconds=request.timeout_seconds,
        stdin_text=stdin_text,
        stdout_budget=budget.observe,
    )
    budget.finalize(
        process_succeeded=process.exit_code == 0 and not process.timed_out,
        observer_stopped=process.budget_exceeded,
    )
    parsed = parse_agent_output(process.stdout)
    return AgentResult(
        backend=backend,
        model=request.model,
        exit_code=process.exit_code,
        timed_out=process.timed_out,
        events=parsed.events,
        stdout=process.stdout,
        stderr=process.stderr,
        duration_seconds=process.duration_seconds,
        semantic_events=parsed.semantic_events,
        usage=parsed.usage,
        cost=parsed.cost,
        effort=effort,
        invocation=invocation,
        budget_exceeded=budget.budget_exceeded,
        budget_enforcement_failed=budget.enforcement_failed,
        budget_reason=budget.stop_reason,
        observed_turns=budget.observed_turns,
    )
