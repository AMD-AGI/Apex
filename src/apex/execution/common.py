"""Shared normalization helpers for command-line agent adapters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
from typing import Mapping

from apex.core import AgentBackendName, ContractError, DependencyError, sha256_file
from apex.ports import (
    AGENT_PROCESS_CONTAINMENT_POLICY,
    AgentCaptureStatus,
    AgentInvocationReceipt,
    AgentRequest,
    AgentResult,
    AgentTerminationKind,
)

from .environment import (
    AGENT_CONFIG_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    build_subprocess_environment,
)
from .supervisor import ProcessResult, SubprocessSupervisor
from .secret_redaction import redact_secret_values
from .transcript import parse_agent_output
from .turn_budget import AgentTurnBudget, TURN_POLICY


_BACKEND_CREDENTIAL_KEYS = {
    AgentBackendName.CODEX: "OPENAI_API_KEY",
    AgentBackendName.CLAUDE: "ANTHROPIC_API_KEY",
    AgentBackendName.CURSOR: "CURSOR_API_KEY",
}


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
    credential_environment_key: str,
    turn_policy: str,
    isolation: Mapping[str, str],
) -> AgentInvocationReceipt:
    """Bind an invocation to exact entrypoint bytes and explicit policy claims."""

    assert request.execution_authority is not None
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
        execution_authority=request.execution_authority,
        credential_environment_key=credential_environment_key,
        requested_allowed_files=request.allowed_files,
        allowed_files_enforced_by_cli=False,
        max_turns=request.max_turns,
        turn_policy=turn_policy,
        process_containment_policy_id=AGENT_PROCESS_CONTAINMENT_POLICY,
        isolation=tuple(sorted(isolation.items())),
        runtime_closure_sha256=request.runtime_closure_sha256,
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
    secret_values: tuple[str, ...],
) -> str:
    """Return the CLI's own bounded version output or fail provenance closed."""

    result = supervisor.run(
        [executable, "--version"],
        cwd=workspace,
        environment=environment,
        timeout_seconds=min(timeout_seconds, 30),
        require_pid_namespace=True,
    )
    safe_stdout = redact_secret_values(result.stdout, secret_values)
    safe_stderr = redact_secret_values(result.stderr, secret_values)
    output = safe_stdout.text.strip() or safe_stderr.text.strip()
    redaction_count = safe_stdout.replacements + safe_stderr.replacements
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
        or not result.cleanup_succeeded
        or not output
        or len(output) > 512
        or redaction_count
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
    credential_environment_key: str,
    prompt_transport: str,
    isolation: Mapping[str, str],
    effort: str | None,
    stdin_text: str | None = None,
) -> AgentResult:
    """Execute one isolated structured CLI stream with common budget evidence."""

    _validate_execution_authority(
        request,
        backend,
        credential_environment_key,
        prompt_transport,
        argv,
        stdin_text,
    )
    secret_values = _secret_values(environment, credential_environment_key)
    if redact_secret_values("\0".join(argv), secret_values).replacements:
        raise ContractError(
            "Backend credential is present in formal process argv",
            "agent_credential_in_argv",
        )
    cli_version = resolve_cli_version(
        supervisor,
        executable=executable,
        workspace=request.workspace,
        environment=environment,
        timeout_seconds=request.timeout_seconds,
        secret_values=secret_values,
    )
    budget = AgentTurnBudget(request.max_turns)
    invocation = invocation_receipt(
        request,
        cli_name=cli_name,
        cli_version=cli_version,
        executable=executable,
        argv=argv,
        prompt_transport=prompt_transport,
        credential_environment_key=credential_environment_key,
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
        require_pid_namespace=True,
    )
    budget.finalize(
        process_succeeded=process.exit_code == 0 and not process.timed_out,
        observer_stopped=process.observer_stopped,
    )
    return _agent_result(
        request=request,
        backend=backend,
        effort=effort,
        process=process,
        budget=budget,
        invocation=invocation,
        secret_values=secret_values,
    )


def _agent_result(
    *,
    request: AgentRequest,
    backend: AgentBackendName,
    effort: str | None,
    process: ProcessResult,
    budget: AgentTurnBudget,
    invocation: AgentInvocationReceipt,
    secret_values: tuple[str, ...],
) -> AgentResult:
    stdout = redact_secret_values(process.stdout, secret_values)
    stderr = redact_secret_values(process.stderr, secret_values)
    redaction_count = stdout.replacements + stderr.replacements
    parsed = parse_agent_output(stdout.text)
    capture_status = _capture_status(process, redaction_count)
    termination_kind, termination_reason = _termination(process, budget)
    return AgentResult(
        backend=backend,
        model=request.model,
        exit_code=process.exit_code,
        timed_out=process.timed_out,
        events=parsed.events,
        stdout=stdout.text,
        stderr=stderr.text,
        duration_seconds=process.duration_seconds,
        semantic_events=parsed.semantic_events,
        usage=parsed.usage,
        cost=parsed.cost,
        effort=effort,
        invocation=invocation,
        termination_kind=termination_kind,
        capture_status=capture_status,
        termination_reason=termination_reason,
        observed_turns=budget.observed_turns,
        observer_stop_sent=process.observer_stopped,
        process_containment=process.process_containment,
        discarded_stdout_lines=process.discarded_stdout_lines,
        discarded_stdout_bytes=process.discarded_stdout_bytes,
        discarded_stdout_sha256=process.discarded_stdout_sha256,
        credential_redaction_count=redaction_count,
    )


def _capture_status(
    process: ProcessResult, credential_redaction_count: int
) -> AgentCaptureStatus:
    if credential_redaction_count:
        return AgentCaptureStatus.CREDENTIAL_REDACTED
    if not process.cleanup_succeeded:
        return AgentCaptureStatus.CLEANUP_FAILED
    if process.stdout_truncated or process.stderr_truncated:
        return AgentCaptureStatus.OUTPUT_TRUNCATED
    return AgentCaptureStatus.COMPLETE


def _validate_execution_authority(
    request: AgentRequest,
    backend: AgentBackendName,
    credential_environment_key: str,
    prompt_transport: str,
    argv: list[str],
    stdin_text: str | None,
) -> None:
    authority = request.execution_authority
    if authority is None:
        raise ContractError(
            "Formal agent execution authority is missing",
            "agent_execution_authority_missing",
        )
    expected = (
        authority.run_id == request.run_id,
        authority.attempt_id == request.attempt_id,
        authority.backend == request.backend.value == backend.value,
        credential_environment_key == _BACKEND_CREDENTIAL_KEYS[backend],
        authority.workspace == str(request.workspace),
        authority.allowed_files == request.allowed_files,
        authority.requested_environment_keys
        == tuple(sorted(request.environment)),
        prompt_transport == "stdin",
        stdin_text == request.prompt,
        not any(value == request.prompt for value in argv),
    )
    if not all(expected):
        raise ContractError(
            "Formal agent execution authority does not bind this invocation",
            "agent_execution_authority_mismatch",
        )


def _secret_values(
    environment: Mapping[str, str], credential_environment_key: str
) -> tuple[str, ...]:
    value = environment.get(credential_environment_key)
    return (value,) if value else ()


def _termination(
    process: ProcessResult, budget: AgentTurnBudget
) -> tuple[AgentTerminationKind, str | None]:
    if process.timed_out:
        return AgentTerminationKind.TIMEOUT, "agent_process_timeout"
    if budget.termination_kind is not None:
        assert budget.termination_reason is not None
        return budget.termination_kind, budget.termination_reason
    if process.exit_code != 0:
        return AgentTerminationKind.PROCESS_FAILED, "agent_process_failed"
    return AgentTerminationKind.COMPLETED, None
