"""Fail-closed lifecycle helpers around one prepared Magpie observer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from apex.execution import ProcessResult
from apex.ports import MagpieExecutionAttestor

from .magpie_launch_argv import validated_magpie_launch_argv


class _Supervisor(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
    ) -> ProcessResult: ...


def abort_magpie_execution(
    attestor: MagpieExecutionAttestor, session: object, reason: str
) -> str:
    """Close an observer and preserve both the trigger and cleanup failure."""

    try:
        attestor.abort(session, reason=reason)
    except Exception as error:
        return (
            f"{reason};magpie_execution_attestor_abort_failed:"
            f"{type(error).__name__}:{error}"
        )
    return reason


def prepare_magpie_execution(
    attestor: MagpieExecutionAttestor,
    prepare: Callable[[], object],
    canonical_argv: tuple[str, ...],
) -> tuple[object | None, tuple[str, ...] | None, str | None]:
    """Prepare and validate launch argv, aborting any returned session on error."""

    session: object | None = None
    try:
        session = prepare()
        launch = attestor.launch_argv(session)
        return session, validated_magpie_launch_argv(canonical_argv, launch), None
    except Exception as error:
        reason = (
            "magpie_execution_attestor_prepare_failed:"
            f"{type(error).__name__}:{error}"
        )
        if session is not None:
            reason = abort_magpie_execution(attestor, session, reason)
        return None, None, reason


def run_magpie_execution(
    attestor: MagpieExecutionAttestor,
    supervisor: _Supervisor,
    session: object,
    launch_argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> tuple[ProcessResult | None, str | None]:
    """Start Magpie or abort its prepared observer if process creation fails."""

    try:
        process = supervisor.run(
            launch_argv,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:
        reason = f"magpie_process_start_failed:{type(error).__name__}:{error}"
        return None, abort_magpie_execution(attestor, session, reason)
    return process, None


__all__ = ["prepare_magpie_execution", "run_magpie_execution"]
