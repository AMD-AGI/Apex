"""Direct-argv bounded subprocess runner for evaluator-owned tools."""

from __future__ import annotations

from apex.execution import SubprocessSupervisor
from apex.ports import SafetyToolRunRequest, SafetyToolRunResult


class SubprocessSafetyToolRunner:
    """Run one verification tool without a shell or inherited agent env.

    This adapter supplies process-group timeout/cleanup and concurrent bounded
    pipe draining.  Container/image isolation remains the responsibility of a
    concrete Magpie verification backend; its exact identity must still appear
    in the evaluator-owned, checksummed tool report consumed by
    :class:`SafetyGate`.
    """

    def __init__(self, *, kill_grace_seconds: float = 2.0) -> None:
        self._kill_grace_seconds = kill_grace_seconds

    def run(self, request: SafetyToolRunRequest) -> SafetyToolRunResult:
        supervisor = SubprocessSupervisor(
            max_output_bytes=request.output_limit_bytes,
            kill_grace_seconds=self._kill_grace_seconds,
        )
        result = supervisor.run(
            request.argv,
            cwd=request.cwd,
            environment=dict(request.environment),
            timeout_seconds=request.timeout_seconds,
        )
        return SafetyToolRunResult(
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            stdout=result.stdout,
            stderr=result.stderr,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            duration_seconds=result.duration_seconds,
            report_path=request.report_path if request.report_path.is_file() else None,
            report_origin="evaluator",
        )


__all__ = ["SubprocessSafetyToolRunner"]
