"""Ports for isolated, evaluator-owned safety verification.

The runner sees a frozen argv request.  It does not receive a shell command or
an agent-controlled policy object.  Concrete Magpie/sidecar adapters implement
``SafetyToolRunner``; orchestration consumes ``SafetyVerificationPort``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar


@dataclass(frozen=True, slots=True)
class SafetyToolRunRequest:
    tool: str
    plan_fingerprint: str
    argv: tuple[str, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]
    timeout_seconds: int
    output_limit_bytes: int
    report_path: Path
    candidate_root: Path
    artifact_root: Path


@dataclass(frozen=True, slots=True)
class SafetyToolRunResult:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: float
    report_path: Path | None
    report_origin: str = "evaluator"


class SafetyToolRunner(Protocol):
    def run(self, request: SafetyToolRunRequest) -> SafetyToolRunResult: ...


SafetyRequestT = TypeVar("SafetyRequestT", contravariant=True)
SafetyResultT = TypeVar("SafetyResultT", covariant=True)


class SafetyVerificationPort(Protocol[SafetyRequestT, SafetyResultT]):
    def evaluate(self, request: SafetyRequestT) -> SafetyResultT: ...


__all__ = [
    "SafetyToolRunRequest",
    "SafetyToolRunResult",
    "SafetyToolRunner",
    "SafetyVerificationPort",
]
