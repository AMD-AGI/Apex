"""Advisory kernel analysis boundary; never a grading or reward authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from apex.core import ContractError


@dataclass(frozen=True, slots=True)
class KernelDiagnosticCommand:
    argv: tuple[str, ...]
    cwd: str
    env: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.argv or any(not value for value in self.argv):
            raise ContractError(
                "Diagnostic command is empty", "invalid_kernel_diagnostic"
            )
        if not self.cwd or Path(self.cwd).is_absolute() or ".." in Path(self.cwd).parts:
            raise ContractError(
                "Diagnostic cwd is invalid", "invalid_kernel_diagnostic"
            )


@dataclass(frozen=True, slots=True)
class KernelDiagnosticRequest:
    run_id: str
    attempt_id: str
    mode: str
    kernel_type: str
    source_files: tuple[str, ...]
    candidate_root: Path
    baseline_root: Path | None
    output_root: Path
    compile: KernelDiagnosticCommand
    correctness: KernelDiagnosticCommand
    performance: KernelDiagnosticCommand
    timeout_seconds: int

    def __post_init__(self) -> None:
        if self.mode not in {"analyze", "compare"}:
            raise ContractError(
                "Diagnostic mode is invalid", "invalid_kernel_diagnostic"
            )
        roots = (self.candidate_root, self.output_root)
        if any(not root.is_absolute() for root in roots) or not self.source_files:
            raise ContractError(
                "Diagnostic paths are invalid", "invalid_kernel_diagnostic"
            )
        if self.output_root.is_relative_to(self.candidate_root):
            raise ContractError(
                "Diagnostic output is candidate-visible", "invalid_kernel_diagnostic"
            )
        if any(
            Path(path).is_absolute() or ".." in Path(path).parts
            for path in self.source_files
        ):
            raise ContractError(
                "Diagnostic source path is invalid", "invalid_kernel_diagnostic"
            )
        if self.mode == "compare" and self.baseline_root is None:
            raise ContractError(
                "Compare requires a baseline", "invalid_kernel_diagnostic"
            )
        if self.baseline_root is not None and not self.baseline_root.is_absolute():
            raise ContractError(
                "Diagnostic baseline is invalid", "invalid_kernel_diagnostic"
            )
        if self.timeout_seconds <= 0:
            raise ContractError(
                "Diagnostic timeout is invalid", "invalid_kernel_diagnostic"
            )


@dataclass(frozen=True, slots=True)
class KernelDiagnosticOutput:
    adapter_id: str
    mode: str
    report_path: Path
    config_path: Path
    execution: Mapping[str, object]


class KernelDiagnosticsPort(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def run(self, request: KernelDiagnosticRequest) -> KernelDiagnosticOutput: ...


__all__ = [
    "KernelDiagnosticCommand",
    "KernelDiagnosticOutput",
    "KernelDiagnosticRequest",
    "KernelDiagnosticsPort",
]
