"""Trace analysis boundary implemented by TraceLens adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(frozen=True, slots=True)
class DiagnosticsRequest:
    run_id: str
    benchmark_workspace: Path
    output_dir: Path
    provenance_hash: str


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    run_id: str
    succeeded: bool
    artifacts: tuple[Path, ...]
    summary: Mapping[str, object]
    error: str | None = None


class DiagnosticsPort(Protocol):
    def analyze(self, request: DiagnosticsRequest) -> DiagnosticsResult: ...
