"""Measurement boundary implemented only by the pinned Magpie adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol


class BenchmarkPass(str, Enum):
    """Profiler-off measurements and profiler-on diagnostics never mix."""

    MEASUREMENT = "measurement"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    run_id: str
    config_path: Path
    output_dir: Path
    pass_type: BenchmarkPass
    timeout_seconds: int = 5400
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    run_id: str
    pass_type: BenchmarkPass
    succeeded: bool
    report_path: Path | None
    workspace_path: Path
    metrics: Mapping[str, float | int | str | None]
    artifact_paths: tuple[Path, ...] = ()
    error: str | None = None


class BenchmarkPort(Protocol):
    def run(self, request: BenchmarkRequest) -> BenchmarkResult: ...
