"""Trace analysis boundary implemented by TraceLens adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from apex.core import ContractError


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class DiagnosticsRequest:
    run_id: str
    benchmark_workspace: Path
    output_dir: Path
    provenance_hash: str
    preserve_raw_trace: bool = False


@dataclass(frozen=True, slots=True)
class DiagnosticsResult:
    run_id: str
    succeeded: bool
    artifacts: tuple[Path, ...]
    summary: Mapping[str, object]
    error: str | None = None
    artifact_roles: Mapping[str, str] = field(default_factory=dict)
    benchmark_workspace: Path | None = None


class DiagnosticsPort(Protocol):
    def analyze(self, request: DiagnosticsRequest) -> DiagnosticsResult: ...


class TraceComparisonStatus(str, Enum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TraceComparisonArtifact:
    """One CAS receipt plus the producer-relative name needed by TraceLens."""

    role: str
    logical_path: str
    digest: str
    size: int
    media_type: str
    receipt_relative_path: str

    def __post_init__(self) -> None:
        logical = PurePosixPath(self.logical_path)
        receipt = PurePosixPath(self.receipt_relative_path)
        if (
            not self.role
            or not self.media_type
            or not _SHA256.fullmatch(self.digest)
            or self.size < 0
            or logical.is_absolute()
            or not logical.parts
            or ".." in logical.parts
            or logical.as_posix() != self.logical_path
            or receipt.as_posix()
            != f"sha256/{self.digest[:2]}/{self.digest}"
        ):
            raise ContractError(
                "Trace comparison artifact binding is invalid",
                "invalid_trace_comparison_artifact",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "logical_path": self.logical_path,
            "receipt": {
                "digest": self.digest,
                "size": self.size,
                "media_type": self.media_type,
                "relative_path": self.receipt_relative_path,
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TraceComparisonArtifact":
        try:
            receipt = value["receipt"]
            if not isinstance(receipt, Mapping):
                raise TypeError("receipt")
            return cls(
                role=str(value["role"]),
                logical_path=str(value["logical_path"]),
                digest=str(receipt["digest"]),
                size=int(receipt["size"]),
                media_type=str(receipt["media_type"]),
                receipt_relative_path=str(receipt["relative_path"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Trace comparison artifact binding is malformed",
                "invalid_trace_comparison_artifact",
            ) from error


@dataclass(frozen=True, slots=True)
class TraceDiagnosticEvidence:
    """CAS-bound files from one profiler-on diagnostic observation."""

    trace_evidence_sha256: str | None
    cas_root: Path
    artifacts: tuple[TraceComparisonArtifact, ...]

    def __post_init__(self) -> None:
        if (
            (self.trace_evidence_sha256 is not None and not _SHA256.fullmatch(
                self.trace_evidence_sha256
            ))
            or not self.cas_root.is_absolute()
            or len({item.logical_path for item in self.artifacts}) != len(self.artifacts)
        ):
            raise ContractError(
                "Trace diagnostic evidence is invalid", "invalid_trace_comparison"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_evidence_sha256": self.trace_evidence_sha256,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, cas_root: Path
    ) -> "TraceDiagnosticEvidence":
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, list):
            raise ContractError(
                "Trace diagnostic evidence is malformed", "invalid_trace_comparison"
            )
        if any(not isinstance(item, Mapping) for item in artifacts):
            raise ContractError(
                "Trace diagnostic evidence is malformed", "invalid_trace_comparison"
            )
        digest = value.get("trace_evidence_sha256")
        return cls(
            None if digest is None else str(digest),
            cas_root,
            tuple(
                TraceComparisonArtifact.from_mapping(item)
                for item in artifacts
            ),
        )


@dataclass(frozen=True, slots=True)
class TraceComparisonRequest:
    run_id: str
    gpu_arch: str
    baseline: TraceDiagnosticEvidence
    terminal: TraceDiagnosticEvidence
    terminal_benchmark_sha256: str
    output_dir: Path

    def __post_init__(self) -> None:
        if (
            not self.run_id
            or not self.gpu_arch
            or self.baseline.trace_evidence_sha256 is None
            or not _SHA256.fullmatch(self.terminal_benchmark_sha256)
            or not self.output_dir.is_absolute()
        ):
            raise ContractError(
                "Trace comparison request is invalid", "invalid_trace_comparison"
            )


@dataclass(frozen=True, slots=True)
class TraceComparisonResult:
    status: TraceComparisonStatus
    reason_code: str
    summary: Mapping[str, object]
    reward_eligible: bool = False
    artifacts: tuple[Path, ...] = ()
    artifact_roles: Mapping[str, str] = field(default_factory=dict)
    output_root: Path | None = None

    def __post_init__(self) -> None:
        paths = tuple(path.resolve() for path in self.artifacts)
        if (
            not self.reason_code
            or self.reward_eligible
            or len(set(paths)) != len(paths)
            or bool(paths) != (self.output_root is not None)
            or (
                self.output_root is not None
                and (
                    not self.output_root.is_absolute()
                    or any(not path.is_relative_to(self.output_root.resolve()) for path in paths)
                )
            )
        ):
            raise ContractError(
                "Trace comparison cannot carry reward authority",
                "invalid_trace_comparison",
            )


class TraceComparisonPort(Protocol):
    def compare(self, request: TraceComparisonRequest) -> TraceComparisonResult: ...
