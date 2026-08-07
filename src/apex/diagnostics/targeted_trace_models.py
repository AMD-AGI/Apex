"""Immutable contracts shared by targeted-trace validation stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError


SCHEMA_NAME = "magpie.targeted-kernel-trace"
SCHEMA_VERSION = "1.0.0"
ZERO_CHECKSUM = "0" * 64
MAX_JSONL_LINE_BYTES = 16 * 1024 * 1024
ENVELOPE_TYPES = frozenset({"header", "event", "end"})
ARTIFACT_KINDS = frozenset(
    {
        "benchmark_report",
        "gap_analysis_csv",
        "targeted_manifest",
        "targeted_summary",
        "targeted_shard",
        "tracelens_artifact",
    }
)


def require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or value.lower() != value:
        raise ContractError(
            f"{field} must be a lowercase SHA-256 digest", "invalid_digest"
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise ContractError(
            f"{field} must be a lowercase SHA-256 digest", "invalid_digest"
        ) from error


def checked_sha256(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise IntegrityError(f"{field} is malformed", "invalid_digest")
    try:
        require_sha256(value, field)
    except ContractError as error:
        raise IntegrityError(f"{field} is malformed", "invalid_digest") from error
    return value


def strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{field} must be an integer", "invalid_targeted_trace")
    return value


def strict_nonnegative_int(value: object, field: str) -> int:
    result = strict_int(value, field)
    if result < 0:
        raise IntegrityError(f"{field} cannot be negative", "invalid_targeted_trace")
    return result


def nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(
            f"{field} must be a non-empty string", "invalid_targeted_trace"
        )
    return value


@dataclass(frozen=True, slots=True)
class AcquisitionCoverage:
    """Loss accounting shared by the manifest, shard sentinel, and evidence."""

    seen: int
    sampled: int
    written: int
    dropped: int
    dropped_by_reason: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        values = (self.seen, self.sampled, self.written, self.dropped)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ContractError(
                "Coverage counters must be non-negative integers",
                "invalid_trace_coverage",
            )
        reasons: set[str] = set()
        reason_total = 0
        for reason, count in self.dropped_by_reason:
            if (
                not reason.strip()
                or reason in reasons
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ContractError("Invalid drop accounting", "invalid_trace_coverage")
            reasons.add(reason)
            reason_total += count
        if reason_total != self.dropped:
            raise ContractError(
                "dropped_by_reason must sum to dropped", "invalid_trace_coverage"
            )
        if self.seen != self.written + self.dropped:
            raise ContractError(
                "seen must equal written + dropped", "invalid_trace_coverage"
            )
        unsampled = dict(self.dropped_by_reason).get("sampling", 0)
        if self.sampled != self.written + self.dropped - unsampled:
            raise ContractError(
                "sampled must equal written + non-sampling drops",
                "invalid_trace_coverage",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "sampled": self.sampled,
            "written": self.written,
            "dropped": self.dropped,
            "dropped_by_reason": dict(self.dropped_by_reason),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AcquisitionCoverage":
        def coverage_int(raw: object, field: str) -> int:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError(field)
            return raw

        try:
            reasons = value["dropped_by_reason"]
            if not isinstance(reasons, Mapping):
                raise TypeError("dropped_by_reason")
            return cls(
                coverage_int(value["seen"], "seen"),
                coverage_int(value["sampled"], "sampled"),
                coverage_int(value["written"], "written"),
                coverage_int(value["dropped"], "dropped"),
                tuple(
                    sorted(
                        (str(reason), coverage_int(count, f"drop:{reason}"))
                        for reason, count in reasons.items()
                    )
                ),
            )
        except ContractError as error:
            raise IntegrityError(
                "Malformed TargetedKernelTrace coverage", error.reason_code
            ) from error
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError(
                "Malformed TargetedKernelTrace coverage", "invalid_trace_coverage"
            ) from error

    @classmethod
    def aggregate(
        cls, values: tuple["AcquisitionCoverage", ...]
    ) -> "AcquisitionCoverage":
        reasons: dict[str, int] = {}
        for value in values:
            for reason, count in value.dropped_by_reason:
                reasons[reason] = reasons.get(reason, 0) + count
        return cls(
            sum(value.seen for value in values),
            sum(value.sampled for value in values),
            sum(value.written for value in values),
            sum(value.dropped for value in values),
            tuple(sorted(reasons.items())),
        )


@dataclass(frozen=True, slots=True)
class EvidenceArtifactReceipt:
    """Typed, relocatable receipt for one diagnostic input artifact."""

    kind: str
    relative_path: str
    sha256: str
    byte_count: int
    media_type: str

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_KINDS:
            raise ContractError(
                "Unknown diagnostic artifact kind", "invalid_artifact_receipt"
            )
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ContractError(
                "Artifact path must be workspace-relative", "invalid_artifact_receipt"
            )
        require_sha256(self.sha256, "artifact sha256")
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
        ):
            raise ContractError(
                "Artifact byte_count is invalid", "invalid_artifact_receipt"
            )
        if not self.media_type.strip():
            raise ContractError(
                "Artifact media type is required", "invalid_artifact_receipt"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceArtifactReceipt":
        try:
            byte_count = value["byte_count"]
            if isinstance(byte_count, bool) or not isinstance(byte_count, int):
                raise TypeError("byte_count")
            return cls(
                kind=str(value["kind"]),
                relative_path=str(value["relative_path"]),
                sha256=str(value["sha256"]),
                byte_count=byte_count,
                media_type=str(value["media_type"]),
            )
        except (ContractError, KeyError, TypeError, ValueError) as error:
            raise IntegrityError(
                "Malformed diagnostic artifact receipt", "invalid_artifact_receipt"
            ) from error


@dataclass(frozen=True, slots=True)
class ValidatedTargetedEvent:
    """One event yielded only after its complete shard has validated."""

    payload: Mapping[str, Any]
    payload_sha256: str
    shard_relative_path: str
    sequence: int


@dataclass(frozen=True, slots=True)
class ShardReceipt:
    path: Path
    relative_path: str
    rank: int
    pid: int
    sequence_end: int
    chain_checksum: str
    file_sha256: str
    byte_count: int
    coverage: AcquisitionCoverage
    complete: bool


@dataclass(frozen=True, slots=True)
class ValidatedTargetedTrace:
    schema_name: str
    schema_version: str
    run_id: str
    acquisition_backend: str
    coverage: AcquisitionCoverage
    artifacts: tuple[EvidenceArtifactReceipt, ...]
    warnings: tuple[str, ...]


__all__ = [
    "AcquisitionCoverage",
    "EvidenceArtifactReceipt",
    "MAX_JSONL_LINE_BYTES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "ShardReceipt",
    "ValidatedTargetedEvent",
    "ValidatedTargetedTrace",
    "ZERO_CHECKSUM",
    "checked_sha256",
    "nonempty_text",
    "strict_nonnegative_int",
]
